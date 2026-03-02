#!/usr/bin/env python3
"""
MCP Client - CLI приложение для взаимодействия с RAG MCP сервером.
Поддерживает SSE (Server-Sent Events) транспорт.
"""
import json
import uuid
import sys
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    print("Требуется установить requests: pip install requests")
    sys.exit(1)


class MCPClient:
    """Клиент для MCP сервера с поддержкой SSE."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
        self.session_id: Optional[str] = None
        self.server_info: Optional[Dict] = None
        self._endpoint = None
    
    def _parse_sse(self, text: str) -> Dict[str, Any]:
        """Парсит SSE формат."""
        for line in text.strip().split('\n'):
            if line.startswith('data: '):
                data = line[6:]  # Убираем 'data: '
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    continue
        return {}
    
    def _send_request(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Отправить JSON-RPC запрос к MCP серверу с SSE."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": str(uuid.uuid4())
        }
        if params:
            request["params"] = params
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        if self.session_id:
            headers["MCP-Session-Id"] = self.session_id
        
        # FastMCP HTTP использует /sse для SSE соединений
        # Пробуем /sse затем /mcp
        endpoints = ["/sse", "/mcp"]
        
        last_error = None
        for endpoint in endpoints:
            try:
                response = requests.post(
                    f"{self.server_url}{endpoint}",
                    json=request,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 404:
                    last_error = f"Endpoint {endpoint} not found (404)"
                    continue
                    
                response.raise_for_status()
                
                # Сохраняем найденный endpoint
                self._endpoint = endpoint
                logger.info(f"Using MCP endpoint: {endpoint}")
                
                # Сохраняем session ID из заголовков
                if "mcp-session-id" in response.headers:
                    self.session_id = response.headers["mcp-session-id"]
                
                content_type = response.headers.get("content-type", "")
                
                # Если это SSE (text/event-stream)
                if "text/event-stream" in content_type:
                    # Читаем SSE поток
                    full_text = ""
                    for line in response.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            full_text += decoded + '\n'
                            if decoded.startswith('data: '):
                                break  # Получили первое сообщение
                    # Парсим SSE
                    return self._parse_sse(full_text)
                else:
                    # Обычный JSON ответ
                    return response.json()
                    
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                continue
        
        return {"error": f"Failed to connect to any MCP endpoint. Last error: {last_error}"}
    
    def initialize(self) -> bool:
        """Инициализировать соединение с сервером."""
        result = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "mcp-client-cli",
                "version": "1.0.0"
            }
        })
        
        if "error" in result:
            error_data = result.get("error", {})
            if isinstance(error_data, dict):
                error_msg = error_data.get("message", str(result))
            else:
                error_msg = str(error_data)
            print(f"Ошибка инициализации: {error_msg}")
            return False
        
        self.server_info = result.get("result", {})
        server_name = self.server_info.get('serverInfo', {}).get('name', 'unknown')
        server_version = self.server_info.get('serverInfo', {}).get('version', '')
        print(f"✓ Подключено к серверу: {server_name} ({server_version})")
        return True
    
    def list_tools(self) -> list:
        """Получить список доступных инструментов."""
        result = self._send_request("tools/list")
        
        if "error" in result:
            print(f"Ошибка получения списка инструментов: {result['error']}")
            return []
        
        tools = result.get("result", {}).get("tools", [])
        return tools
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Вызвать инструмент на сервере."""
        result = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        return result


# Простой логгер
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_tool_info(tools: list):
    """Вывести информацию об инструментах."""
    print("\n" + "=" * 60)
    print("Доступные инструменты:")
    print("=" * 60)
    
    for i, tool in enumerate(tools, 1):
        print(f"\n{i}. {tool['name']}")
        print(f"   Описание: {tool.get('description', 'Нет описания')}")
        
        # Вывести параметры
        input_schema = tool.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        if properties:
            print("   Параметры:")
            for param_name, param_info in properties.items():
                required_mark = " (обязательный)" if param_name in required else ""
                print(f"     - {param_name}{required_mark}: {param_info.get('description', '')}")


def get_tool_arguments(tool_name: str, input_schema: Dict) -> Dict[str, Any]:
    """Получить аргументы от пользователя для инструмента."""
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    arguments = {}
    
    if not properties:
        return {}
    
    print("\nВведите значения параметров:")
    
    for param_name, param_info in properties.items():
        is_required = param_name in required
        default = param_info.get("default")
        
        prompt = f"  {param_name}"
        if is_required:
            prompt += " (обязательный)"
        if default is not None:
            prompt += f" [по умолчанию: {default}]"
        prompt += ": "
        
        value = input(prompt).strip()
        
        if not value:
            if default is not None:
                arguments[param_name] = default
            elif is_required:
                print(f"    Ошибка: {param_name} обязателен")
                return {}
        else:
            # Простое преобразование типов
            param_type = param_info.get("type", "string")
            if param_type == "integer":
                try:
                    arguments[param_name] = int(value)
                except ValueError:
                    print(f"    Ошибка: ожидалось целое число")
                    return {}
            elif param_type == "number":
                try:
                    arguments[param_name] = float(value)
                except ValueError:
                    print(f"    Ошибка: ожидалось число")
                    return {}
            elif param_type == "boolean":
                arguments[param_name] = value.lower() in ("true", "yes", "1", "y")
            else:
                arguments[param_name] = value
    
    return arguments


def main():
    """Основная функция CLI приложения."""
    print("=" * 60)
    print("MCP Client для RAG сервера")
    print("=" * 60)
    
    # Создаём клиент
    client = MCPClient()
    
    # Инициализируем соединение
    print("\nПодключение к серверу...")
    if not client.initialize():
        sys.exit(1)
    
    # Получаем список инструментов
    print("Получение списка инструментов...")
    tools = client.list_tools()
    
    if not tools:
        print("Не удалось получить список инструментов")
        sys.exit(1)
    
    # Выводим информацию об инструментах
    print_tool_info(tools)
    
    # Главный цикл
    while True:
        print("\n" + "=" * 60)
        print("Выберите действие:")
        print("  0 - Выйти")
        for i, tool in enumerate(tools, 1):
            print(f"  {i} - {tool['name']}")
        print("  l - Показать список инструментов")
        
        choice = input("\n> ").strip()
        
        if choice == "0" or choice.lower() in ("q", "quit", "exit"):
            print("До свидания!")
            break
        
        if choice.lower() == "l":
            print_tool_info(tools)
            continue
        
        try:
            tool_index = int(choice) - 1
            if tool_index < 0 or tool_index >= len(tools):
                print("Неверный выбор")
                continue
        except ValueError:
            print("Неверный выбор")
            continue
        
        # Выбран инструмент
        selected_tool = tools[tool_index]
        tool_name = selected_tool["name"]
        
        print(f"\nВыбран инструмент: {tool_name}")
        
        # Получаем аргументы от пользователя
        input_schema = selected_tool.get("inputSchema", {})
        arguments = get_tool_arguments(tool_name, input_schema)
        
        if not arguments and input_schema.get("properties"):
            print("Не удалось получить аргументы")
            continue
        
        # Вызываем инструмент
        print("\nВызов инструмента...")
        result = client.call_tool(tool_name, arguments)
        
        # Выводим результат
        print("\n" + "-" * 40)
        print("Результат:")
        print("-" * 40)
        
        if "error" in result:
            print(f"Ошибка: {result['error']}")
        else:
            # Красивый вывод результата
            output = result.get("result", {})
            if isinstance(output, dict):
                print(json.dumps(output, indent=2, ensure_ascii=False))
            else:
                print(output)
        
        input("\nНажмите Enter для продолжения...")


if __name__ == "__main__":
    main()