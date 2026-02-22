"""
Example Python module demonstrating various code patterns.
"""


class DataProcessor:
    """A class for processing data."""
    
    def __init__(self, name: str):
        """Initialize the data processor.
        
        Args:
            name: Name of the processor.
        """
        self.name = name
        self.data = []
    
    def add(self, item: str) -> None:
        """Add an item to the processor.
        
        Args:
            item: Item to add.
        """
        self.data.append(item)
    
    def process(self) -> list[str]:
        """Process all items.
        
        Returns:
            List of processed items.
        """
        return [item.upper() for item in self.data]
    
    def clear(self) -> None:
        """Clear all data."""
        self.data = []


def calculate_statistics(numbers: list[float]) -> dict[str, float]:
    """Calculate basic statistics for a list of numbers.
    
    Args:
        numbers: List of numbers.
        
    Returns:
        Dictionary with min, max, and average.
    """
    if not numbers:
        return {"min": 0, "max": 0, "avg": 0}
    
    return {
        "min": min(numbers),
        "max": max(numbers),
        "avg": sum(numbers) / len(numbers),
    }


def fetch_user(user_id: int) -> dict[str, any]:
    """Fetch a user by ID.
    
    Args:
        user_id: User ID.
        
    Returns:
        User data dictionary.
    """
    # Mock implementation
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
    }


async def async_fetch_data(url: str) -> dict[str, any]:
    """Asynchronously fetch data from URL.
    
    Args:
        url: URL to fetch from.
        
    Returns:
        Response data.
    """
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()


def main():
    """Main entry point."""
    processor = DataProcessor("test")
    processor.add("hello")
    processor.add("world")
    
    result = processor.process()
    print(f"Processed: {result}")
    
    stats = calculate_statistics([1, 2, 3, 4, 5])
    print(f"Statistics: {stats}")
    
    user = fetch_user(1)
    print(f"User: {user}")


if __name__ == "__main__":
    main()