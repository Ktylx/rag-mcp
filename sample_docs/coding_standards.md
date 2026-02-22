# Coding Standards

## Overview

This document outlines the coding standards and best practices for our project.

## Python Style Guide

### Naming Conventions

- **Variables**: `snake_case` (e.g., `user_name`, `total_count`)
- **Functions**: `snake_case` (e.g., `get_user()`, `calculate_total()`)
- **Classes**: `PascalCase` (e.g., `UserModel`, `DataProcessor`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRY_COUNT`, `DEFAULT_TIMEOUT`)

### Docstrings

All public functions and classes must have docstrings. Use Google style:

```python
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers.
    
    Args:
        a: First number.
        b: Second number.
        
    Returns:
        The sum of a and b.
        
    Raises:
        ValueError: If a or b is negative.
    """
    if a < 0 or b < 0:
        raise ValueError("Numbers must be non-negative")
    return a + b
```

### Type Hints

Always use type hints for function parameters and return values:

```python
def process_data(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Process a list of data records.
    
    Args:
        data: List of data dictionaries.
        
    Returns:
        Processed results dictionary.
    """
    ...
```

## Code Organization

### Imports

Organize imports in the following order:
1. Standard library
2. Third-party libraries
3. Local application imports

```python
import os
import sys

from typing import List, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from myapp.models import User
from myapp.utils import helpers
```

### File Structure

```
src/
├── __init__.py
├── main.py           # Application entry point
├── config.py         # Configuration
├── models/           # Data models
├── routes/          # API routes
├── services/        # Business logic
└── utils/           # Helper functions
```

## Testing

### Unit Tests

- Test files should be named `test_<module>.py`
- Use pytest framework
- Aim for 80% code coverage

```python
def test_calculate_sum():
    """Test sum calculation."""
    assert calculate_sum(2, 3) == 5
    assert calculate_sum(-1, 1) == 0
```

### Integration Tests

- Test file should be named `test_integration_<feature>.py`
- Test complete workflows
- Use fixtures for common setup

## Error Handling

- Use specific exception types
- Log errors with appropriate levels
- Return user-friendly error messages

```python
try:
    result = await fetch_data()
except DataNotFoundError as e:
    logger.warning(f"Data not found: {e}")
    raise HTTPException(status_code=404, detail="Data not found")
except NetworkError as e:
    logger.error(f"Network error: {e}")
    raise HTTPException(status_code=503, detail="Service unavailable")
```

## Performance

- Use async/await for I/O operations
- Implement caching where appropriate
- Profile code before optimizing

## Security

- Never commit secrets to version control
- Validate all user inputs
- Use parameterized queries for database operations