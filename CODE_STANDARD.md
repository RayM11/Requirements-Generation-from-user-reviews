# Code Standard

## PEP 8 Standard

This project follows the [PEP 8](https://peps.python.org/pep-0008/) coding standard, which is the official style guide for Python code. Below are key aspects of PEP 8 that we adhere to:

### Code Layout

- Use 4 spaces per indentation level
- Limit all lines to a maximum of 79 characters
- Surround top-level function and class definitions with two blank lines
- Method definitions inside a class are surrounded by a single blank line

### Imports

Imports should be grouped in the following order:
1. Standard library imports
2. Related third-party imports
3. Local application/library specific imports

Each group should be separated by a blank line.

### Naming Conventions

- Function names, variable names, and module names: `lowercase_with_underscores` (snake_case)
- Class names: `CapitalizedWords` (CamelCase)
- Constants: `ALL_CAPITAL_LETTERS_WITH_UNDERSCORES`

### Docstrings

All public modules, functions, classes, and methods should have docstrings.
We use the reStructuredText (reST) format for consistency:

```python
def example_function(param1, param2):
    """Summary line of the function.
    
    More detailed description if needed.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
    
    Returns:
        type: Description of return value
    """