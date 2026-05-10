from src.agent.code_cleaning import extract_python_code, is_valid_python

raw = """
To solve this problem, we need to write Python code.

### Solution Code
```python
def add(a, b):
    return a + b

print(add(1, 2))
Explanation

This function adds two numbers.
"""

code = extract_python_code(raw)

print("=== Extracted Code ===")
print(code)
print("=== Valid Python ===")
print(is_valid_python(code))
