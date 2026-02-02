"""Calculator tool implementation.

This module provides safe mathematical calculation functionality.
"""

import math
import operator
import re
from typing import Any

import structlog

from agent_engine.exceptions import ToolExecutionError

logger = structlog.get_logger()

# Allowed operators for safe evaluation
OPERATORS: dict[str, Any] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "^": operator.pow,
}

# Allowed mathematical functions
FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": math.pow,
}

# Constants
CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def calculate(expression: str) -> dict[str, Any]:
    """Perform a mathematical calculation.

    This function safely evaluates mathematical expressions without using eval().
    It supports basic arithmetic operations, common math functions, and constants.

    Supported operations:
        - Arithmetic: +, -, *, /, //, %, ** (or ^)
        - Functions: abs, round, min, max, sum, sqrt, sin, cos, tan, log, log10, exp, floor, ceil
        - Constants: pi, e, tau

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4", "sqrt(16)", "sin(pi/2)")

    Returns:
        Dictionary containing:
            - expression: The original expression
            - result: The calculated result
            - type: The type of the result (int or float)

    Raises:
        ToolExecutionError: If the expression is invalid or calculation fails

    Examples:
        >>> calculate("2 + 3 * 4")
        {"expression": "2 + 3 * 4", "result": 14, "type": "int"}

        >>> calculate("sqrt(16) + pi")
        {"expression": "sqrt(16) + pi", "result": 7.141592653589793, "type": "float"}
    """
    logger.info("calculate_executed", expression=expression)

    try:
        result = _safe_eval(expression)

        # Determine result type
        result_type = "int" if isinstance(result, int) else "float"

        return {
            "expression": expression,
            "result": result,
            "type": result_type,
        }
    except ZeroDivisionError as e:
        raise ToolExecutionError(
            "Division by zero",
            tool_name="calculate",
            tool_args={"expression": expression},
            details={"error": str(e)},
        ) from e
    except ValueError as e:
        raise ToolExecutionError(
            f"Invalid mathematical expression: {e}",
            tool_name="calculate",
            tool_args={"expression": expression},
            details={"error": str(e)},
        ) from e
    except Exception as e:
        raise ToolExecutionError(
            f"Calculation failed: {e}",
            tool_name="calculate",
            tool_args={"expression": expression},
            details={"error_type": type(e).__name__},
        ) from e


def _safe_eval(expression: str) -> int | float:
    """Safely evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Calculated result
    """
    # Clean the expression
    expr = expression.strip()

    # Replace constants
    for const_name, const_value in CONSTANTS.items():
        expr = re.sub(rf"\b{const_name}\b", str(const_value), expr, flags=re.IGNORECASE)

    # Handle function calls
    for func_name, func in FUNCTIONS.items():
        pattern = rf"{func_name}\s*\(\s*([^)]+)\s*\)"
        while re.search(pattern, expr, re.IGNORECASE):
            match = re.search(pattern, expr, re.IGNORECASE)
            if match:
                args_str = match.group(1)
                # Recursively evaluate the argument
                args = [_safe_eval(arg.strip()) for arg in args_str.split(",")]
                result = func(*args)
                expr = expr[: match.start()] + str(result) + expr[match.end() :]

    # Replace ^ with ** for power
    expr = expr.replace("^", "**")

    # Parse and evaluate the expression
    return _parse_expression(expr)


def _parse_expression(expr: str) -> int | float:
    """Parse and evaluate a simple arithmetic expression.

    Args:
        expr: Arithmetic expression with numbers and operators

    Returns:
        Calculated result
    """
    expr = expr.strip()

    # Handle parentheses first
    while "(" in expr:
        # Find innermost parentheses
        start = expr.rfind("(")
        end = expr.find(")", start)
        if end == -1:
            raise ValueError("Mismatched parentheses")
        inner = expr[start + 1 : end]
        result = _parse_expression(inner)
        expr = expr[:start] + str(result) + expr[end + 1 :]

    # Handle negative numbers at the start
    expr = expr.strip()
    if expr.startswith("-"):
        expr = "0" + expr

    # Tokenize
    tokens = _tokenize(expr)

    # Apply operator precedence
    # First pass: ** (power)
    tokens = _apply_operators(tokens, ["**"])
    # Second pass: *, /, //, %
    tokens = _apply_operators(tokens, ["*", "/", "//", "%"])
    # Third pass: +, -
    tokens = _apply_operators(tokens, ["+", "-"])

    if len(tokens) != 1:
        raise ValueError(f"Invalid expression: {expr}")

    result = tokens[0]
    # Convert to int if it's a whole number
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def _tokenize(expr: str) -> list[Any]:
    """Tokenize an arithmetic expression.

    Args:
        expr: Expression to tokenize

    Returns:
        List of tokens (numbers and operators)
    """
    tokens: list[Any] = []
    current_num = ""

    i = 0
    while i < len(expr):
        char = expr[i]

        if char.isdigit() or char == ".":
            current_num += char
        elif char in "+-*/%":
            if current_num:
                tokens.append(float(current_num))
                current_num = ""

            # Check for ** operator
            if char == "*" and i + 1 < len(expr) and expr[i + 1] == "*":
                tokens.append("**")
                i += 1
            # Check for // operator
            elif char == "/" and i + 1 < len(expr) and expr[i + 1] == "/":
                tokens.append("//")
                i += 1
            else:
                tokens.append(char)
        elif char == " ":
            if current_num:
                tokens.append(float(current_num))
                current_num = ""
        else:
            raise ValueError(f"Invalid character in expression: {char}")

        i += 1

    if current_num:
        tokens.append(float(current_num))

    return tokens


def _apply_operators(tokens: list[Any], operators: list[str]) -> list[Any]:
    """Apply operators to token list.

    Args:
        tokens: List of tokens
        operators: List of operators to apply

    Returns:
        Reduced token list
    """
    result: list[Any] = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if isinstance(token, str) and token in operators and result:
            left = result.pop()
            right = tokens[i + 1]
            op_func = OPERATORS[token]
            result.append(op_func(left, right))
            i += 2
        else:
            result.append(token)
            i += 1

    return result
