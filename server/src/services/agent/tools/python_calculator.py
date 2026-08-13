import math
import logging
from langchain_core.tools import tool
from server.src.services.agent.tools.base_tools import Tools

logger = logging.getLogger("agent.tools.python_calculator")

@tool
def python_calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression (e.g. '125 * 4.5', 'math.sqrt(144)', 'sum([10, 20, 30])').

    Args:
        expression: A string mathematical expression to evaluate.

    Returns:
        The calculated numerical result as a string.
    """
    allowed_names = {
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "int": int,
        "float": float,
    }
    try:
        logger.info("Evaluating calculator expression: %r", expression)
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as exc:
        logger.exception("Calculator evaluation failed for expression %r: %s", expression, exc)
        return f"Calculation error: {str(exc)}"


class PythonCalculatorTool(Tools):
    cls_tool = python_calculator
