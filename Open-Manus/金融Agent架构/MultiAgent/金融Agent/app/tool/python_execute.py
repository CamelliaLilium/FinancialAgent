"""
PythonExecute - 持久化命名空间的 Python 执行工具，内置金融变量命名规范。

支持多步计算：变量在多次调用间保留。每次 flow 开始时重置环境。
支持 REPL 风格：代码末尾的裸表达式（如 interest_expense, a, b）会自动 print 输出。

金融变量命名规范：
- 使用描述性 snake_case 命名（英文，Python兼容）
- 格式: <entity>_<metric>_<year>[_<context>]
- 示例: novartis_net_sales_2011, interest_expense_2022_notes
- 禁止: var_a, var_b, col1, value_1, x, y 等无意义命名
- 变量名不能以数字开头（用 net_profit_2024 而非 2024_net_profit）
"""
import ast
import re
import sys
from io import StringIO
from typing import Any, Dict, List, Tuple

from pydantic import PrivateAttr

from app.tool.base import BaseTool


# 金融变量命名规范
FINANCIAL_NAMING_GUIDELINES = """
# FINANCIAL VARIABLE NAMING STANDARDS (ENFORCED)
# 
# Format: <entity>_<metric>_<year>[_<context>]
# 
# GOOD EXAMPLES:
#   - novartis_net_sales_2011
#   - total_compensation_cost_2007
#   - interest_expense_2022_notes
#   - operating_income_2018
#   - ebit_2019_q3
# 
# BAD EXAMPLES (NEVER USE):
#   - var_a, var_b, col1, value_1
#   - x, y, temp, result
#   - 2024_net_profit (starts with number)
#
# RULES:
#   1. Use descriptive snake_case names in English
#   2. Must be valid Python identifiers
#   3. Include year/time period when relevant
#   4. Use entity name prefix for clarity (company, product, etc.)
"""

# 禁止的变量名模式
FORBIDDEN_PATTERNS = [
    r'^var_[a-z]$',           # var_a, var_b
    r'^col\d+$',              # col1, col2
    r'^value_\d+$',           # value_1, value_2
    r'^[xy]$',                # x, y
    r'^temp$',                # temp
    r'^result$',              # result
    r'^\d',                   # starts with number
]

# 推荐的命名模板
RECOMMENDED_TEMPLATES = {
    'net_sales': '{entity}_net_sales_{year}',
    'revenue': '{entity}_revenue_{year}',
    'operating_income': '{entity}_operating_income_{year}',
    'ebit': '{entity}_ebit_{year}',
    'interest_expense': '{entity}_interest_expense_{year}',
    'net_profit': '{entity}_net_profit_{year}',
    'net_income': '{entity}_net_income_{year}',
    'cogs': '{entity}_cogs_{year}',
    'total_assets': '{entity}_total_assets_{year}',
    'total_liabilities': '{entity}_total_liabilities_{year}',
    'equity': '{entity}_equity_{year}',
}


def _default_global_env() -> Dict[str, Any]:
    """构建初始 globals，含 __builtins__。"""
    if isinstance(__builtins__, dict):
        return {"__builtins__": __builtins__}
    return {"__builtins__": __builtins__.__dict__.copy()}


def _wrap_last_expression(code: str) -> str:
    """
    若代码最后一条是裸表达式（如 x 或 x, y），自动添加 print 以产生输出。
    解决「哑巴代码」：单写变量名在 exec() 中无输出。
    """
    code = code.strip()
    if not code:
        return code
    try:
        tree = ast.parse(code)
        if not tree.body:
            return code
        last = tree.body[-1]
        if isinstance(last, ast.Expr):
            # 裸表达式，需要 print
            try:
                expr_src = ast.unparse(last.value)
            except AttributeError:
                # Python < 3.9 无 unparse，用 compile+exec 提取
                return code
            return code + f"\nprint({expr_src})"
    except SyntaxError:
        pass
    return code


def _extract_assigned_variables(code: str) -> List[Tuple[str, int]]:
    """提取代码中赋值的变量名及其行号"""
    variables = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append((target.id, node.lineno))
    except SyntaxError:
        pass
    return variables


def _validate_variable_name(name: str) -> Tuple[bool, str]:
    """
    验证变量名是否符合金融命名规范
    
    Returns:
        (is_valid, message)
    """
    # 检查是否以数字开头
    if name[0].isdigit():
        return False, f"Variable '{name}' starts with a number. Use 'net_profit_2024' instead of '2024_net_profit'"
    
    # 检查禁止的模式
    for pattern in FORBIDDEN_PATTERNS:
        if re.match(pattern, name):
            return False, f"Variable '{name}' uses forbidden pattern. Use descriptive names like 'novartis_net_sales_2011'"
    
    # 检查是否太短（少于4个字符）
    if len(name) < 4:
        return False, f"Variable '{name}' is too short. Use descriptive names"
    
    return True, ""


class PythonExecute(BaseTool):
    """A tool for executing Python code with persistent namespace and financial naming standards."""

    name: str = "python_execute"
    description: str = (
        "Executes Python code for deterministic computation and data transformation. "
        "Use this tool when arithmetic, parsing, ratio/aggregation, or algorithmic processing is required. "
        "Variables PERSIST across calls within the same flow. "
        "Always use print() for output, or write a bare expression as the last line. "
        "\n\nFINANCIAL NAMING STANDARDS (ENFORCED):\n"
        "- Use descriptive snake_case names: novartis_net_sales_2011, interest_expense_2022\n"
        "- Format: <entity>_<metric>_<year>[_<context>]\n"
        "- NEVER use: var_a, var_b, col1, value_1, x, y, temp, result\n"
        "- NEVER start with numbers: use net_profit_2024 not 2024_net_profit\n"
        "\nWhen assigning values, use: variable = value  # Source: 'exact snippet'"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Variables persist. Follow financial naming: entity_metric_year (e.g., novartis_net_sales_2011)",
            },
        },
        "required": ["code"],
    }

    _global_env: Dict[str, Any] = PrivateAttr(default_factory=_default_global_env)

    def reset_env(self) -> None:
        """重置执行环境。Flow 开始时调用。"""
        self._global_env.clear()
        self._global_env.update(_default_global_env())

    def _validate_code(self, code: str) -> Tuple[bool, str]:
        """
        验证代码是否符合金融命名规范
        
        Returns:
            (is_valid, error_message)
        """
        variables = _extract_assigned_variables(code)
        
        for var_name, line_no in variables:
            is_valid, message = _validate_variable_name(var_name)
            if not is_valid:
                return False, f"Line {line_no}: {message}"
        
        return True, ""

    def _run_code(self, code: str) -> tuple[str, bool]:
        """执行代码，返回 (observation, success)。"""
        # 先验证命名规范
        is_valid, error_msg = self._validate_code(code)
        if not is_valid:
            return f"[NAMING ERROR] {error_msg}\n\nPlease fix variable names and retry.", False
        
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer

            # REPL 风格：末尾裸表达式自动 print
            code = _wrap_last_expression(code)

            exec(code, self._global_env, self._global_env)
            return output_buffer.getvalue(), True
        except Exception as e:
            return str(e), False
        finally:
            sys.stdout = original_stdout

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        Executes Python code with financial naming validation.

        Args:
            code: Python code to execute.
            timeout: Reserved for future use.

        Returns:
            Dict with 'observation' and 'success' status.
        """
        observation, success = self._run_code(code)
        return {"observation": observation, "success": success}
