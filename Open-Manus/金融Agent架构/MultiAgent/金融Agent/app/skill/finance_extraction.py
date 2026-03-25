"""
FinanceExtractionSkill - 全图 OCR 转写 + LLM 结构化提取为主，VLM 交叉校验与冲突裁决为辅。

模型分工（见 config.toml）：
  - 版面转写 OCR：``[ocr]`` 段，例如 ``deepseek-ai/DeepSeek-OCR``（仅把图转成文本/HTML 表）。
  - 从 OCR 转写中映射 variable=value：复用 ``[llm.vision]`` 同一套 LLM，通过纯文本 ``llm.ask()`` 调用（无图），
    与 VLM 看图提取共用 vision 配置，不是 OCR 模型本身。
  - VLM 整图候选 / 冲突裁决：同一 ``LLM(config_name="vision")`` 的多模态接口。

默认流程（use_region_guidance=False）：
  1. VLM 整图提取（候选值，带行列约束 prompt）
  2. 全图 OCR → 清洗为 HTML/文本 → **优先结构化本地解析**（HTML 表行列对齐）；仍缺则 **纯文本 LLM（ask，无图）** 从转写中补缺——与 VLM 不同，文本 LLM 只读 OCR 转写，适合宽表单元格定位，但不「看图」。
  3. 数值一致则采纳；若 VLM 与 OCR 冲突：**来自结构化表格解析的 OCR 优先**；若 OCR 来自纯文本 LLM 补缺，则 VLM 看图仲裁并校验行列语义一致。

可选 use_region_guidance=True：在具备 vision+OCR 时启用「区域裁剪 + 局部 OCR」作为补充路径。

无 vision 配置时退化为 OCR+LLM，保证兼容性。
"""
import contextvars
import io
import inspect
import base64
import html
import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import config
from app.logger import logger
from app.tool.base import BaseTool
from app.tool.ocr import OcrExtract, get_step_context, get_step_images_for_ocr

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from PIL import Image
except ImportError:
    Image = None
    logger.warning("[finance_extraction_skill] Pillow not installed, sniper crop OCR disabled.")

_shared_python_execute_ctx: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "shared_python_execute", default=None
)


def set_shared_python_execute(instance: Optional[Any]) -> None:
    """由 Flow 注入共享的 PythonExecute 实例。"""
    _shared_python_execute_ctx.set(instance)


def get_shared_python_execute() -> Optional[Any]:
    """获取共享的 PythonExecute 实例。"""
    return _shared_python_execute_ctx.get()


def _normalize_var_key(s: str) -> str:
    """规范化变量名用于匹配：小写、空格/下划线统一。"""
    s = (s or "").strip().lower()
    s = re.sub(r"[\s]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _apply_unit_conversion(val_str: str) -> Optional[float]:
    """
    解析数值并应用单位换算：万→×10000，亿→×1e8，%→×0.01。
    精准处理财务括号负数，如 $(4,614) 或 (4,614) 应解析为 -4614。
    返回 float 或 None（解析失败）。
    """
    val_str = (val_str or "").strip()
    if not val_str:
        return None

    # 1. 识别财务括号负数模式：$(4,614) 或 (4,614) 或 ($4,614)
    is_negative = False
    # 匹配括号包裹的数值（可选$符号和单位）
    if re.search(r'\(\s*\$?\s*[\d,.]+\s*[万亿%]?\s*\)', val_str):
        is_negative = True
    # 匹配原生负号（确保是数值开头的负号）
    elif re.match(r'^\s*-\s*[\d$]', val_str):
        is_negative = True

    # 2. 清洗特殊符号（保留数字、小数点）
    val_str_clean = val_str.replace(",", "").replace("(", "").replace(")", "").replace("$", "").replace("-", "")
    
    scale = 1.0
    m = re.search(r"([\d.]+)\s*([万亿%])", val_str_clean)
    if m:
        num_part = m.group(1)
        unit = m.group(2)
        if unit == "万":
            scale = 10000
        elif unit == "亿":
            scale = 1e8
        elif unit == "%":
            scale = 0.01
        try:
            num = float(num_part) * scale
            return -num if is_negative else num
        except ValueError:
            return None

    try:
        num = float(re.sub(r"[^\d.]", "", val_str_clean))
        return -num if is_negative else num
    except (ValueError, TypeError):
        return None


def _parse_numeric_from_sniper_ocr_text(text: str) -> Optional[float]:
    """
    Sniper 局部 OCR 可能返回一行 Markdown/废话，优先整段解析再回退到首个金额 token。
    """
    if not text or not str(text).strip():
        return None
    t = str(text).strip()
    n = _apply_unit_conversion(t)
    if n is not None:
        return n
    flat = re.sub(r"\s+", " ", t)
    for m in re.finditer(
        r"\(?\$?\s*[\d,]+\.?\d*\s*(?:[万亿%])?\s*\)?",
        flat,
    ):
        n = _apply_unit_conversion(m.group(0))
        if n is not None:
            return n
    return None


_INFER_EXCLUDE = frozenset({
    "result", "output", "ratio", "value", "total", "sum", "print", "abs",
    "from", "image", "extract", "and", "the", "step", "plan", "use", "get",
    "formula", "calculate", "apply", "context", "multimodal", "finance",
})


def _looks_like_var(name: str) -> bool:
    """变量名特征：含下划线/末尾为数字/长度≥4，且非排除词。"""
    n = _normalize_var_key(name)
    if not n or n in _INFER_EXCLUDE:
        return False
    return "_" in n or (n[-1].isdigit() if n else False) or len(n) >= 4


def _extract_from_markdown_table(text: str, var_name: str) -> Tuple[Optional[float], str]:
    """从Markdown/HTML表格中提取变量值。
    
    变量名格式: entity_variable_time (e.g. net_sales_2011)
    支持表格格式:
    | Variable | 2013 | 2012 | 2011 |
    | Net sales | $8,367 | $8,846 | $9,381 |
    """
    parts = var_name.lower().split('_')
    if len(parts) < 2:
        return None, ""
    
    # 提取entity和time
    time_part = parts[-1] if parts[-1].isdigit() else None
    entity_parts = parts[:-1] if time_part else parts
    
    lines = text.splitlines()
    headers = []
    in_table = False
    
    for i, line in enumerate(lines):
        # 检测表格头行
        if '|' in line and not line.strip().startswith('|---'):
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if len(cells) >= 2:
                # 检查是否包含时间列
                if time_part and any(time_part in h for h in cells):
                    headers = cells
                    in_table = True
                    continue
                # 没有时间部分，可能是行名行
                if not time_part and i > 0:
                    headers = cells
                    in_table = True
                    continue
        
        # 跳过分隔线
        if in_table and '---' in line:
            continue
        
        # 解析数据行
        if in_table and '|' in line and '---' not in line:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if len(cells) >= 2:
                row_name = cells[0].lower()
                # 匹配行名: 检查entity_parts是否都在row_name中
                row_name_normalized = row_name.replace(' ', '_')
                entity_key = '_'.join(entity_parts)
                
                # 匹配策略: 直接包含或模糊匹配
                match = entity_key in row_name_normalized
                if not match:
                    match = all(p in row_name for p in entity_parts)
                
                if match:
                    # 找到时间列
                    if time_part and len(headers) > 0:
                        for j, h in enumerate(headers):
                            if time_part in h:
                                # 修复: 确保列索引有效，处理可能的空单元格
                                if j < len(cells):
                                    val_str = cells[j]
                                    num = _apply_unit_conversion(val_str)
                                    if num is not None:
                                        return num, val_str
                                # 如果j超出范围，尝试使用j+1（兼容旧格式）
                                elif j + 1 < len(cells):
                                    val_str = cells[j + 1]
                                    num = _apply_unit_conversion(val_str)
                                    if num is not None:
                                        return num, val_str
                    else:
                        # 取第一个数值列
                        val_str = cells[1]
                        num = _apply_unit_conversion(val_str)
                        if num is not None:
                            return num, val_str
    
    return None, ""


def _is_valid_number(val: Any) -> bool:
    """检查值是否为有效的数字（不是None且不是nan）。"""
    if val is None:
        return False
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return False
    except:
        pass
    return True


def _parse_var_value_line(line: str, variables: List[str]) -> Optional[Tuple[str, float, str]]:
    """
    解析单行 var=value 格式。
    返回 (var_name, value, raw_str) 或 None。
    注意：如果值为NOT_FOUND或无法解析，返回None（而不是nan）。
    """
    line = line.strip()
    if not line or line.startswith('|') or line.startswith('-'):
        return None
    
    sep = "=" if "=" in line else (":" if ":" in line else None)
    if not sep:
        return None
    
    parts = line.split(sep, 1)
    if len(parts) != 2:
        return None
    
    key = parts[0].strip()
    val_str = parts[1].strip()
    
    # 检查是否是目标变量
    for var in variables:
        if _normalize_var_key(key) == _normalize_var_key(var):
            # 如果值为NOT_FOUND，返回None表示提取失败
            if val_str.upper() == "NOT_FOUND":
                return None
            num = _apply_unit_conversion(val_str)
            if num is not None:
                return (var, num, val_str)
            # 无法解析数值，返回None
            return None
    
    return None


def _parse_var_value_text(
    text: str, variables: List[str]
) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
    """解析 var=value 格式文本，支持Markdown/HTML表格，返回 (values, raw_values)。"""
    result: Dict[str, Optional[float]] = {}
    raw_values: Dict[str, str] = {}
    var_norm = {v: _normalize_var_key(v) for v in variables}

    for var_name in variables:
        result[var_name] = None
        raw_values[var_name] = ""
        target_key = var_norm[var_name]

        # 策略1: 解析 var=value 格式
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('|') or line.startswith('-'):
                continue
            sep = "=" if "=" in line else (":" if ":" in line else None)
            if sep:
                parts = line.split(sep, 1)
                if len(parts) == 2:
                    key = _normalize_var_key(parts[0].strip())
                    if key == target_key:
                        val_str = parts[1].strip()
                        raw_values[var_name] = val_str
                        if val_str.upper() != "NOT_FOUND":
                            num = _apply_unit_conversion(val_str)
                            if num is not None:
                                result[var_name] = num
                        break

        # 策略2: 从Markdown表格解析
        if result[var_name] is None:
            num, raw = _extract_from_markdown_table(text, var_name)
            if num is not None:
                result[var_name] = num
                raw_values[var_name] = raw

        # 策略3: 正则匹配
        if result[var_name] is None and target_key:
            pat = re.escape(target_key).replace("\\_", r"[\s_]*")
            m = re.search(rf"(?i){pat}\s*[=:]\s*([\d,.\-]+(?:\s*[万亿%])?)", text)
            if m:
                raw_values[var_name] = m.group(1)
                num = _apply_unit_conversion(m.group(1))
                if num is not None:
                    result[var_name] = num

    return result, raw_values


def _parse_lenient_extraction_text(text: str, variables: List[str]) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
    """
    宽容解析器：专门处理 8B 模型和 DeepSeek OCR 生成的 Markdown 表格或松散文本。
    确保提取到的值严格映射回 variables 列表，保证后续存储安全。
    """
    result: Dict[str, Optional[float]] = {v: None for v in variables}
    raw_values: Dict[str, str] = {v: "" for v in variables}

    var_norm_map = {v: _normalize_var_key(v) for v in variables}

    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith('---'):
            continue

        parts = []
        if '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
        elif '=' in line:
            parts = [p.strip() for p in line.split('=', 1)]
        elif ':' in line:
            parts = [p.strip() for p in line.split(':', 1)]

        if len(parts) >= 2:
            key_cand = _normalize_var_key(parts[0])
            val_str = parts[1]

            for orig_var, norm_var in var_norm_map.items():
                if norm_var in key_cand or key_cand in norm_var:
                    if result[orig_var] is None:
                        if "not_found" not in val_str.lower() and "none" not in val_str.lower():
                            num = _apply_unit_conversion(val_str)
                            if num is not None:
                                result[orig_var] = num
                                raw_values[orig_var] = val_str
                    break

    return result, raw_values


_FINANCIAL_SYNONYMS = {
    "ebit": ["operating income", "operating profit", "earnings before interest and taxes", "operating earnings", "营业利润", "息税前利润"],
    "net_sales": ["revenue", "sales", "total revenue", "net revenue", "销售收入", "营业收入", "销售额"],
    "interest_expense": ["interest", "interest cost", "interest paid", "利息费用", "利息支出"],
    "net_profit": ["net income", "profit", "earnings", "net earnings", "净利润", "净收益"],
    "basic_eps": ["earnings per share", "eps", "basic earnings per share", "每股收益", "基本每股收益"],
    "selling_expenses": ["selling costs", "sales expenses", "sg&a", "销售费用", "销售成本"],
    "fair_value": ["carrying value", "book value", "market value", "公允价值", "账面价值"],
    "unrecognized_compensation": ["unrecognized stock compensation", "unvested compensation", "未确认补偿", "未归属补偿"],
    "operating_expenses": ["operating cost", "opex", "运营费用", "经营费用"],
    "cost_of_goods_sold": ["cogs", "cost of sales", "销售成本", "主营业务成本"],
    "gross_profit": ["gross margin", "gross income", "毛利润", "毛利"],
    "total_assets": ["assets", "总资产"],
    "total_liabilities": ["liabilities", "总负债"],
    "shareholders_equity": ["equity", "stockholders equity", "股东权益", "所有者权益"],
    "cash_flow": ["cash flow from operations", "operating cash flow", "现金流", "经营现金流"],
    "depreciation": ["depreciation expense", "折旧", "折旧费用"],
    "amortization": ["amortization expense", "摊销", "摊销费用"],
    "research_development": ["r&d", "rd expense", "研发费用", "研究与开发"],
    "marketing_expenses": ["marketing cost", "advertising expense", "营销费用", "市场费用"],
    "admin_expenses": ["administrative expenses", "g&a", "管理费用", "行政费用"],
    "income_tax": ["tax expense", "income tax expense", "所得税", "所得税费用"],
    "dividend": ["dividends paid", "dividend payment", "股息", "分红"],
    "patent_expiry": ["patent expiration", "patent expiry year", "专利到期", "专利过期"],
    "quarter": ["q1", "q2", "q3", "q4", "first quarter", "second quarter", "third quarter", "fourth quarter", "一季度", "二季度", "三季度", "四季度"],
}


def _expand_var_synonyms(var_name: str) -> List[str]:
    """扩展变量名的同义词列表，用于更灵活的语义匹配。"""
    var_lower = _normalize_var_key(var_name)
    synonyms = [var_lower, var_name.lower()]
    
    for key, syn_list in _FINANCIAL_SYNONYMS.items():
        if key in var_lower:
            synonyms.extend(syn_list)
            break
        for syn in syn_list:
            if syn in var_lower:
                synonyms.extend([key] + syn_list)
                break
    
    return list(set(synonyms))


def _fuzzy_match_var_key(output_key: str, target_var: str) -> bool:
    """模糊匹配输出键与目标变量，支持同义词和中英文混合。"""
    output_norm = _normalize_var_key(output_key)
    target_norm = _normalize_var_key(target_var)
    
    if output_norm == target_norm:
        return True
    if output_norm in target_norm or target_norm in output_norm:
        return True
    
    target_synonyms = _expand_var_synonyms(target_var)
    for syn in target_synonyms:
        if syn in output_norm or output_norm in syn:
            return True
    
    return False


def _parse_natural_extraction_text(text: str, variables: List[str]) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
    """
    自然格式解析器：支持VLM自然对话式输出。
    
    支持格式：
    1. variable_name = value
    2. variable_name = NEEDS_CALCULATION
       → component_1 = value1
       → component_2 = value2
    3. variable_name: value
    """
    result: Dict[str, Optional[float]] = {v: None for v in variables}
    raw_values: Dict[str, str] = {v: "" for v in variables}
    extracted_components: Dict[str, Dict[str, float]] = {}
    
    lines = text.splitlines()
    current_calc_var = None
    current_components: Dict[str, float] = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('→') or line.startswith('->'):
            if current_calc_var and current_components:
                extracted_components[current_calc_var] = current_components.copy()
            continue
        
        parts = None
        if '=' in line:
            parts = [p.strip() for p in line.split('=', 1)]
        elif ':' in line:
            parts = [p.strip() for p in line.split(':', 1)]
        
        if not parts or len(parts) < 2:
            continue
        
        key_cand = parts[0].lstrip('→->').strip()
        val_str = parts[1].strip()
        
        for orig_var in variables:
            if result[orig_var] is not None:
                continue
            
            if _fuzzy_match_var_key(key_cand, orig_var):
                if "needs_calculation" in val_str.lower():
                    result[orig_var] = "NEEDS_CALCULATION"
                    raw_values[orig_var] = val_str
                    current_calc_var = orig_var
                    current_components = {}
                elif "not_found" not in val_str.lower() and "none" not in val_str.lower():
                    num = _apply_unit_conversion(val_str)
                    if num is not None:
                        result[orig_var] = num
                        raw_values[orig_var] = val_str
                break
    
    for var, components in extracted_components.items():
        if var in result and result[var] == "NEEDS_CALCULATION":
            for comp_name, comp_value in components.items():
                comp_var = f"{var}_{comp_name}"
                if comp_var not in result:
                    result[comp_var] = comp_value
                    raw_values[comp_var] = str(comp_value)
    
    return result, raw_values


def _row_matches_keywords(row_name: str, keywords: str) -> bool:
    """
    检查行名是否匹配关键词。增强对中文财报的兼容性。
    """
    row_lower = row_name.lower().replace(' ', '')
    kw_clean = keywords.lower().replace(' ', '')
    
    if not kw_clean:
        return False
        
    # 1. 纯中文或中英混合的高级包含匹配
    if re.search(r'[\u4e00-\u9fff]', kw_clean) or re.search(r'[\u4e00-\u9fff]', row_lower):
        if kw_clean in row_lower or row_lower in kw_clean:
            return True
        # 如果关键词里的某个长中文核心词出现在行名里
        for kw in keywords.split():
            if len(kw) >= 2 and re.search(r'[\u4e00-\u9fff]', kw) and kw in row_lower:
                return True

    # EPS / per-share 指标不能误落到 $M / 利润分子行。
    if (
        any(term in kw_clean for term in ("basiceps", "earningspershare", "pershare", "每股收益"))
        and any(marker in row_lower for marker in ("($m)", "(&m)", "$m", "profitfortheperiod", "usedinthecalculation", "attributabletoequityholders"))
    ):
        return False
                
    # 2. 英文的单词颗粒度匹配（长度≥4 的词；多词时至少命中 2 个，避免 "interest" 单命中股息行）
    keyword_list = [k.strip().lower() for k in keywords.split() if k.strip()]
    significant = list(dict.fromkeys([kw for kw in keyword_list if len(kw) >= 4]))
    if significant:
        for anchor in ("before", "after"):
            if anchor in keyword_list and anchor not in row_lower:
                return False
        hits = sum(1 for kw in significant if kw in row_lower)
        if len(significant) >= 4:
            return hits >= 3
        if len(significant) >= 2:
            return hits >= 2
        return hits >= 1
    for kw in keyword_list:
        if len(kw) >= 3 and kw in row_lower:
            return True

    return False


def _parse_markdown_table_for_value(text: str, row_keyword: str, col_keyword: str) -> Tuple[Optional[float], str]:
    lines = text.split('\n')
    headers = []
    col_index = -1
    
    col_keywords = [col_keyword.lower()] if col_keyword else []
    
    for line in lines:
        line = line.strip()
        if not line or '|' not in line:
            continue
            
        cells = [c.strip() for c in line.strip('|').split('|')]
        
        # 1. 找表头
        if not headers and any(c for c in cells if c):
            headers = cells
            if col_keywords:
                for kw in col_keywords:
                    for i, h in enumerate(headers):
                        h_lower = h.lower()
                        # 防贪婪陷阱：如果是年份，禁止匹配 "Change 2018 to 2019" 这种变动列
                        if re.match(r'^20\d{2}$', kw):
                            escaped_kw = re.escape(kw)
                            if re.search(rf'\b{escaped_kw}\b', h_lower) and not re.search(r'(change|growth|variance|%|变动|增减)', h_lower):
                                col_index = i
                                break
                        else:
                            if kw in h_lower or h_lower in kw:
                                col_index = i
                                break
                    if col_index != -1:
                        break
            continue
            
        if '---' in line:
            continue
            
        # 2. 找数据行
        if _row_matches_keywords(cells[0], row_keyword):
            # 情况A: 有明确年份且找到了列
            if col_index != -1 and col_index < len(cells):
                val_str = cells[col_index]
                num = _apply_unit_conversion(val_str)
                if num is not None:
                    return num, val_str
            # 情况B: 没有指定年份（如"市场份额"），直接抓取该行第一个有效数字
            elif not col_keywords:
                for cell in cells[1:]:
                    num = _apply_unit_conversion(cell)
                    if num is not None:
                        return num, cell
                    
    return None, ""


def _parse_ocr_layout_blocks(raw_text: str) -> List[Dict[str, Any]]:
    """
    解析 DeepSeek-OCR 风格的 `<|ref|>...<|det|>[[...]]` 块，保留类型、bbox 与原文。
    """
    if not raw_text:
        return []
    pattern = re.compile(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\[\[(.*?)\]\]<\|/det\|>\s*(.*?)(?=(?:<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>\[\[)|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    blocks: List[Dict[str, Any]] = []
    for match in pattern.finditer(raw_text):
        block_type = (match.group(1) or "").strip().lower()
        coord_tokens = [c.strip() for c in (match.group(2) or "").split(",")]
        if len(coord_tokens) != 4:
            continue
        try:
            bbox = [float(token) for token in coord_tokens]
        except ValueError:
            continue
        blocks.append(
            {
                "type": block_type,
                "bbox": bbox,
                "content": (match.group(3) or "").strip(),
            }
        )
    return blocks


def _html_tr_cells(row_html: str) -> List[str]:
    cells = re.findall(r"<t[dh].*?>(.*?)</t[dh]>", row_html, re.IGNORECASE | re.DOTALL)
    return [re.sub(r"<.*?>", "", c).strip() for c in cells]


def _split_cramped_financial_line(line: str) -> List[str]:
    """
    DeepSeek-OCR 常把多行行名挤成一行，用小写/逗号后接大写词边界切分短语（英文财报）。
    """
    line = re.sub(r"\s+", " ", (line or "").strip())
    if not line:
        return []
    parts = re.split(r"(?<=[a-z0-9,])(?:\s+)(?=[A-Z])", line)
    return [p.strip() for p in parts if p.strip()]


def _is_boilerplate_ocr_line(s: str) -> bool:
    sl = (s or "").lower()
    return "accompanying notes" in sl or "integral part" in sl


def _is_all_caps_section_header(s: str) -> bool:
    t = (s or "").strip()
    if len(t) < 6:
        return False
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 4:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio > 0.85


def _extract_ordered_labels_from_ocr_prefix(prefix: str) -> List[str]:
    """从首个 <table> 之前的正文按顺序抽出「行名」短语列表（与右侧数值表自上而下对齐）。"""
    labels: List[str] = []
    for raw in (prefix or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if _is_boilerplate_ocr_line(s):
            continue
        if _is_all_caps_section_header(s):
            continue
        labels.extend(_split_cramped_financial_line(s))
    return labels


def _group_table_rows_by_blank_lines(rows: List[List[str]]) -> List[List[List[str]]]:
    groups: List[List[List[str]]] = []
    current: List[List[str]] = []
    for cells in rows:
        non_empty = [cell for cell in cells if (cell or "").strip()]
        if not non_empty:
            if current:
                groups.append(current)
                current = []
            continue
        current.append(cells)
    if current:
        groups.append(current)
    return groups


def _extract_label_groups_from_ocr_blocks(blocks: List[Dict[str, Any]]) -> List[List[str]]:
    """
    从 OCR 文本块中提取按版面顺序排列的标签组。
    - 多标签块直接作为一组
    - 全大写单行块若后续紧跟多标签块，则视为 section header 丢弃
    - 其余单行块视为单标签组
    """
    text_blocks = [b for b in blocks if b.get("type") == "text"]
    groups: List[List[str]] = []
    for idx, block in enumerate(text_blocks):
        content = (block.get("content") or "").strip()
        if not content or _is_boilerplate_ocr_line(content):
            continue
        parts = _split_cramped_financial_line(content)
        if len(parts) > 1:
            groups.append(parts)
            continue
        single = parts[0] if parts else content
        next_parts: List[str] = []
        if idx + 1 < len(text_blocks):
            next_parts = _split_cramped_financial_line(
                (text_blocks[idx + 1].get("content") or "").strip()
            )
        if _is_all_caps_section_header(single) and len(next_parts) > 1:
            continue
        groups.append([single])
    return groups


def _row_is_section_subtotal(cells: List[str], primary_col: int, running_values: List[float]) -> bool:
    if primary_col >= len(cells):
        return False
    v_primary = _apply_unit_conversion(cells[primary_col])
    if v_primary is None or len(running_values) < 2:
        return False
    return abs(sum(running_values) - v_primary) < 1e-2


def _rebuild_table_with_side_labels(table_html: str, label_groups: List[List[str]]) -> str:
    """
    将“左侧标签 + 右侧空首列表”重建为带首列标签的 HTML 表。
    适用于财报中标签位于表外、数值位于表内的典型版式。
    """
    rows_html = re.findall(r"<tr.*?>(.*?)</tr>", table_html or "", re.IGNORECASE | re.DOTALL)
    if len(rows_html) < 2:
        return table_html

    parsed_rows = [_html_tr_cells(row) for row in rows_html]
    header = parsed_rows[0]
    data_groups = _group_table_rows_by_blank_lines(parsed_rows[1:])
    if not data_groups or not label_groups:
        return table_html

    primary_col = -1
    for idx, cell in enumerate(header):
        if re.search(r"\b20\d{2}\b", cell or ""):
            primary_col = idx
            break
    if primary_col == -1:
        for row in parsed_rows[1:]:
            for idx, cell in enumerate(row):
                if _apply_unit_conversion(cell) is not None:
                    primary_col = idx
                    break
            if primary_col != -1:
                break
    if primary_col == -1:
        return table_html

    rebuilt_groups: List[List[List[str]]] = []
    label_group_idx = 0
    for group in data_groups:
        current_labels = label_groups[label_group_idx] if label_group_idx < len(label_groups) else []
        label_group_idx += 1

        running_values: List[float] = []
        label_pos = 0
        rebuilt_rows: List[List[str]] = []
        for cells in group:
            row = list(cells)
            if primary_col < len(row):
                numeric_val = _apply_unit_conversion(row[primary_col])
            else:
                numeric_val = None
            if numeric_val is not None and _row_is_section_subtotal(row, primary_col, running_values):
                rebuilt_rows.append(row)
                running_values = []
                continue

            if label_pos < len(current_labels) and (not row or not (row[0] or "").strip() or row[0].strip() == "$"):
                if not row:
                    row = [""]
                row[0] = current_labels[label_pos]
                label_pos += 1
            rebuilt_rows.append(row)
            if numeric_val is not None:
                running_values.append(float(numeric_val))
        rebuilt_groups.append(rebuilt_rows)

    all_rows: List[List[str]] = [header]
    for group_idx, group in enumerate(rebuilt_groups):
        all_rows.extend(group)
        if group_idx != len(rebuilt_groups) - 1:
            all_rows.append(["" for _ in header])

    def _cells_to_html(cells: List[str]) -> str:
        return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"

    return "<table>" + "".join(_cells_to_html(cells) for cells in all_rows) + "</table>"


def _reconstruct_ocr_side_label_tables(raw_text: str) -> str:
    """
    利用 OCR 原始块结构，将“表外左标签 + 表内数值列”重建成首列带标签的 HTML 表。
    若无法可靠重建，则返回原文。
    """
    blocks = _parse_ocr_layout_blocks(raw_text)
    if not blocks:
        return raw_text
    label_groups = _extract_label_groups_from_ocr_blocks(blocks)
    if not label_groups:
        return raw_text

    rebuilt = raw_text
    for block in blocks:
        if block.get("type") != "table":
            continue
        content = block.get("content") or ""
        if "<table" not in content.lower():
            continue
        rebuilt_table = _rebuild_table_with_side_labels(content, label_groups)
        if rebuilt_table != content:
            rebuilt = rebuilt.replace(content, rebuilt_table, 1)
            break
    return rebuilt


def _row_has_numeric_in_columns(cells: List[str], col_indices: List[int]) -> bool:
    for i in col_indices:
        if i < len(cells) and _apply_unit_conversion(cells[i]) is not None:
            return True
    return False


def _parse_html_table_with_floating_labels(
    text: str, row_keyword: str, col_keyword: str
) -> Tuple[Optional[float], str]:
    """
    当 HTML 表首列无行名（空或仅 $），但表前正文按顺序给出了行名时，按行序对齐后再匹配 row_keyword。
    典型版式：左侧行名在表格外侧文本块，右侧为独立 HTML 表。
    """
    lower = (text or "").lower()
    idx = lower.find("<table")
    if idx == -1:
        return None, ""
    prefix = text[:idx]
    labels = _extract_ordered_labels_from_ocr_prefix(prefix)
    if not labels:
        return None, ""

    tables = re.findall(r"<table.*?>.*?</table>", text, re.IGNORECASE | re.DOTALL)
    if not tables:
        return None, ""

    col_keywords = [col_keyword.lower()] if col_keyword else []
    if not col_keywords:
        return None, ""

    table = tables[0]
    rows = re.findall(r"<tr.*?>(.*?)</tr>", table, re.IGNORECASE | re.DOTALL)
    if len(rows) < 2:
        return None, ""

    header_row_idx = -1
    col_indices: List[int] = []
    for ri, row in enumerate(rows):
        cells = _html_tr_cells(row)
        if not cells:
            continue
        local_cols: List[int] = []
        for kw in col_keywords:
            for i, h in enumerate(cells):
                h_lower = h.lower()
                if re.match(r"^20\d{2}$", kw):
                    escaped_kw = re.escape(kw)
                    if re.search(rf"\b{escaped_kw}\b", h_lower) and not re.search(
                        r"(change|growth|variance|%|变动|增减)", h_lower
                    ):
                        local_cols.append(i)
                else:
                    if kw in h_lower or h_lower in kw:
                        local_cols.append(i)
        if local_cols:
            header_row_idx = ri
            col_indices = local_cols
            break

    if header_row_idx == -1 or not col_indices:
        return None, ""

    label_pos = 0
    primary_col = col_indices[0]
    running_section_vals: List[float] = []

    for row in rows[header_row_idx + 1 :]:
        cells = _html_tr_cells(row)
        if not cells:
            continue
        if not _row_has_numeric_in_columns(cells, col_indices):
            continue
        if primary_col >= len(cells):
            continue
        v_primary = _apply_unit_conversion(cells[primary_col])
        if v_primary is None:
            continue

        # 无行名表常见：小计/合计行在正文中没有单独标签，不能消耗 label_pos，否则会整体错位。
        if (
            len(running_section_vals) >= 2
            and abs(sum(running_section_vals) - v_primary) < 1e-2
        ):
            running_section_vals = []
            continue

        if label_pos >= len(labels):
            break
        synthetic = labels[label_pos]
        label_pos += 1
        running_section_vals.append(float(v_primary))

        if _row_matches_keywords(synthetic, row_keyword):
            for col_idx in col_indices:
                if col_idx < len(cells):
                    val_str = cells[col_idx]
                    num = _apply_unit_conversion(val_str)
                    if num is not None:
                        return num, val_str
    return None, ""


def _parse_html_table_for_value(text: str, row_keyword: str, col_keyword: str) -> Tuple[Optional[float], str]:
    tables = re.findall(r'<table.*?>.*?</table>', text, re.IGNORECASE | re.DOTALL)
    if not tables:
        return None, ""

    col_keywords = [col_keyword.lower()] if col_keyword else []

    for table in tables:
        rows = re.findall(r'<tr.*?>(.*?)</tr>', table, re.IGNORECASE | re.DOTALL)
        if len(rows) < 2:
            continue

        col_indices = []
        first_row_cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', rows[0], re.IGNORECASE | re.DOTALL)
        first_row_cells = [re.sub(r'<.*?>', '', c).strip() for c in first_row_cells]

        if col_keywords:
            for kw in col_keywords:
                for i, h in enumerate(first_row_cells):
                    h_lower = h.lower()
                    if re.match(r'^20\d{2}$', kw):
                        escaped_kw = re.escape(kw)
                        if re.search(rf'\b{escaped_kw}\b', h_lower) and not re.search(r'(change|growth|variance|%|变动|增减)', h_lower):
                            col_indices.append(i)
                    else:
                        if kw in h_lower or h_lower in kw:
                            col_indices.append(i)

        for row_idx, row in enumerate(rows):
            cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', row, re.IGNORECASE | re.DOTALL)
            cells = [re.sub(r'<.*?>', '', c).strip() for c in cells]
            if not cells:
                continue

            if _row_matches_keywords(cells[0], row_keyword):
                # 如果有目标列
                if col_indices:
                    for col_idx in col_indices:
                        if col_idx < len(cells):
                            val_str = cells[col_idx]
                            num = _apply_unit_conversion(val_str)
                            if num is not None:
                                return num, val_str
                # 如果没有目标列，抓第一个数字
                elif not col_keywords:
                    for cell in cells[1:]:
                        num = _apply_unit_conversion(cell)
                        if num is not None:
                            return num, cell
    return _parse_html_table_with_floating_labels(text, row_keyword, col_keyword)


def _html_table_to_markdown(html_table: str) -> str:
    """将 OCR 返回的 HTML 表格转换为紧凑 Markdown，便于 LLM 阅读。"""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_table or "", re.IGNORECASE | re.DOTALL)
    if not rows:
        return html_table

    markdown_lines: List[str] = []
    for row_idx, row in enumerate(rows):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.IGNORECASE | re.DOTALL)
        cleaned_cells = [
            re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", cell or "")).strip() for cell in cells
        ]
        if not cleaned_cells:
            continue
        markdown_lines.append("| " + " | ".join(cleaned_cells) + " |")
        if row_idx == 0:
            markdown_lines.append("| " + " | ".join(["---"] * len(cleaned_cells)) + " |")

    return "\n".join(markdown_lines) if markdown_lines else html_table


def _replace_html_tables_with_markdown(text: str) -> str:
    """将文本中的 HTML 表格替换为 Markdown 表格，保留其它 OCR 文本不变。"""
    if not text or "<table" not in text.lower():
        return text
    return re.sub(
        r"<table.*?>.*?</table>",
        lambda m: "\n" + _html_table_to_markdown(m.group(0)) + "\n",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _parse_variable_semantics(var_name: str, semantic_query: str = "") -> Tuple[str, str]:
    text_to_parse = semantic_query if semantic_query else var_name
    text_lower = text_to_parse.lower()
    text_for_axis = text_lower.replace("_", " ").replace("-", " ")
    
    # 1. 提取年份 (如 2018, 2019)
    year_match = re.search(r'\b(20\d{2})\b', text_for_axis)
    col_keyword = year_match.group(1) if year_match else ""
    
    # 2. 提取行关键词
    row_part = text_lower
    if col_keyword:
        row_part = row_part.replace(col_keyword, '')
        
    # 移除无用的 Prompt 介词
    stop_words = {'extract', 'the', 'value', 'for', 'of', 'in', 'and', 'specifically', 'save_as', 'row:', 'column:'}
    words = []
    
    # 支持中英文混合的分词清洗
    for w in re.split(r'[\s_]+', row_part):
        w_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', w)  # 保留字母数字和中文
        if w_clean and w_clean not in stop_words:
            words.append(w_clean)
            
    row_keyword = ' '.join(words)
    return row_keyword.strip(), col_keyword.strip()


def _query_has_explicit_axis_constraint(text: str) -> bool:
    text = (text or "").lower().replace("_", " ").replace("-", " ")
    if not text:
        return False
    return bool(
        re.search(
            r"\b(19|20)\d{2}\b|\bq[1-4]\b|"
            r"\b(first|second|third|fourth)\s+quarter\b|"
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b|"
            r"\b(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b|"
            r"年|季度|季报|月|上半年|下半年|本期|上期|同期|期末|期初",
            text,
            re.IGNORECASE,
        )
    )


def _parse_lenient_extraction_text_v2(text: str, variables: List[str], semantic_queries: Dict[str, str] = None) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
    result: Dict[str, Optional[float]] = {v: None for v in variables}
    raw_values: Dict[str, str] = {v: "" for v in variables}
    semantic_queries = semantic_queries or {}

    for var in variables:
        query = semantic_queries.get(var, "")
        row_kw, col_kw = _parse_variable_semantics(var, query)
        
        if row_kw:
            if '<table' in text.lower():
                num, raw_str = _parse_html_table_for_value(text, row_kw, col_kw)
                if num is not None:
                    result[var], raw_values[var] = num, raw_str
                    continue
            if '|' in text:
                num, raw_str = _parse_markdown_table_for_value(text, row_kw, col_kw)
                if num is not None:
                    result[var], raw_values[var] = num, raw_str
                    continue

    # 单行键值对兜底
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith('---'):
            continue

        parts = []
        if '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
        elif '=' in line:
            parts = [p.strip() for p in line.split('=', 1)]
        elif ':' in line:
            parts = [p.strip() for p in line.split(':', 1)]

        if len(parts) >= 2:
            key_cand, val_str = parts[0], parts[1]
            if '_位置' in key_cand or '_location' in key_cand.lower():
                continue
            # 区域行（_bbox）与模糊键名可能匹配到目标变量，绝不能把坐标当数值写入
            if "_bbox" in key_cand.lower():
                continue
            vs = val_str.strip()
            if vs.startswith("[") and vs.endswith("]"):
                inner = vs[1:-1].replace(" ", "")
                if inner.count(",") == 3:
                    try:
                        [float(x) for x in inner.split(",")]
                        continue
                    except ValueError:
                        pass

            for orig_var in variables:
                if result[orig_var] is not None:
                    continue
                if _fuzzy_match_var_key(key_cand, orig_var):
                    if "not_found" not in val_str.lower() and "none" not in val_str.lower():
                        num = _apply_unit_conversion(val_str)
                        if num is not None:
                            result[orig_var], raw_values[orig_var] = num, val_str
                    break
    return result, raw_values


def _parse_decoupled_extraction(step_text: str) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    
    pattern = r'extract\s+[\'"]([^\'"]+)[\'"]\s+save_as\s+([a-z_][a-z0-9_]*)'
    
    for match in re.finditer(pattern, step_text, re.IGNORECASE):
        semantic_query = match.group(1).strip()
        python_variable = match.group(2).strip()
        if semantic_query and python_variable:
            result.append((semantic_query, python_variable))
    
    return result


def _infer_variables_from_plan(
    step_text: str, next_step_text: Optional[str] = None
) -> List[str]:
    """从 step 文本推断需提取的变量名。"""
    # 首先尝试解析解耦格式
    decoupled = _parse_decoupled_extraction(step_text)
    if decoupled:
        return [var for _, var in decoupled]
    
    # 回退到旧格式解析
    seen: set = set()
    result: List[str] = []

    def _add_var(v: str) -> None:
        vn = _normalize_var_key(v)
        if vn and vn not in seen and _looks_like_var(vn):
            seen.add(vn)
            result.append(vn)

    if next_step_text:
        for m in re.finditer(r"\b([a-z][a-z0-9_]*)\b", next_step_text, re.I):
            _add_var(m.group(1))
    if step_text:
        for m in re.finditer(r"\b([a-z][a-z0-9_]*)\b", step_text, re.I):
            _add_var(m.group(1))

    return result


def _get_semantic_queries(step_text: str) -> Dict[str, str]:
    """
    获取变量名到语义查询的映射。
    返回 {python_variable: semantic_query, ...}
    """
    decoupled = _parse_decoupled_extraction(step_text)
    return {var: query for query, var in decoupled}


def _normalize_vlm_bbox_to_fraction(
    raw: List[float], w: int, h: int
) -> Optional[Tuple[float, float, float, float]]:
    """
    将 VLM 输出的 4 个数转为归一化 [x1,y1,x2,y2]（左上-右下），兼容：
    - 归一化坐标（常见于 [0,1]）
    - 像素坐标（常见于 >1，易与归一化混淆导致裁剪错误）
    - 宽/高形式（当 x2<=x1 或 y2<=y1 时按宽、高解释，与 _crop_image 一致）
    """
    if len(raw) != 4 or w <= 0 or h <= 0:
        return None
    try:
        a, b, c, d = (float(x) for x in raw)
    except (TypeError, ValueError):
        return None

    pixel_mode = max(a, b, c, d) > 1.0
    if pixel_mode:
        if c > a and d > b:
            x1, y1, x2, y2 = a / w, b / h, c / w, d / h
        else:
            x1, y1 = a / w, b / h
            x2, y2 = (a + max(0.0, c)) / w, (b + max(0.0, d)) / h
    else:
        x1, y1, x2, y2 = a, b, c, d
        if x2 <= x1 or y2 <= y1:
            x2 = x1 + max(0.0, x2)
            y2 = y1 + max(0.0, y2)

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _expand_bbox_for_ocr_crop_context(
    bbox: Tuple[float, float, float, float],
    expand_context: bool = True,
) -> Tuple[float, float, float, float]:
    """
    在归一化 bbox 上保守扩展，便于 OCR 看到：
    - 向上：表头/年份列（常见于表格顶部）
    - 向左：宽表中数值列在右侧、行标签在左侧时的「行」上下文（与仅向上扩展配合）
    """
    if not expand_context:
        return bbox
    x1, y1, x2, y2 = bbox
    h_span = y2 - y1
    w_span = x2 - x1
    dy = min(0.12, max(0.03, 2.5 * max(h_span, 0.01)))
    y1_new = max(0.0, y1 - dy)
    x1_new = x1
    # 右侧窄框：数值在右、标签在左的宽表布局
    if x1 > 0.35 and w_span < 0.20 and w_span > 0:
        dx = min(0.48, max(0.05, 3.5 * w_span))
        x1_new = max(0.0, x1 - dx)
    return (x1_new, y1_new, x2, y2)


def _clean_ocr_text_for_llm_extraction(text: str) -> str:
    """
    剥离 DeepSeek-OCR 等输出的版式控制 token（如 <|ref|>、<|det|> 坐标块），
    再送入 LLM/解析器，避免与真实表格数字混排导致串行与误匹配。
    仅删除标记与压缩空白，不改动普通财务文本与数字字面量。
    """
    if not text:
        return ""
    t = html.unescape(text)
    t = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<\|det\|>.*?<\|/det\|>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<\|/ref\|>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<\|[^|]+\|>", "", t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _extract_vlm_location_hints(
    vlm_text: str, variables: List[str]
) -> Dict[str, str]:
    """从 VLM 原始输出中解析 `<var>_位置 = ...` 或 `<var>_location = ...` 提示。"""
    hints: Dict[str, str] = {}
    if not vlm_text:
        return hints
    for line in (vlm_text or "").splitlines():
        if "=" not in line:
            continue
        key_part, val_part = line.split("=", 1)
        key_norm = _normalize_var_key(key_part)
        if not key_norm.endswith("位置") and not key_norm.endswith("location"):
            continue
        key_base = re.sub(r"(_?位置|_?location)$", "", key_norm, flags=re.I)
        for var in variables:
            if _fuzzy_match_var_key(key_base, var):
                hints[var] = val_part.strip()
                break
    return hints


def _is_high_risk_location_hint(location_hint: str, semantic_query: str = "") -> bool:
    """
    某些位置提示会把 OCR-LLM 锚定到错误列，例如 principal / maturity amount / share count。
    这类提示只作为高风险信号，不直接传给 OCR 文本抽取提示。
    """
    loc = _normalize_var_key(location_hint)
    query = _normalize_var_key(semantic_query)
    if not loc:
        return False
    risky_loc_terms = {
        "maturity_amount",
        "principal",
        "principal_amount",
        "shares_outstanding",
        "share_count",
        "weighted_average_shares",
        "basic_shares",
        "diluted_shares",
        "unvested",
        "vested",
        "granted",
    }
    risky_query_terms = {
        "expense",
        "cost",
        "eps",
        "earnings_per_share",
        "per_share",
        "ratio",
        "market_share",
    }
    aligned_pairs = (
        ("maturity_amount", "maturity_amount"),
        ("principal_amount", "principal_amount"),
        ("principal", "principal"),
        ("shares_outstanding", "shares_outstanding"),
        ("weighted_average_shares", "weighted_average_shares"),
        ("basic_shares", "basic_shares"),
        ("diluted_shares", "diluted_shares"),
    )
    for loc_term, query_term in aligned_pairs:
        if loc_term in loc and query_term in query:
            return False
    if any(term in loc for term in risky_loc_terms):
        if not query:
            return True
        if any(term in query for term in risky_query_terms):
            return True
    return False


def _extract_direct_value_hint_from_text(
    text: str, semantic_query: str = ""
) -> Optional[Tuple[float, str]]:
    """
    对 OCR 转写做少量高精度启发式补充：
    - 债券/票据 annual interest expense：若文本显式给出 “$X million per year”
      或可由 principal × coupon rate 直接推出，则优先返回该值。
    仅在命中特征非常明确时返回，避免污染其它任务。
    """
    q = (semantic_query or "").lower()
    if not text or not q:
        return None

    flat = re.sub(r"\s+", " ", html.unescape(text)).strip()
    flat_lower = flat.lower()

    is_basic_eps_query = any(term in q for term in ("basic earnings per share", "basic_eps", "eps", "per share"))
    if is_basic_eps_query:
        target_year_match = re.search(r"\b(20\d{2})\b", q)
        target_year = target_year_match.group(1) if target_year_match else ""
        first_2019 = flat_lower.find("2019")
        first_2018 = flat_lower.find("2018")
        if target_year and first_2019 != -1 and first_2018 != -1:
            first_year, second_year = ("2019", "2018") if first_2019 < first_2018 else ("2018", "2019")
            eps_patterns = [
                re.compile(
                    r"basic earnings per share(?!\s*\(\$m\)|\s*\(&m\)).{0,80}?\b([\d,.]+)\b.{0,20}?\b([\d,.]+)\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"basic eps(?!\s*\(\$m\)|\s*\(&m\)).{0,80}?\b([\d,.]+)\b.{0,20}?\b([\d,.]+)\b",
                    re.IGNORECASE,
                ),
            ]
            for pat in eps_patterns:
                match = pat.search(flat_lower)
                if not match:
                    continue
                first_val, second_val = match.group(1), match.group(2)
                target_val = first_val if target_year == first_year else second_val
                num = _apply_unit_conversion(target_val)
                if num is not None:
                    return num, target_val

    is_note_interest_query = (
        ("interest" in q or "coupon" in q)
        and ("note" in q or "bond" in q)
        and ("expense" in q or "annual" in q or "per year" in q)
    )
    if not is_note_interest_query:
        return None

    target_year_match = re.search(r"\b(20\d{2})\b", q)
    target_year = target_year_match.group(1) if target_year_match else ""

    # 1) 先尝试直接读取 prose 中 “... $10 million and $25 million per year, respectively”
    if target_year:
        respectively_patterns = [
            re.compile(
                r"interest on the .*?(\d{4}) notes\s+and\s+(?:the\s+)?(\d{4}) notes\s+of\s+approximately\s+"
                r"\$?\s*([\d,.]+)\s*(million|billion|thousand)?\s*and\s*"
                r"\$?\s*([\d,.]+)\s*(million|billion|thousand)?\s*(?:per year|annually|a year)"
                r".{0,40}?respectively",
                re.IGNORECASE,
            ),
            re.compile(
                r"(\d{4}) notes.*?(\d{4}) notes.*?"
                r"\$?\s*([\d,.]+)\s*(million|billion|thousand)?\s*and\s*"
                r"\$?\s*([\d,.]+)\s*(million|billion|thousand)?\s*(?:per year|annually|a year)"
                r".{0,40}?respectively",
                re.IGNORECASE,
            ),
        ]
        for respectively_pat in respectively_patterns:
            for m in respectively_pat.finditer(flat_lower):
                y1, y2 = m.group(1), m.group(2)
                v1, u1 = m.group(3), (m.group(4) or "million").lower()
                v2, u2 = m.group(5), (m.group(6) or "million").lower()
                target_val, target_unit = (v1, u1) if target_year == y1 else ((v2, u2) if target_year == y2 else (None, None))
                if target_val is not None:
                    num = float(target_val.replace(",", ""))
                    scale = 1000.0 if target_unit == "billion" else (0.001 if target_unit == "thousand" else 1.0)
                    return num * scale, f"{target_val} {target_unit} per year"

    # 2) 再尝试 principal × coupon rate（例如 750 million of 3.375% notes due 2022）
    principal_rate_pat = re.compile(
        r"\$?\s*([\d,.]+)\s*(billion|million|thousand)?\s+of\s+([\d.]+)\s*%"
        r".{0,80}?(?:notes?|bonds?).{0,80}?\b(20\d{2})\b",
        re.IGNORECASE,
    )
    for m in principal_rate_pat.finditer(flat_lower):
        principal_str, unit, rate_str, year = m.groups()
        if target_year and year != target_year:
            continue
        principal = float(principal_str.replace(",", ""))
        unit = (unit or "million").lower()
        if unit == "billion":
            principal *= 1000.0
        elif unit == "thousand":
            principal *= 0.001
        rate = float(rate_str)
        return principal * rate / 100.0, f"{principal_str} {unit} * {rate_str}%"

    # 3) HTML/Markdown 表格行：例如 “3.375% Notes due 2022 | 750 | ...”
    row_coupon_pat = re.compile(
        r"([\d.]+)\s*%\s*(?:notes?|bonds?)\s+due\s+(20\d{2}).{0,40}?"
        r"(?:\(?\$?\s*([\d,]+(?:\.\d+)?)\)?)(?!\s*%)",
        re.IGNORECASE,
    )
    for m in row_coupon_pat.finditer(flat_lower):
        rate_str, year, principal_str = m.groups()
        if target_year and year != target_year:
            continue
        principal = float(principal_str.replace(",", ""))
        rate = float(rate_str)
        # 表格语境里该列通常已是“in millions”，默认按 million 计
        return principal * rate / 100.0, f"{principal_str} million * {rate_str}%"

    return None


_SKILL_DESCRIPTION = """Finance extraction skill: full-page OCR + LLM extraction, VLM cross-check.

Use when: extracting numeric variables from financial tables/charts in images.
Workflow: (1) VLM extract candidates; (2) full-page OCR transcript, cleaned, then LLM maps variables;
          (3) agree → use; conflict → VLM judges on image. Optional region mode via use_region_guidance.

Parameters:
- variables: list of variable names to extract (e.g. ["net_sales_2019", "interest_expense"])
- base64_image: optional single image; if omitted and use_context_image=true, uses step image
- use_context_image: when true (default), use current step's image from Flow context
- use_region_guidance: when false (default), full-page OCR + LLM; when true, also run region-based OCR crop path
"""


class FinanceExtractionSkill(BaseTool):
    """全图 OCR + LLM 提取为主，VLM 校验与可选区域 OCR 为辅。"""

    name: str = "finance_extraction_skill"
    description: str = _SKILL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Variable names to extract (e.g. ['ebit_2018', 'interest_expense_2018'])",
            },
            "base64_image": {
                "type": "string",
                "description": "Optional single base64-encoded image. If omitted and use_context_image=true, uses step image.",
            },
            "use_context_image": {
                "type": "boolean",
                "description": "When true (default), use current step's image from Flow context.",
                "default": True,
            },
            "use_region_guidance": {
                "type": "boolean",
                "description": "When false (default): full-page OCR + LLM extraction. When true: additionally run VLM bbox + cropped OCR (Sniper) for missing cells.",
                "default": False,
            },
        },
        "required": ["variables"],
    }

    def __init__(self, **data):
        super().__init__(**data)
        self._client = None

    def _get_ocr_client(self):
        """获取 OCR 客户端。"""
        if self._client is not None:
            return self._client
        ocr_cfg = getattr(config, "ocr_config", None)
        if not ocr_cfg:
            return None
        if AsyncOpenAI is None:
            return None
        self._client = AsyncOpenAI(
            api_key=ocr_cfg.api_key,
            base_url=ocr_cfg.base_url,
        )
        return self._client

    def _get_vision_llm(self):
        """获取 ``[llm.vision]`` 配置的 LLM：既用于 VLM 看图，也用于 OCR 转写后的纯文本 ``ask()`` 抽数。"""
        try:
            llm_cfg = getattr(config, "llm", None)
            if llm_cfg and "vision" in llm_cfg:
                from app.llm import LLM
                return LLM(config_name="vision")
        except Exception as e:
            logger.debug(f"[finance_extraction_skill] No vision LLM: {e}")
        return None

    def _decode_base64_image(self, base64_str: str):
        """解码为 PIL Image；失败返回 None。"""
        if Image is None:
            return None
        try:
            raw = base64_str.split(",", 1)[1] if "," in base64_str else base64_str
            img_data = base64.b64decode(raw)
            return Image.open(io.BytesIO(img_data))
        except Exception as e:
            logger.warning(f"[finance_extraction_skill] image decode failed: {e}")
            return None

    def _crop_pil_to_png_base64(
        self,
        img: Any,
        bbox: List[float],
        padding: float = 0.08,
    ) -> Optional[str]:
        """对已打开的 PIL 图按归一化 bbox 裁剪，返回 PNG base64（无 data: 前缀）。"""
        if Image is None or img is None:
            return None
        if not bbox or len(bbox) != 4:
            return None
        try:
            w, h = img.size
            x1, y1, x2, y2 = bbox
            # 同时兼容 [x,y,w,h] 和 [x1,y1,x2,y2]
            if x2 <= x1 or y2 <= y1:
                x2 = x1 + max(0.0, x2)
                y2 = y1 + max(0.0, y2)
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            if x2 <= x1 or y2 <= y1:
                return None

            px1 = max(0, int((x1 - padding) * w))
            py1 = max(0, int((y1 - padding) * h))
            px2 = min(w, int((x2 + padding) * w))
            py2 = min(h, int((y2 + padding) * h))
            if px2 <= px1 or py2 <= py1:
                return None

            cropped = img.crop((px1, py1, px2, py2))
            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning(f"[finance_extraction_skill] crop failed: {e}")
            return None

    def _crop_image(
        self, base64_str: str, bbox: List[float], padding: float = 0.08
    ) -> Optional[str]:
        """根据归一化坐标裁剪 Base64 图片，返回 PNG 的 base64（无 data: 前缀）。"""
        img = self._decode_base64_image(base64_str)
        return self._crop_pil_to_png_base64(img, bbox, padding=padding)

    async def _vlm_extract(
        self, base64_image: str, variables: List[str], semantic_queries: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Phase 1: VLM 极简提取 - 一句话prompt，释放模型原生能力
        
        Args:
            variables: 变量名列表
            semantic_queries: 可选，变量名到语义查询的映射
        """
        llm = self._get_vision_llm()
        if llm is None:
            return {"values": {}, "raw_values": {}, "vlm_text": "", "error": "No vision LLM"}

        # 修复：强制VLM输出变量名=数值格式，但保持简洁
        if semantic_queries:
            items = [semantic_queries.get(var, var) for var in variables]
        else:
            items = variables
        
        items_text = "\n".join([f"- {item}" for item in items])
        
        prompt = f"""<EXTRACTION_TASK>
从财务表格图片中提取以下数据：
{items_text}
</EXTRACTION_TASK>

<ATTENTION_CHECK>
You MUST explicitly double-check the COLUMN HEADER (Year/Date) and ROW HEADER before extracting.
If the column says 2012 and the query asks for 2011, DO NOT extract it.
Look at the intersection of the correct ROW and COLUMN.
</ATTENTION_CHECK>

<OUTPUT_FORMAT>
For each item, output TWO lines:
1. 变量名_位置 = 行名, 列名
2. 变量名 = 数值

Example:
interest_expense_2018_位置 = Interest Expense, 2018
interest_expense_2018 = 97.0
net_sales_2011_位置 = Net Sales, 2011
net_sales_2011 = 9381
</OUTPUT_FORMAT>"""

        try:
            url = base64_image if base64_image.startswith("data:") else f"data:image/png;base64,{base64_image}"
            vlm_text = await llm.ask_with_images(
                messages=[{"role": "user", "content": prompt}],
                images=[url],
                stream=False,
            )
            vlm_text = (vlm_text or "").strip()
            logger.info(f"[finance_extraction_skill] VLM response: {vlm_text[:500]}...")

            # 修复：使用更强大的解析器，传入semantic_queries实现解耦
            values, raw_values = _parse_lenient_extraction_text_v2(vlm_text, variables, semantic_queries)
            location_hints = _extract_vlm_location_hints(vlm_text, variables)
            
            for var in variables:
                if values.get(var) is None:
                    logger.warning(f"[finance_extraction_skill] VLM未能提取变量 {var}")

            return {
                "values": values,
                "raw_values": raw_values,
                "location_hints": location_hints,
                "vlm_text": vlm_text,
                "variables": variables,
            }
        except Exception as e:
            logger.warning(f"[finance_extraction_skill] VLM extract failed: {e}")
            return {"values": {}, "raw_values": {}, "vlm_text": "", "error": str(e)}

    async def _vlm_extract_with_regions(
        self, base64_image: str, variables: List[str]
    ) -> Dict[str, Any]:
        """Phase 1: VLM 提取数值 + 数值区域 bbox（供 Sniper OCR 裁剪）。"""
        llm = self._get_vision_llm()
        if llm is None:
            return {"values": {}, "raw_values": {}, "regions": {}, "vlm_text": "", "error": "No vision LLM"}

        var_desc_text = "\n".join([f"- {v}" for v in variables])
        prompt = f"""Extract requested values from this financial image.

Variables:
{var_desc_text}

Output STRICTLY with two lines per variable:
<var>_bbox = [x_min, y_min, x_max, y_max]
<var> = <value or NOT_FOUND>

Rules:
- _bbox is ONE axis-aligned RECTANGLE (four numbers), not a single point: [left, top, right, bottom].
- Prefer normalized coordinates in [0,1] relative to full image width/height. If easier, you may use pixel coordinates; the pipeline will normalize them.
- The rectangle should tightly cover the target NUMBER/cell; downstream OCR may expand upward slightly to include headers—do not draw huge boxes.
- Keep variable names EXACTLY as given.
"""
        try:
            url = base64_image if base64_image.startswith("data:") else f"data:image/png;base64,{base64_image}"
            vlm_text = await llm.ask_with_images(
                messages=[{"role": "user", "content": prompt}],
                images=[url],
                stream=False,
            )
            vlm_text = (vlm_text or "").strip()
            logger.info(f"[finance_extraction_skill] VLM region response: {vlm_text[:260]}...")

            values, raw_values = _parse_lenient_extraction_text_v2(vlm_text, variables)
            location_hints = _extract_vlm_location_hints(vlm_text, variables)
            regions: Dict[str, List[float]] = {}
            for line in vlm_text.splitlines():
                if "_bbox" not in line.lower() or "=" not in line:
                    continue
                key_part, val_part = line.split("=", 1)
                key_cand = re.sub(r"_bbox", "", key_part, flags=re.I).strip()
                m = re.search(r"\[([^\]]+)\]", val_part)
                if not m:
                    continue
                nums = [x.strip() for x in m.group(1).split(",") if x.strip()]
                if len(nums) != 4:
                    continue
                try:
                    coords = [float(x) for x in nums]
                except Exception:
                    continue
                for var in variables:
                    if _fuzzy_match_var_key(key_cand, var):
                        regions[var] = coords
                        break

            return {
                "values": values,
                "raw_values": raw_values,
                "location_hints": location_hints,
                "regions": regions,
                "vlm_text": vlm_text,
                "variables": variables,
            }
        except Exception as e:
            logger.warning(f"[finance_extraction_skill] VLM region extract failed: {e}")
            return {"values": {}, "raw_values": {}, "regions": {}, "vlm_text": "", "error": str(e)}

    async def _ocr_region_extract(
        self,
        base64_image: str,
        variables: List[str],
        vlm_regions: Dict[str, List[float]],
        region_expand_header: bool = True,
    ) -> Dict[str, Any]:
        """Phase 2 Sniper: 按 VLM bbox 裁剪后送 OCR 精读数值。"""
        client = self._get_ocr_client()
        ocr_cfg = getattr(config, "ocr_config", None)
        if not client or not ocr_cfg:
            return {"values": {}, "raw_values": {}, "ocr_text": "", "error": "OCR client not available"}
        if Image is None:
            return {"values": {}, "raw_values": {}, "ocr_text": "", "error": "Pillow not available"}

        img = self._decode_base64_image(base64_image)
        if img is None:
            return {"values": {}, "raw_values": {}, "ocr_text": "", "error": "Image decode failed"}

        model = ocr_cfg.model
        values: Dict[str, Optional[float]] = {v: None for v in variables}
        raw_values: Dict[str, str] = {v: "" for v in variables}
        snippets: List[str] = []
        iw, ih = img.size
        # 与 DeepSeek-OCR 官方 README 对齐；勿用长英文指令，易触发非转写输出。
        sniper_prompt = "<|grounding|>OCR this image."
        for var in variables:
            raw_bbox = vlm_regions.get(var)
            if not raw_bbox:
                continue
            norm = _normalize_vlm_bbox_to_fraction(raw_bbox, iw, ih)
            if not norm:
                logger.debug(
                    f"[finance_extraction_skill] skip invalid bbox for {var}: {raw_bbox!r}"
                )
                continue
            x1, y1, x2, y2 = _expand_bbox_for_ocr_crop_context(
                norm, expand_context=region_expand_header
            )
            crop_b64 = self._crop_pil_to_png_base64(img, bbox=[x1, y1, x2, y2])
            if not crop_b64:
                continue
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": sniper_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_b64}"}},
                            ],
                        }
                    ],
                    max_tokens=512,
                    temperature=0,
                )
                text = (resp.choices[0].message.content or "").strip()
                snippets.append(f"{var}={text}")
                num = _parse_numeric_from_sniper_ocr_text(text)
                if num is not None:
                    values[var] = num
                    raw_values[var] = text
                    logger.info(f"[finance_extraction_skill] Sniper OCR hit: {var}={text} -> {num}")
            except Exception as e:
                logger.warning(f"[finance_extraction_skill] Sniper OCR failed ({var}): {e}")

        return {
            "values": values,
            "raw_values": raw_values,
            "ocr_text": "\n".join(snippets),
            "variables": variables,
        }

    async def _vlm_verify(
        self,
        base64_image: str,
        var_name: str,
        vlm_val: float,
        ocr_val: float,
        semantic_query: Optional[str] = None,
    ) -> float:
        """Phase 3: VLM 裁决冲突 - 保守策略，避免引入新错误"""
        llm = self._get_vision_llm()
        if llm is None:
            return vlm_val

        try:
            url = base64_image if base64_image.startswith("data:") else f"data:image/png;base64,{base64_image}"

            query_hint = semantic_query if semantic_query else var_name

            # 保守策略：让VLM选择A或B，不强制要求找第三个值；强调行列与指标名一致（避免 swap gain vs cash flow hedge 等混行）
            prompt = f"""Which value is correct for: {query_hint}

Candidate A (VLM whole-image): {vlm_val}
Candidate B (from OCR transcript / table parse): {ocr_val}

Look at the image. Before choosing:
1) Identify the table ROW whose label matches the requested metric (e.g. cash flow hedges vs gain on swaps vs net income effect are different rows).
2) Identify the COLUMN for the requested date/year/period if the query mentions one.
3) The correct number must sit at that row–column intersection. Reject a candidate that clearly comes from another row/column even if the number looks plausible.

Reply EXACTLY in this format:
Choice: <A or B>
Reason: <brief explanation naming row/column alignment>"""

            resp = await llm.ask_with_images(
                messages=[{"role": "user", "content": prompt}],
                images=[url],
                stream=False,
            )
            resp_text = (resp or "").strip()
            logger.info(f"[finance_extraction_skill] VLM Verify:\n{resp_text}")

            # 解析选择
            choice_match = re.search(r'Choice:\s*([AB])', resp_text, re.IGNORECASE)
            if choice_match:
                choice = choice_match.group(1).upper()
                if choice == 'A':
                    logger.info(f"[finance_extraction_skill] VLM Verify chose A: {vlm_val}")
                    return vlm_val
                elif choice == 'B':
                    logger.info(f"[finance_extraction_skill] VLM Verify chose B: {ocr_val}")
                    return ocr_val

            # 兜底：返回VLM值
            logger.warning(f"[finance_extraction_skill] VLM Verify no clear choice, fallback to VLM: {vlm_val}")
            return vlm_val

        except Exception as e:
            logger.warning(f"[finance_extraction_skill] VLM verify failed: {e}")
            return vlm_val

    async def _ocr_semantic_extract(
        self,
        base64_image: str,
        variables: List[str],
        semantic_queries: Optional[Dict[str, str]] = None,
        location_hints: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """全图 OCR 转写 + 清洗 + LLM 仅从文本映射变量（与图像解耦，避免版式 token 污染）。"""
        client = self._get_ocr_client()
        if not client:
            return {"values": {}, "raw_values": {}, "ocr_text": "", "error": "OCR client not available"}

        # 1. 优先复用 OcrExtract 的清洗与质量评估链路，避免脏输出直通语义提取。
        ocr_text = ""
        ocr_error = ""
        try:
            ocr_tool = OcrExtract()
            tool_result = await ocr_tool.execute(base64_image=base64_image)
            if tool_result.get("success"):
                ocr_text = (
                    (tool_result.get("markdown") or tool_result.get("text") or "").strip()
                )
            else:
                ocr_error = str(tool_result.get("error", ""))
        except Exception as e:
            ocr_error = str(e)

        # 2. 兜底：若工具通道失败，仍保留直连 OCR API 的退路，确保兼容性。
        if not ocr_text:
            ocr_prompt = OcrExtract._OCR_PROMPT
            try:
                url = (
                    base64_image
                    if base64_image.startswith("data:")
                    else f"data:image/png;base64,{base64_image}"
                )
                ocr_cfg = getattr(config, "ocr_config", None)
                model = ocr_cfg.model if ocr_cfg else "deepseek-ai/DeepSeek-OCR"
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": ocr_prompt},
                                {"type": "image_url", "image_url": {"url": url}},
                            ],
                        }
                    ],
                    max_tokens=2000,
                    temperature=0,
                )
                ocr_text = (response.choices[0].message.content or "").strip()
            except Exception as e:
                logger.warning(
                    f"[finance_extraction_skill] OCR transcription failed (tool_error={ocr_error}): {e}"
                )
                return {"values": {}, "raw_values": {}, "ocr_text": "", "error": str(e)}

        reconstructed_ocr_text = _reconstruct_ocr_side_label_tables(ocr_text)
        if reconstructed_ocr_text != ocr_text:
            logger.info(
                "[finance_extraction_skill] Reconstructed side-label OCR table into first-column-labeled table before cleaning."
            )

        ocr_for_llm = _clean_ocr_text_for_llm_extraction(reconstructed_ocr_text)
        if not (ocr_for_llm or "").strip():
            ocr_for_llm = (reconstructed_ocr_text or ocr_text or "").strip()

        logger.info(
            f"[finance_extraction_skill] OCR text (cleaned for LLM, preview): {ocr_for_llm[:300]}..."
        )

        # 2. 先做结构化本地解析：HTML/Markdown 表格或规范键值对可直接确定时，不交给 LLM 覆盖。
        result, raw_values = _parse_lenient_extraction_text_v2(
            ocr_for_llm, variables, semantic_queries
        )
        provenance: Dict[str, str] = {}
        for var in variables:
            if result.get(var) is not None:
                provenance[var] = "structure"
        missing_vars = [var for var in variables if result.get(var) is None]
        if not missing_vars:
            logger.info(
                "[finance_extraction_skill] OCR structural parsing resolved all requested variables without LLM fallback."
            )
        else:
            logger.info(
                f"[finance_extraction_skill] OCR structural parsing resolved {len(variables) - len(missing_vars)}/{len(variables)} variables; missing={missing_vars}"
            )

        # 3. 对结构化解析仍缺失的变量，再使用 LLM 仅做补缺，避免覆盖已确定值。
        llm = self._get_vision_llm()
        llm_ready_text = _replace_html_tables_with_markdown(ocr_for_llm)

        if llm and missing_vars and llm_ready_text.strip():
            if semantic_queries:
                query_descriptions = [
                    f'• {semantic_queries.get(var, var)} (save as: {var})'
                    for var in missing_vars
                ]
                var_list = "\n".join(query_descriptions)
                logger.info(
                    f"[finance_extraction_skill] LLM processing OCR text only for missing variables: {missing_vars}"
                )
            else:
                var_list = "\n".join([f"• {var}" for var in missing_vars])

            safe_hint_lines: List[str] = []
            for var in missing_vars:
                hint = (location_hints or {}).get(var, "").strip()
                query = semantic_queries.get(var, var) if semantic_queries else var
                if hint and not _is_high_risk_location_hint(hint, query):
                    safe_hint_lines.append(f"- {var}: {hint}")
            hint_block = ""
            if safe_hint_lines:
                hint_block = (
                    "\nPossible visual anchor hints from the image model "
                    "(use only if they semantically match the request and the OCR text):\n"
                    + "\n".join(safe_hint_lines)
                    + "\n"
                )

            extract_prompt = f"""The following is plain text transcribed from a financial image (tables and labels). No image is attached; use only this text.

<document>
{llm_ready_text}
</document>

Extract one numeric value per item below. Match each request to the correct row and column (e.g. year or period) using row labels and column headers in the text.
{var_list}
{hint_block}

Rules:
- Use semantic matching for row/column names (including synonyms and another language if present in the text).
- Take the number from the cell at the intersection of the correct row and column; do not use numbers from other metrics (e.g. net income vs income before tax) unless the request clearly refers to that metric.
- If the requested metric is stated directly in narrative text, footnotes, or prose, prefer that direct statement over a nearby table row with a related but different meaning.
- Do not substitute nearby supporting numbers such as principal amount, maturity amount, coupon rate, share count, row total, or net income effect for the requested metric unless the request explicitly asks for them.
- Preserve the displayed unit for per-share / EPS / ratio style metrics; do not rescale them unless the document explicitly states a different unit.
- Output exactly one line per variable: variable_name = value
- If no matching number exists in the document, output: variable_name = NOT_FOUND
- No explanations, no markdown fences, no extra lines.
"""
            try:
                # 纯文本阅读理解：必须用 ask()，勿用 ask_with_images(images=[])。
                # 后者仍走多模态消息组装路径，部分网关会对「无图」报 ValidationError。
                resp = await llm.ask(
                    messages=[{"role": "user", "content": extract_prompt}],
                    stream=False,
                )
                llm_text = (resp or "").strip()
                logger.info(f"[finance_extraction_skill] LLM extraction from OCR text: {llm_text[:300]}...")

                llm_values, llm_raw_values = _parse_lenient_extraction_text_v2(
                    llm_text,
                    missing_vars,
                    {var: semantic_queries.get(var, var) for var in missing_vars}
                    if semantic_queries
                    else None,
                )
                for var in missing_vars:
                    if llm_values.get(var) is None:
                        continue
                    result[var] = llm_values[var]
                    raw_values[var] = llm_raw_values.get(var, "")
                    provenance[var] = "llm_text"
            except Exception as e:
                logger.warning(f"[finance_extraction_skill] LLM text extraction failed: {e}")

        # 4. 对少数高精度可推断的 narrative / footnote 场景做启发式补充（仅补缺，不覆盖已有结果）
        for var in variables:
            if result.get(var) is not None:
                continue
            query = semantic_queries.get(var, var) if semantic_queries else var
            hinted = _extract_direct_value_hint_from_text(ocr_for_llm, query)
            if hinted is not None:
                num, raw = hinted
                result[var] = num
                raw_values[var] = raw
                provenance[var] = "heuristic"
                logger.info(
                    f"[finance_extraction_skill] Heuristic text hint hit: {var} = {raw} -> {num}"
                )

        for var in variables:
            if result.get(var) is None:
                logger.warning(f"[finance_extraction_skill] OCR/LLM未能提取变量 {var}")

        return {
            "values": result,
            "raw_values": raw_values,
            "ocr_text": ocr_for_llm,
            "variables": variables,
            "provenance": provenance,
        }

    async def execute(
        self,
        variables: Optional[List[str]] = None,
        base64_image: Optional[str] = None,
        use_context_image: bool = True,
        use_region_guidance: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        默认：全图 OCR（清洗后）+ LLM 抽数，VLM 整图候选与冲突裁决；可选区域 Sniper。

        use_region_guidance=False（默认）：
          1. VLM 整图提取（带行列约束）
          2. 全图 OCR → 清洗版式 token → LLM 仅从转写文本抽数
          3. 冲突时 VLM 看图裁决

        use_region_guidance=True 且具备 vision+OCR：
          在以上之前增加 VLM bbox + 局部 OCR（Sniper），缺失时仍走全图 OCR+LLM。

        kwargs:
          region_expand_header: 仅 Sniper 路径；保守扩展裁剪框以纳入表头/行标签。
        """
        agent_vars = variables or kwargs.get("variables") or []
        base64_image = base64_image or kwargs.get("base64_image")
        use_context_image = kwargs.get("use_context_image", use_context_image)
        # 修复：尊重函数入参默认值，避免把增强链路无条件覆盖为 False。
        use_region_guidance = kwargs.get("use_region_guidance", use_region_guidance)
        region_expand_header = kwargs.get("region_expand_header", True)

        # 从 step 上下文推断变量
        step_ctx = get_step_context()
        step_text = (step_ctx or {}).get("step_text", "")
        next_step_text = (step_ctx or {}).get("next_step_text", "")
        inferred = _infer_variables_from_plan(step_text, next_step_text or None)
        
        # 获取语义查询映射（解耦模式）
        semantic_queries = _get_semantic_queries(step_text) if step_text else {}
        if semantic_queries:
            logger.info(f"[finance_extraction_skill] 解耦模式: 语义查询映射 = {semantic_queries}")

        if inferred:
            variables = inferred if next_step_text else list(dict.fromkeys(inferred + agent_vars))
        elif agent_vars:
            variables = agent_vars
        else:
            variables = []

        if not variables:
            return {
                "observation": "[ERROR] finance_extraction_skill requires 'variables' or step_context with formula.",
                "success": False,
            }

        # 获取图片
        if use_context_image and not base64_image:
            ctx_images = get_step_images_for_ocr()
            if ctx_images and len(ctx_images) > 0:
                base64_image = ctx_images[0]
            else:
                return {
                    "observation": "[ERROR] use_context_image=True but no images in context.",
                    "success": False,
                }

        if not base64_image or not str(base64_image).strip():
            return {
                "observation": "[ERROR] No image provided.",
                "success": False,
            }

        vlm_values: Dict[str, float] = {}
        ocr_values: Dict[str, float] = {}
        vlm_result: Dict[str, Any] = {}
        ocr_result: Dict[str, Any] = {}
        ocr_text = ""
        ocr_provenance: Dict[str, str] = {}
        location_hints: Dict[str, str] = {}

        if use_region_guidance and self._get_vision_llm() and self._get_ocr_client():
            # ==========================================
            # 可选：区域识别增强（Sniper）+ 全图兜底
            # ==========================================
            
            # Phase 1: VLM 提取 + 区域定位
            logger.info(f"[finance_extraction_skill] Phase 1 (VLM提取+区域定位): variables={variables}")
            vlm_result = await self._vlm_extract_with_regions(base64_image, variables)
            vlm_values = vlm_result.get("values", {})
            location_hints = vlm_result.get("location_hints", {}) or {}
            vlm_regions = vlm_result.get("regions", {})
            
            if vlm_regions:
                logger.info(f"[finance_extraction_skill] VLM定位到 {len(vlm_regions)} 个区域: {list(vlm_regions.keys())}")
            
            # Phase 2A: Sniper 模式（裁剪后局部 OCR）
            logger.info(f"[finance_extraction_skill] Phase 2A (Sniper OCR): variables={variables}")
            ocr_result = await self._ocr_region_extract(
                base64_image,
                variables,
                vlm_regions,
                region_expand_header=region_expand_header,
            )
            ocr_values = ocr_result.get("values", {}) or {}
            ocr_text = ocr_result.get("ocr_text", "") or ""
            ocr_provenance = dict(ocr_result.get("provenance") or {})

            # Phase 2B: Shotgun 兜底（全图 OCR + LLM 文本理解）
            missing_ocr = [
                v for v in variables if not _is_valid_number(ocr_values.get(v))
            ]
            if missing_ocr:
                logger.info(
                    f"[finance_extraction_skill] Phase 2B (Shotgun fallback): missing={missing_ocr}"
                )
                fallback_ocr = await self._ocr_semantic_extract(
                    base64_image,
                    missing_ocr,
                    semantic_queries if semantic_queries else None,
                    location_hints=location_hints,
                )
                fb_values = fallback_ocr.get("values", {}) or {}
                fb_raw = fallback_ocr.get("raw_values", {}) or {}
                fb_prov = fallback_ocr.get("provenance") or {}
                for v in missing_ocr:
                    if _is_valid_number(fb_values.get(v)):
                        ocr_values[v] = fb_values[v]
                        if "raw_values" not in ocr_result:
                            ocr_result["raw_values"] = {}
                        ocr_result["raw_values"][v] = fb_raw.get(v, "")
                        if v in fb_prov:
                            ocr_provenance[v] = fb_prov[v]
                if fallback_ocr.get("ocr_text"):
                    ocr_text = (
                        f"{ocr_text}\n\n--- fallback_ocr ---\n{fallback_ocr.get('ocr_text')}"
                        if ocr_text
                        else fallback_ocr.get("ocr_text")
                    )

            if ocr_result.get("error") and not vlm_values and not any(
                _is_valid_number(ocr_values.get(v)) for v in variables
            ):
                return {
                    "observation": f"[OCR_ERROR] {ocr_result['error']}",
                    "success": False,
                    "extraction_result": ocr_result,
                }
        else:
            # ==========================================
            # 默认：全图 OCR（清洗）+ LLM，VLM 整图候选
            # ==========================================
            
            # Phase 1: VLM 先提取（若有 vision 配置）
            if self._get_vision_llm():
                logger.info(f"[finance_extraction_skill] Phase 1 (VLM 提取): variables={variables}")
                # 传递语义查询映射（解耦模式）
                vlm_result = await self._vlm_extract(base64_image, variables, semantic_queries if semantic_queries else None)
                vlm_values = vlm_result.get("values", {})
                location_hints = vlm_result.get("location_hints", {}) or {}

            # Phase 2: 全图 OCR 转写 + 清洗 + LLM 映射
            logger.info(f"[finance_extraction_skill] Phase 2 (全图 OCR + LLM): variables={variables}")
            # 关键改进：传递语义查询给OCR，让OCR也知道要提取什么
            ocr_result = await self._ocr_semantic_extract(
                base64_image,
                variables,
                semantic_queries if semantic_queries else None,
                location_hints=location_hints,
            )
            ocr_values = ocr_result.get("values", {})
            ocr_text = ocr_result.get("ocr_text", "")
            ocr_provenance = dict(ocr_result.get("provenance") or {})
            
            if ocr_result.get("error") and not vlm_values:
                return {
                    "observation": f"[OCR_ERROR] {ocr_result['error']}",
                    "success": False,
                    "extraction_result": ocr_result,
                }

        # Phase 3: 合并与冲突裁决
        extracted: Dict[str, float] = {}
        raw_values: Dict[str, str] = {}
        needs_calculation: Dict[str, Dict[str, float]] = {}
        
        for var_name in variables:
            vv = vlm_values.get(var_name)
            ov = ocr_values.get(var_name)
            
            if vv == "NEEDS_CALCULATION":
                logger.info(f"[finance_extraction_skill] 变量 {var_name}: VLM标记为需要计算")
                needs_calculation[var_name] = {}
                for key, val in vlm_values.items():
                    if key.startswith(f"{var_name}_") and _is_valid_number(val):
                        comp_name = key[len(f"{var_name}_"):]
                        needs_calculation[var_name][comp_name] = val
                        extracted[key] = val
                        raw_values[key] = str(val)
                        logger.info(f"[finance_extraction_skill] 提取计算组件: {key} = {val}")
                continue
            
            vv_valid = _is_valid_number(vv)
            ov_valid = _is_valid_number(ov)
            semantic_query = semantic_queries.get(var_name) if semantic_queries else None
            location_hint = location_hints.get(var_name, "")
            location_hint_high_risk = _is_high_risk_location_hint(location_hint, semantic_query or var_name)
            explicit_axis_constraint = _query_has_explicit_axis_constraint(
                semantic_query or var_name
            )
            
            logger.info(f"[finance_extraction_skill] 变量 {var_name}: VLM值={vv}({vv_valid}), OCR值={ov}({ov_valid})")
            
            if vv_valid and ov_valid:
                if abs(vv - ov) < 1e-6:
                    extracted[var_name] = vv
                    raw_values[var_name] = vlm_result.get("raw_values", {}).get(var_name, str(vv))
                    logger.info(f"[finance_extraction_skill] 变量 {var_name}: VLM和OCR一致，使用值 {vv}")
                elif (
                    ocr_provenance.get(var_name) == "structure"
                    and not explicit_axis_constraint
                ):
                    extracted[var_name] = ov
                    raw_values[var_name] = ocr_result.get("raw_values", {}).get(var_name, str(ov))
                    logger.info(
                        f"[finance_extraction_skill] 变量 {var_name}: VLM({vv})与OCR({ov})冲突，"
                        f"优先采用结构化表格解析的 OCR（避免整图 VLM 错行）"
                    )
                else:
                    if ocr_provenance.get(var_name) == "structure" and explicit_axis_constraint:
                        logger.info(
                            f"[finance_extraction_skill] 变量 {var_name}: 请求含明确时间/期间轴，"
                            "结构化 OCR 与 VLM 冲突时改走看图裁决，避免跨列/跨年误绑。"
                        )
                    final = await self._vlm_verify(base64_image, var_name, vv, ov, semantic_query)
                    extracted[var_name] = final
                    raw_values[var_name] = str(final)
                    logger.info(
                        f"[finance_extraction_skill] 变量 {var_name}: VLM({vv})和OCR({ov})冲突，VLM裁决为 {final}"
                    )
            elif vv_valid:
                if location_hint_high_risk:
                    logger.warning(
                        f"[finance_extraction_skill] 变量 {var_name}: 仅VLM有效，但位置提示高风险，拒绝盲信 ({location_hint})"
                    )
                else:
                    extracted[var_name] = vv
                    raw_values[var_name] = vlm_result.get("raw_values", {}).get(var_name, str(vv))
                    logger.info(f"[finance_extraction_skill] 变量 {var_name}: 仅VLM有效，使用值 {vv}")
            elif ov_valid:
                extracted[var_name] = ov
                raw_values[var_name] = ocr_result.get("raw_values", {}).get(var_name, str(ov))
                logger.info(f"[finance_extraction_skill] 变量 {var_name}: 仅OCR有效，使用值 {ov}")
            else:
                logger.warning(f"[finance_extraction_skill] 变量 {var_name}: VLM和OCR都未能提取有效值")

        missing = [v for v in variables if v not in extracted]

        # ==========================================
        # Phase 4: 回退机制 - 当VLM和OCR都失败时
        # ==========================================
        if missing and ocr_text:
            logger.info(f"[finance_extraction_skill] Phase 4 (回退机制): 尝试从OCR文本中提取缺失变量 {missing}")
            
            all_numbers = re.findall(r'[\d,]+\.?\d*\s*[万亿%]?', ocr_text)
            if all_numbers:
                logger.info(f"[finance_extraction_skill] OCR文本中发现 {len(all_numbers)} 个数值: {all_numbers[:10]}...")
            
            for var_name in missing:
                var_synonyms = _expand_var_synonyms(var_name)
                for syn in var_synonyms:
                    syn_norm = _normalize_var_key(syn)
                    for line in ocr_text.splitlines():
                        line_lower = line.lower()
                        if syn_norm in _normalize_var_key(line_lower):
                            nums = re.findall(r'[\d,]+\.?\d*\s*[万亿%]?', line)
                            if nums:
                                num = _apply_unit_conversion(nums[0])
                                if num is not None:
                                    extracted[var_name] = num
                                    raw_values[var_name] = nums[0]
                                    logger.info(f"[finance_extraction_skill] 回退提取成功: {var_name} = {num} (via synonym '{syn}')")
                                    break
                    if var_name in extracted:
                        break

        missing = [v for v in variables if v not in extracted]

        # ==========================================
        # 核心改动：数据落地与状态持久化 (State Persistence)
        # 坚决不让 Agent 自己写赋值代码，由 Skill 自动注入
        # ==========================================

        # 1. 如果什么都没提取到，直接报错拦截
        if not extracted:
            vlm_status = "成功" if vlm_values else "失败"
            ocr_status = "成功" if ocr_values else "失败"
            return {
                "observation": f"[EXTRACTION_FAILED] 未能提取任何变量。VLM: {vlm_status}, OCR: {ocr_status}. 缺失: {', '.join(missing)}. 请检查概念词是否准确。",
                "success": False,
                "extraction_result": {"vlm_values": vlm_values, "ocr_values": ocr_values}
            }

        # 2. 自动注入 Python 状态环境（修复：正确处理异步调用）
        auto_save_status = ""
        python_executor = get_shared_python_execute()

        if python_executor:
            try:
                assign_lines = [f"{k} = {v}" for k, v in extracted.items()]
                assign_code = "\n".join(assign_lines)

                # 修复：判断是否为异步函数并正确 await
                if hasattr(python_executor, 'run'):
                    if inspect.iscoroutinefunction(python_executor.run):
                        await python_executor.run(assign_code)
                    else:
                        python_executor.run(assign_code)
                elif hasattr(python_executor, 'execute'):
                    if inspect.iscoroutinefunction(python_executor.execute):
                        await python_executor.execute(assign_code)
                    else:
                        python_executor.execute(assign_code)

                auto_save_status = "\n[SYSTEM ACTION] ALL extracted variables have been AUTOMATICALLY saved to the Python environment. They are ready to use."
                logger.info(f"[finance_extraction_skill] 自动注入 Python 变量成功: {list(extracted.keys())}")
            except Exception as e:
                logger.error(f"[finance_extraction_skill] 自动注入 Python 失败: {e}")
                auto_save_status = f"\n[SYSTEM WARNING] Auto-save failed ({e}). You MUST call python_execute manually to store: {', '.join(extracted.keys())}"
        else:
            auto_save_status = "\n[SYSTEM WARNING] No shared Python executor found. You MUST call python_execute manually to store these variables."

        calc_info = ""
        if needs_calculation:
            calc_info = "\n\n[NEEDS_CALCULATION] The following variables need calculation:"
            for var, components in needs_calculation.items():
                comp_str = ", ".join([f"{k}={v}" for k, v in components.items()])
                calc_info += f"\n  {var}: components extracted = {comp_str}"

        if missing and needs_calculation:
            result_msg = f"""[PARTIAL_SUCCESS] 部分变量提取成功（部分需要计算）：
{chr(10).join(f'  {k}: {raw_values.get(k, str(v))} → {v}' for k, v in extracted.items() if not k.endswith('_' + k.split('_')[-1]))}{calc_info}

Missing Variables: {', '.join(missing)}{auto_save_status}"""
        elif missing:
            result_msg = f"""[PARTIAL_SUCCESS] 部分变量提取成功：
{chr(10).join(f'  {k}: {raw_values.get(k, str(v))} → {v}' for k, v in extracted.items())}

Missing Variables: {', '.join(missing)}{auto_save_status}"""
        elif needs_calculation:
            result_msg = f"""[EXTRACTED_SUCCESS] 所有变量提取成功（部分需要计算）！
{chr(10).join(f'  {k}: {raw_values.get(k, str(v))} → {v}' for k, v in extracted.items())}{calc_info}{auto_save_status}"""
        else:
            result_msg = f"""[EXTRACTED_SUCCESS] 所有变量提取成功！
{chr(10).join(f'  {k}: {raw_values.get(k, str(v))} → {v}' for k, v in extracted.items())}{auto_save_status}"""

        return {
            "observation": result_msg,
            "success": True,
            "extracted_values": extracted,
            "raw_values": raw_values,
            "missing": missing,
            "needs_calculation": needs_calculation,
        }
