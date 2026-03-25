"""
OcrExtract - 基于 DeepSeek-OCR 的图像文字提取工具。

遵循 OpenManus 统一工具格式，支持单图与多图（批量）输入。
用于表格、图表等金融文档的预处理，为 Planning 与 Multimodal 提供结构化文本输入。

输出格式：统一 Markdown，包含：
  - 主内容：表格用 Markdown 表格，段落用段落
  - 空间坐标：每个文本区域附带 bbox [x,y,w,h]，归一化坐标 0~1（相对图片宽高）

Agent 调用支持：当 MultimodalAgent 执行步骤时，Flow 会注入当前步骤的图片到上下文。
Agent 可调用 ocr_extract(use_context_image=True) 使用当前步骤图片，无需传 base64。
"""
import contextvars
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from app.config import config
from app.logger import logger
from app.tool.base import BaseTool

# 当前步骤的图片上下文，由 Flow 在执行 multimodal 步骤前注入，供 Agent 调用 ocr_extract(use_context_image=True) 使用
_step_images_ctx: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
    "ocr_step_images", default=None
)


def set_step_images_for_ocr(images: Optional[List[str]]) -> None:
    """由 Flow 在执行 multimodal 步骤前调用，注入当前步骤可用的图片。"""
    _step_images_ctx.set(images)


def get_step_images_for_ocr() -> Optional[List[str]]:
    """获取当前上下文中由 Flow 注入的图片列表。"""
    return _step_images_ctx.get()


# 当前步骤上下文（含公式），供 Skill 推断需提取的变量，减少对错误 plan 的依赖
_step_context_ctx: contextvars.ContextVar[Optional[Dict[str, str]]] = contextvars.ContextVar(
    "step_context", default=None
)


def set_step_context(ctx: Optional[Dict[str, str]]) -> None:
    """由 Flow 在执行 multimodal 步骤前调用，注入 step_text 与 next_step_text（含公式）。"""
    _step_context_ctx.set(ctx)


def get_step_context() -> Optional[Dict[str, str]]:
    """获取当前步骤上下文，供 Skill 推断变量。"""
    return _step_context_ctx.get()


_OCR_DESCRIPTION = """Extract text from image(s). Output: unified Markdown with tables and per-region bbox [x,y,w,h] (0~1 normalized). Supports: (1) base64_image for single; (2) base64_images for batch; (3) use_context_image=true for current step's image(s). Returns {markdown, regions, success} for single, {results: [{markdown, regions, index}], success} for batch."""


def _ensure_data_url(base64_image: str) -> str:
    """Ensure base64 image has data URL prefix for OpenAI-compatible API."""
    s = (base64_image or "").strip()
    if not s:
        return ""
    if s.startswith("data:"):
        return s
    return f"data:image/png;base64,{s}"


class OcrExtract(BaseTool):
    """OCR tool using DeepSeek-OCR model via OpenAI-compatible API."""

    name: str = "ocr_extract"
    description: str = _OCR_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "base64_image": {
                "type": "string",
                "description": "Single base64-encoded image. Use for one image.",
            },
            "base64_images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of base64-encoded images for batch extraction. Use for multi-image tasks.",
            },
            "use_context_image": {
                "type": "boolean",
                "description": "When true, use the current step's image(s) injected by Flow. Use when Agent is in a multimodal step and you need OCR on the image being viewed.",
            },
        },
        "required": [],
    }

    def __init__(self, **data):
        super().__init__(**data)
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-init OpenAI client from OCR config."""
        if self._client is not None:
            return self._client
        ocr_cfg = self._get_ocr_config()
        if not ocr_cfg:
            raise RuntimeError(
                "OCR config not found. Add [ocr] section in config.toml with model, base_url, api_key."
            )
        self._client = AsyncOpenAI(
            api_key=ocr_cfg.api_key,
            base_url=ocr_cfg.base_url,
        )
        return self._client

    def _get_ocr_config(self):
        """Get OCR config from app config."""
        return getattr(config, "ocr_config", None)

    # DeepSeek-OCR 训练分布使用固定指令；随意英文描述易导致输出「说明性废话」而非版面转写。
    # 见官方 README: document -> <|grounding|>Convert the document to markdown.
    _OCR_PROMPT = "<|grounding|>Convert the document to markdown."

    def _assess_ocr_quality(self, text: str) -> Tuple[bool, str, float]:
        """
        评估OCR输出质量。
        返回: (是否可用, 原因, 质量分数0-1)
        """
        if not text:
            return False, "Empty output", 0.0
        
        text_len = len(text)
        
        # 1. 检查长度
        if text_len < 30:
            return False, f"Too short ({text_len} chars)", 0.1
        
        # 2. 检查非ASCII字符比例
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / text_len
        if non_ascii_ratio > 0.7:
            return False, f"High non-ASCII ratio ({non_ascii_ratio:.2%})", 0.2
        
        # 3. 检查重复内容（行级重复）
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) > 3:
            unique_lines = set(lines)
            duplication_ratio = 1 - (len(unique_lines) / len(lines))
            if duplication_ratio > 0.6:  # 超过60%的行重复
                return False, f"High duplication ({duplication_ratio:.2%})", 0.3
        
        # 4. 检查是否包含财务/数据特征
        financial_indicators = [
            r'\$[\d,]+',  # 美元金额
            r'\d+\.?\d*\s*[万亿]',  # 中文单位
            r'\d{4}',  # 年份
            r'\|',  # 表格分隔符
            r'(?:revenue|sales|profit|expense|income|assets|liabilities)',  # 财务关键词
            r'(?:收入|销售|利润|费用|资产|负债)',
        ]
        has_financial_content = any(re.search(pattern, text, re.IGNORECASE) for pattern in financial_indicators)
        
        # 5. 检查乱码模式和HTML标签
        garbage_patterns = [
            r'\[\].*?https?://',  # 带括号的URL乱码
            r'horizontal\s+z-sticky',
            r'Copy\s*code',
            r'layers:\s*$',
            r'<i></i>',  # 空HTML标签重复
            r'\]\].*?\[\[',  # 乱码括号
            r'</td>\s*</tr>\s*</table>',  # 混乱的HTML闭合
        ]
        garbage_count = sum(1 for p in garbage_patterns if re.search(p, text, re.IGNORECASE))
        
        # 计算质量分数
        quality_score = 1.0
        quality_score -= non_ascii_ratio * 0.3
        quality_score -= garbage_count * 0.15
        if not has_financial_content:
            quality_score -= 0.3
        
        quality_score = max(0.0, min(1.0, quality_score))
        
        # 质量低于阈值认为不可用
        if quality_score < 0.4:
            return False, f"Low quality score ({quality_score:.2f})", quality_score
        
        return True, f"Quality score {quality_score:.2f}", quality_score

    def _clean_ocr_output(self, raw: str) -> str:
        """轻量级清理 OCR 输出，保留原始内容供下游处理。"""
        text = (raw or "").strip()
        
        # 仅移除明显的代码片段标记，保留所有表格和文本内容
        patterns_to_remove = [
            r"^```\w*\n",  # 代码块开始标记
            r"```$",       # 代码块结束标记
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        
        # 质量评估（仅用于日志，不拒绝输出）
        is_usable, reason, score = self._assess_ocr_quality(text)
        if not is_usable:
            logger.warning(f"OCR output quality warning: {reason}")
        else:
            logger.info(f"OCR output quality: {reason}")
        
        # 保留原始格式（HTML或Markdown），让下游Skill处理
        return text.strip()
    
    def _html_to_markdown(self, html: str) -> str:
        """将HTML表格转换为Markdown格式。"""
        # 先保护特殊字符
        text = html
        
        # 转换表格
        # 提取所有行
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', text, re.DOTALL)
        if not rows:
            return html
        
        markdown_lines = []
        for i, row in enumerate(rows):
            # 提取单元格
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            if not cells:
                cells = re.findall(r'<th[^>]*>(.*?)</th>', row, re.DOTALL)
            
            # 清理单元格内容
            clean_cells = []
            for cell in cells:
                # 移除所有HTML标签，但保留内容
                cell = re.sub(r'<[^>]+>', '', cell)
                # 规范化空格
                cell = re.sub(r'\s+', ' ', cell).strip()
                clean_cells.append(cell)
            
            if clean_cells:
                markdown_lines.append('| ' + ' | '.join(clean_cells) + ' |')
                # 添加表头分隔线
                if i == 0:
                    markdown_lines.append('|' + '|'.join(['---' for _ in clean_cells]) + '|')
        
        return '\n'.join(markdown_lines) if markdown_lines else html

    def _parse_ocr_response(self, raw: str) -> Tuple[str, List[Dict[str, Any]]]:
        """解析 OCR 返回，提取 markdown 内容与 regions。"""
        # 先清理输出
        text = self._clean_ocr_output(raw)
        regions: List[Dict[str, Any]] = []

        # 尝试解析 ## Regions 或 ## 坐标 区块
        regions_match = re.search(
            r"##\s*(?:Regions|坐标)[^\n]*\n(.*?)(?=##|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if regions_match:
            block = regions_match.group(1).strip()
            # 每行: text | x,y,w,h 或 text | x y w h
            for line in block.splitlines():
                line = line.strip()
                if "|" in line:
                    parts = line.split("|", 1)
                    if len(parts) == 2:
                        region_text = parts[0].strip()
                        coords = re.findall(r"[\d.]+", parts[1])
                        if len(coords) >= 4:
                            try:
                                x, y, w, h = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                                if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                                    regions.append({"text": region_text, "bbox": [x, y, w, h]})
                            except (ValueError, IndexError):
                                pass

        # 若有 ## Content，主内容取 Content 块；否则用全文（去除 Regions 块避免重复）
        content_match = re.search(r"##\s*Content[^\n]*\n(.*?)(?=##\s*(?:Regions|坐标)|\Z)", text, re.DOTALL | re.IGNORECASE)
        markdown = content_match.group(1).strip() if content_match else text
        return markdown, regions

    async def _extract_single(self, base64_image: str) -> Dict[str, Any]:
        """Extract text from a single image. Internal helper."""
        url = _ensure_data_url(str(base64_image))
        if not url:
            return {"text": "", "markdown": "", "regions": [], "success": False, "error": "Invalid base64_image."}

        ocr_cfg = self._get_ocr_config()
        if not ocr_cfg:
            return {
                "text": "",
                "markdown": "",
                "regions": [],
                "success": False,
                "error": "OCR config not found. Add [ocr] in config.toml.",
            }

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=ocr_cfg.model,
                max_tokens=4096,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": url},
                            },
                            {
                                "type": "text",
                                "text": self._OCR_PROMPT,
                            },
                        ],
                    }
                ],
            )
            choice = response.choices[0] if response.choices else None
            raw = (choice.message.content if choice and choice.message else "").strip()
            markdown, regions = self._parse_ocr_response(raw)
            # text 保持兼容 Flow/Planning 的 item.get("text","")
            return {
                "text": markdown,
                "markdown": markdown,
                "regions": regions,
                "success": True,
            }
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return {"text": "", "markdown": "", "regions": [], "success": False, "error": str(e)}

    async def execute(
        self,
        base64_image: Optional[str] = None,
        base64_images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract text from image(s) using DeepSeek-OCR.

        Output: unified Markdown (text/markdown) + regions (bbox per text region).
        Supports:
        - Single: base64_image -> {text, markdown, regions, success}
        - Batch: base64_images -> {results: [{text, markdown, regions, index}], success}

        Args:
            base64_image: Single base64-encoded image.
            base64_images: List of base64-encoded images for batch.
            **kwargs: Additional args (e.g. from tool_input).

        Returns:
            Single: {text, markdown, regions, success} or {... error}
            Batch: {results: [...], success} (each item has text, markdown, regions, index)
        """
        base64_image = base64_image or kwargs.get("base64_image")
        base64_images = base64_images if base64_images is not None else kwargs.get("base64_images")
        use_context_image = kwargs.get("use_context_image", False)

        # use_context_image: 使用 Flow 注入的当前步骤图片（供 Agent 在 multimodal 步骤中调用）
        if use_context_image:
            ctx_images = get_step_images_for_ocr()
            if ctx_images and len(ctx_images) > 0:
                base64_images = ctx_images
            elif ctx_images is not None:
                return {"text": "", "success": False, "error": "No images in context."}
            else:
                return {"text": "", "success": False, "error": "use_context_image=True but no images in context. Ensure you are in a multimodal step with image(s)."}

        # Batch mode: base64_images provided (even empty list returns results=[])
        if base64_images is not None:
            results: List[Dict[str, Any]] = []
            for i, img in enumerate(base64_images):
                r = await self._extract_single(img)
                results.append({
                    "text": r.get("text", ""),
                    "markdown": r.get("markdown", r.get("text", "")),
                    "regions": r.get("regions", []),
                    "index": i,
                    "success": r.get("success", False),
                })
            return {"results": results, "success": True}

        # Single mode: base64_image provided
        if base64_image and str(base64_image).strip():
            return await self._extract_single(base64_image)

        return {
            "text": "",
            "success": False,
            "error": "ocr_extract requires 'base64_image' or 'base64_images' argument.",
        }
