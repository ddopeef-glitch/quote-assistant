from __future__ import annotations

from decimal import Decimal
from io import BytesIO
import math
import re
from typing import Iterable, Protocol

import fitz
import pandas as pd
from openpyxl import load_workbook
from PIL import Image, ImageOps

from .core import ProductRecord, normalize_price_marker, normalize_product_name, normalize_specification
from .usage_diagnostics import emit_diagnostic


class AITextExtractor(Protocol):
    def extract_products(self, text: str) -> list[ProductRecord]: ...


class AIVisionExtractor(Protocol):
    def extract_products(self, image_bytes: bytes, context: str | None = None) -> list[ProductRecord]: ...


class MultiFrameImageError(ValueError):
    """The image input contains frames that this route does not process."""


DEFAULT_PRODUCT_TERMS = (
    "丝接机械三通", "丝接机械四通", "沟槽机械三通", "沟槽机械四通",
    "45°弯头", "转换法兰", "异径三通", "异径四通", "机械三通", "机械四通",
    "刚性卡箍", "沟槽机三", "沟槽机四", "大小头", "盲板", "卡箍", "管箍",
    "三通", "四通", "弯头", "转兰", "机三", "机四", "中大",
)
GRADE_WORDS = "蓝袋|黑袋|好|中|a|A|b|B"
CATEGORY_TITLE_RE = re.compile(r"^(?P<category>沟槽|玛钢|球铁)\s*[:：]?$")
GRADE_TITLE_PREFIX_RE = re.compile(r"^[Ll]\s*(?P<grade>中)\s*[:：=]\s*")
AI_REVIEW_WARNING = "OCR/规则识别结果必须人工核对"
AI_FIELDS = {
    "product_name_raw", "specification_raw", "quantity",
    "grade_or_bag_raw", "source_text", "uncertain_fields",
}
IMAGE_SUFFIXES = {"png", "jpg", "jpeg", "webp", "bmp"}
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "BMP")
MAX_IMAGE_PIXELS = 24_000_000
MAX_IMAGE_SIDE = 2400
MAX_PDF_PAGES = 20
MAX_PDF_PIXELS = 24_000_000


def _record(
    source_type: str,
    source_name: str,
    location: str,
    raw: str,
    name: str,
    spec: str,
    qty: str,
    grade: str = "",
    *,
    grade_markers: dict[str, str] | None = None,
    confidence: float = 0.75,
    extraction_method: str = "规则",
    warnings: list[str] | None = None,
) -> ProductRecord:
    quantity = Decimal(str(qty))
    return ProductRecord(
        source_type=source_type,
        source_name=source_name,
        source_location=location,
        original_text=raw,
        original_product_name=name,
        product_name=normalize_product_name(name),
        original_specification=spec,
        specification=normalize_specification(spec),
        quantity_cartons=int(quantity) if quantity == quantity.to_integral_value() else quantity,
        original_grade_text=grade or None,
        price_marker=normalize_price_marker(grade, grade_markers) or None,
        confidence=confidence,
        extraction_method=extraction_method,
        warnings=warnings if warnings is not None else [],
    )


def extract_text(
    text: str,
    source_name: str = "粘贴文字",
    source_type: str = "text",
    product_terms: Iterable[str] | None = None,
    grade_markers: dict[str, str] | None = None,
) -> list[ProductRecord]:
    records: list[ProductRecord] = []
    context_name = ""
    context_category: str | None = None
    context_grade = ""
    product_words = "|".join(re.escape(term) for term in sorted(set(product_terms or DEFAULT_PRODUCT_TERMS), key=len, reverse=True))
    pattern_spec_first = re.compile(
        rf"(?P<spec>\d+(?:[xX×*\-]\d+[GSgs]?)*(?:[GSgs])?)\s*(?P<name>{product_words})"
        rf"\s*(?P<grade_before>{GRADE_WORDS})?\s*(?P<qty>\d+(?:\.\d+)?)\s*(?:件|箱)\s*(?P<grade_after>{GRADE_WORDS})?"
    )
    pattern_name_first = re.compile(
        rf"(?P<name>{product_words})\s*(?P<spec>\d+(?:[xX×*\-]\d+[GSgs]?)*(?:[GSgs])?)"
        rf"\s+(?P<grade_before>{GRADE_WORDS})?\s*(?P<qty>\d+(?:\.\d+)?)\s*(?:件|箱)\s*(?P<grade_after>{GRADE_WORDS})?"
    )
    context_line = re.compile(
        rf"(?P<spec>\d+(?:[xX×*\-]\d+[GSgs]?)*(?:[GSgs])?)的?(?P<qty>\d+(?:\.\d+)?)\s*(?:件|箱)\s*(?P<grade>{GRADE_WORDS})?"
    )
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        category_title = CATEGORY_TITLE_RE.fullmatch(stripped)
        if category_title:
            context_name = ""
            context_category = "球铁" if category_title["category"] == "球铁" else None
            continue
        grade_title = GRADE_TITLE_PREFIX_RE.match(stripped)
        if grade_title:
            context_grade = grade_title["grade"]
            stripped = stripped[grade_title.end() :].strip()
            if not stripped:
                continue
        if re.fullmatch(product_words, stripped):
            context_name = stripped
            continue
        if not re.search(r"\d", stripped):
            context_name = ""
        found_spans: set[tuple[int, int]] = set()
        for pattern in (pattern_spec_first, pattern_name_first):
            for match in pattern.finditer(stripped):
                if match.span() in found_spans:
                    continue
                found_spans.add(match.span())
                explicit_grade = match["grade_before"] or match["grade_after"] or ""
                record = _record(
                    source_type, source_name, f"第 {line_number} 行", match.group(0),
                    match["name"], match["spec"], match["qty"], explicit_grade or context_grade,
                    grade_markers=grade_markers,
                )
                record.category = context_category
                if explicit_grade and context_grade and normalize_price_marker(explicit_grade, grade_markers) != normalize_price_marker(context_grade, grade_markers):
                    record.price_marker = None
                    record.warnings.append("行级等级与标题等级冲突，已阻断自动 a/b")
                records.append(record)
        if context_name:
            for match in context_line.finditer(stripped):
                explicit_grade = match["grade"] or ""
                record = _record(
                    source_type, source_name, f"第 {line_number} 行", match.group(0),
                    context_name, match["spec"], match["qty"], explicit_grade or context_grade, grade_markers=grade_markers,
                )
                record.category = context_category
                if explicit_grade and context_grade and normalize_price_marker(explicit_grade, grade_markers) != normalize_price_marker(context_grade, grade_markers):
                    record.price_marker = None
                    record.warnings.append("行级等级与标题等级冲突，已阻断自动 a/b")
                records.append(record)
    return records


COLUMN_ALIASES = {
    "name": {"产品名称", "品名", "存货名称", "存货全名", "名称"},
    "spec": {"产品规格", "规格", "规格型号", "存货规格", "型号"},
    "quantity": {"数量", "件数", "数量（件）", "数量(件)", "箱数"},
    "grade": {"袋型", "质量等级", "等级", "蓝袋黑袋", "好中", "价格标记"},
}


def extract_excel(
    data: bytes,
    source_name: str,
    column_map: dict[str, str] | None = None,
    grade_markers: dict[str, str] | None = None,
) -> list[ProductRecord]:
    records: list[ProductRecord] = []
    book = pd.ExcelFile(BytesIO(data))
    for sheet_name in book.sheet_names:
        raw = pd.read_excel(book, sheet_name=sheet_name, header=None)
        header_row = None
        mapping: dict[str, int] = {}
        for row_index in range(min(20, len(raw))):
            cells = [str(value).strip() if pd.notna(value) else "" for value in raw.iloc[row_index]]
            trial = {}
            for field, aliases in COLUMN_ALIASES.items():
                match = next((i for i, cell in enumerate(cells) if cell in aliases), None)
                if match is not None:
                    trial[field] = match
            if {"name", "spec", "quantity"}.issubset(trial):
                header_row, mapping = row_index, trial
                break
        if column_map and header_row is not None:
            headers = [str(value).strip() for value in raw.iloc[header_row]]
            mapping.update({field: headers.index(name) for field, name in column_map.items() if name in headers})
        if header_row is None:
            continue
        for row_index in range(header_row + 1, len(raw)):
            row = raw.iloc[row_index]
            name, spec, qty = row.iloc[mapping["name"]], row.iloc[mapping["spec"]], row.iloc[mapping["quantity"]]
            if pd.isna(name) or pd.isna(spec) or pd.isna(qty):
                continue
            grade = "" if "grade" not in mapping or pd.isna(row.iloc[mapping["grade"]]) else str(row.iloc[mapping["grade"]])
            try:
                records.append(_record("excel", source_name, f"{sheet_name}!{row_index + 1}", " | ".join(map(str, row.tolist())), str(name), str(spec), str(qty), grade, grade_markers=grade_markers))
            except ValueError:
                continue
    return records


def validate_image(data: bytes) -> bytes:
    try:
        image = Image.open(BytesIO(data), formats=IMAGE_FORMATS)
    except Image.DecompressionBombError as error:
        raise ValueError("图片像素超过安全上限") from error
    with image:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("图片像素超过 2400 万，请缩小后上传")
        if getattr(image, "n_frames", 1) > 1:
            raise MultiFrameImageError("多帧图片不支持")
        image = ImageOps.exif_transpose(image)
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        output = BytesIO()
        image.convert("RGB").save(output, "JPEG", quality=90)
        return output.getvalue()


def _pdf_pages(document):
    if len(document) > MAX_PDF_PAGES:
        raise ValueError(f"PDF 超过 {MAX_PDF_PAGES} 页，请拆分后上传")
    return enumerate(document, 1)


def _render_pdf_page(page, remaining_pixels):
    width, height = page.rect.width, page.rect.height
    if not all(math.isfinite(size) and size > 0 for size in (width, height)):
        raise ValueError("PDF 页面尺寸无效")
    scale = min(2, MAX_IMAGE_SIDE / width, MAX_IMAGE_SIDE / height)
    pixels = math.ceil(width * scale) * math.ceil(height * scale)
    if pixels > remaining_pixels:
        raise ValueError("PDF 累计渲染像素超过 2400 万，请拆分后上传")
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    return pixmap.tobytes("png"), pixels


def extract_evidence_units(filename: str, data: bytes, ocr=None, *, on_stage=None, on_diagnostic=None) -> list[dict]:
    """Keep every nonempty source row/block, without a product vocabulary filter."""
    units = []

    def append(text, **location):
        if not isinstance(text, str):
            raise ValueError("原始识别内容不是文字")
        if text.strip():
            units.append({"unit_id": f"u{len(units) + 1}", "text": text, **location})
            emit_diagnostic(on_diagnostic, "evidence_unit", unit=units[-1])

    def image_units(image_bytes, page):
        if ocr is None:
            raise ValueError("缺少本地 OCR，未识别此来源")
        # _blocks only unwraps raw block metadata; it does not parse products.
        from .ocr_pipeline import RuleParser
        image_bytes = validate_image(image_bytes)
        if on_stage:
            on_stage("正在识别图片…")
        emit_diagnostic(on_diagnostic, "ocr_started", page=page, reading_method="local_ocr")
        try:
            raw = ocr.extract_raw(image_bytes)
        except Exception as error:
            emit_diagnostic(on_diagnostic, "ocr_failed", page=page, error_type=type(error).__name__, error=str(error))
            raise
        emit_diagnostic(on_diagnostic, "ocr_returned", page=page, raw=raw)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"第 {page} 页本地 OCR 返回结构异常，需人工处理")
        for document in raw:
            result = document.get("res") if isinstance(document, dict) else None
            blocks = result.get("parsing_res_list") if isinstance(result, dict) else None
            if not isinstance(blocks, list) or any(
                not isinstance(block, dict) or not isinstance(block.get("block_content"), str)
                for block in blocks
            ):
                raise ValueError(f"第 {page} 页本地 OCR 返回结构异常，需人工处理")
        blocks = RuleParser._blocks({"raw": raw})
        before = len(units)
        for block in blocks:
            append(block["content"], page=page, ocr_page=block["document_index"] + 1,
                   bbox=block["bbox"], block_label=block["label"], block_order=block["order"], reading_method="local_ocr")
        if len(units) == before:
            raise ValueError(f"第 {page} 页本地 OCR 未返回文字，需人工处理")

    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix == "txt":
        for line, text in enumerate(data.decode("utf-8-sig").splitlines(), 1):
            append(text, line=line)
    elif suffix == "xlsx":
        workbook = load_workbook(BytesIO(data), read_only=True, data_only=False)
        try:
            for sheet in workbook:
                sheet.reset_dimensions()
                for row_number, row in enumerate(sheet.iter_rows(), 1):
                    values = [f"{cell.column_letter}={cell.value}" for cell in row if cell.value is not None]
                    if values:
                        append("\t".join(values), sheet=sheet.title, row=row_number)
        finally:
            workbook.close()
    elif suffix == "xls":
        try:
            with pd.ExcelFile(BytesIO(data)) as workbook:
                for sheet in workbook.sheet_names:
                    rows = pd.read_excel(workbook, sheet_name=sheet, header=None, dtype=object)
                    for index, row in rows.iterrows():
                        values = [f"列{column + 1}={value}" for column, value in enumerate(row) if pd.notna(value)]
                        append("\t".join(values), sheet=sheet, row=index + 1)
        except ImportError:
            raise ValueError("当前环境缺少 .xls 读取依赖，请另存为 .xlsx 后上传") from None
    elif suffix == "pdf":
        with fitz.open(stream=data, filetype="pdf") as document:
            remaining_pixels = MAX_PDF_PIXELS
            for page_number, page in _pdf_pages(document):
                blocks = [block for block in page.get_text("blocks") if block[6] == 0 and block[4].strip()]
                if blocks:
                    for block in blocks:
                        append(block[4], page=page_number, bbox=list(block[:4]), reading_method="pdf_text")
                if not blocks or page.get_image_info():
                    if ocr is None:
                        raise ValueError("缺少本地 OCR，未识别此来源")
                    image_bytes, pixels = _render_pdf_page(page, remaining_pixels)
                    remaining_pixels -= pixels
                    image_units(image_bytes, page_number)
    elif suffix in IMAGE_SUFFIXES:
        image_units(data, 1)
    else:
        raise ValueError("不支持的询价文件类型")
    if not units:
        raise ValueError("来源中没有可读取的文字，请人工处理")
    return units


def _validate_ai_items(payload: object) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != {"items"} or not isinstance(payload["items"], list):
        raise ValueError("AI 返回 JSON 必须是仅含 items 数组的对象")
    validated = []
    for index, item in enumerate(payload["items"], 1):
        if not isinstance(item, dict) or set(item) != AI_FIELDS:
            raise ValueError(f"AI 返回第 {index} 行字段不符合 schema")
        for field in ("product_name_raw", "specification_raw", "grade_or_bag_raw", "source_text"):
            if item[field] is not None and not isinstance(item[field], str):
                raise ValueError(f"AI 返回第 {index} 行 {field} 类型错误")
        quantity = item["quantity"]
        if quantity is not None and (isinstance(quantity, bool) or not isinstance(quantity, (int, float))):
            raise ValueError(f"AI 返回第 {index} 行 quantity 类型错误")
        uncertain = item["uncertain_fields"]
        if not isinstance(uncertain, list) or not all(field in AI_FIELDS - {"uncertain_fields"} for field in uncertain):
            raise ValueError(f"AI 返回第 {index} 行 uncertain_fields 类型错误")
        validated.append(item)
    return validated


def _records_from_ai(
    payload: object,
    context: str | None = None,
    model: str = "本地识别",
    grade_markers: dict[str, str] | None = None,
) -> list[ProductRecord]:
    records = []
    for item in _validate_ai_items(payload):
        name = item["product_name_raw"].strip() if item["product_name_raw"] else None
        spec = item["specification_raw"].strip() if item["specification_raw"] else None
        grade = item["grade_or_bag_raw"].strip() if item["grade_or_bag_raw"] else None
        marker = normalize_price_marker(grade, grade_markers)
        quantity = None if item["quantity"] is None else Decimal(str(item["quantity"]))
        warnings = [AI_REVIEW_WARNING]
        if item["uncertain_fields"]:
            warnings.append("AI 不确定字段：" + "、".join(item["uncertain_fields"]))
        if not name:
            warnings.append("缺少产品名，必须人工填写")
        if not spec:
            warnings.append("缺少规格，必须人工填写")
        if item["quantity"] is None or item["quantity"] <= 0:
            warnings.append("缺少有效数量，必须人工填写")
        if grade and not marker:
            warnings.append("等级/袋色原文未映射 a/A/b，必须人工选择")
        records.append(
            ProductRecord(
                source_type="ai",
                source_name=context or "AI 输入",
                source_location=None,
                original_text=item["source_text"] or "",
                original_product_name=name,
                product_name=name,
                original_specification=spec,
                specification=spec,
                quantity_cartons=None if quantity is None else int(quantity) if quantity == quantity.to_integral_value() else quantity,
                original_grade_text=grade,
                price_marker=marker or None,
                confidence=None,
                extraction_method=model,
                warnings=warnings,
            )
        )
    return records


def _with_source(
    records: list[ProductRecord],
    source_type: str,
    source_name: str,
    location_prefix: str | None = None,
    visual: bool = False,
) -> list[ProductRecord]:
    for record in records:
        record.source_type = source_type
        record.source_name = source_name
        if location_prefix:
            record.source_location = f"{location_prefix} / {record.source_location}" if record.source_location else location_prefix
        if visual and AI_REVIEW_WARNING not in record.warnings:
            record.warnings.append(AI_REVIEW_WARNING)
    return records


def extract_file(
    filename: str,
    data: bytes,
    ai_text: AITextExtractor | None = None,
    ai_vision: AIVisionExtractor | None = None,
    product_terms: Iterable[str] | None = None,
    grade_markers: dict[str, str] | None = None,
) -> list[ProductRecord]:
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix in {"xlsx", "xls"}:
        return extract_excel(data, filename, grade_markers=grade_markers)
    if suffix == "pdf":
        records: list[ProductRecord] = []
        with fitz.open(stream=data, filetype="pdf") as document:
            remaining_pixels = MAX_PDF_PIXELS
            for page_number, page in _pdf_pages(document):
                text = page.get_text().strip()
                page_records = (ai_text.extract_products(text) if ai_text else extract_text(text, filename, "pdf", product_terms, grade_markers)) if text else []
                if page_records:
                    records.extend(_with_source(page_records, "pdf", filename, f"第 {page_number} 页"))
                elif ai_vision:
                    image_bytes, pixels = _render_pdf_page(page, remaining_pixels)
                    remaining_pixels -= pixels
                    page_records = ai_vision.extract_products(validate_image(image_bytes), f"{filename} 第 {page_number} 页")
                    records.extend(_with_source(page_records, "pdf", filename, f"第 {page_number} 页", visual=True))
        return records
    if suffix in IMAGE_SUFFIXES:
        normalized = validate_image(data)
        if not ai_vision:
            return []
        records = ai_vision.extract_products(normalized, filename)
        return _with_source(records, "image", filename, "整张图片", visual=True)
    if suffix == "txt":
        text = data.decode("utf-8-sig", errors="replace")
        records = ai_text.extract_products(text) if ai_text else extract_text(text, filename, product_terms=product_terms, grade_markers=grade_markers)
        return _with_source(records, "text", filename)
    raise ValueError(f"不支持的文件类型：{filename}")
