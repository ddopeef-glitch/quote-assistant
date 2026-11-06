from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_CEILING, localcontext
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import secrets
from threading import Lock
from typing import Iterable
from zoneinfo import ZoneInfo

import fitz
from openpyxl import load_workbook
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MASTER_DATA_DIR = ROOT / "主数据"
OUTPUT_ROOT = ROOT / "data" / "output"
CUSTOMER_HISTORY_PATH = ROOT / "data" / "settings" / "customer_names.json"
SALES_ID_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
SALES_ID_PATTERN = re.compile(rf"QA-[{SALES_ID_ALPHABET}]{{5}}-[{SALES_ID_ALPHABET}]{{5}}")
QUOTE_ROWS_PER_PAGE = 7
BUSINESS_TIMEZONE = ZoneInfo("Asia/Shanghai")
CUSTOMER_HISTORY_LOCK = Lock()
PDF_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
PDF_BOLD_FONT_PATH = Path(r"C:\Windows\Fonts\msyhbd.ttc")
UI_CATEGORIES = {"沟槽", "玛钢", "球铁"}
BALL_IRON_PREFIXES = ("球热", "球铁")


@dataclass
class ProductRecord:
    source_type: str
    source_name: str
    source_location: str | None = None
    original_text: str = ""
    category: str | None = None
    original_product_name: str | None = None
    product_name: str | None = None
    original_specification: str | None = None
    specification: str | None = None
    quantity_cartons: int | Decimal | None = None
    original_grade_text: str | None = None
    price_marker: str | None = None
    grade_review_required: bool = False
    master_packing_quantity: int | None = None
    packing_quantity: int | None = None
    unit_price: Decimal | None = None
    total_price: Decimal | None = None
    confidence: float | None = None
    extraction_method: str = ""
    warnings: list[str] = field(default_factory=list)
    manually_modified: bool = False
    manually_confirmed: bool = False
    confirmed_inquiry: tuple[str, str, str, int, str] | None = None
    price_source_file: str | None = None
    price_source_sheet: str | None = None
    price_source_row: int | None = None
    master_fingerprint: str | None = None
    match_status: str = "pending"
    match_message: str = ""

    def to_dict(self) -> dict:
        row = asdict(self)
        row["warnings"] = "；".join(self.warnings)
        return row


@dataclass(frozen=True)
class PriceRecord:
    source_file: str
    sheet_name: str
    source_row: int
    product_name: str
    specification: str
    price_marker: str
    unit_price: Decimal | None
    packing_quantity: int | None
    aliases: tuple[str, ...] = ()
    category: str = ""
    blank_weight: Decimal | None = None
    finished_weight: Decimal | None = None
    pure_weight: Decimal | None = None
    notes: str = ""
    multi_target_aliases: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    status: str = "启用"


def clean_text(value: object) -> str:
    return re.sub(r"\s+", "", "" if value is None else str(value)).strip()


def normalize_specification(value: object) -> str:
    text = clean_text(value).lower().replace("×", "x").replace("*", "x").replace("-", "x")
    return re.sub(r"(?<=\d)[gs](?=$|x)", "", text)


def normalize_product_name(value: object, aliases: dict[str, object] | None = None) -> str:
    name = clean_text(value).lower()
    shortened = name.removeprefix("demo")
    normalized = "90°弯头" if shortened == "弯头" else shortened
    if not aliases:
        return normalized
    target = aliases.get(name, aliases.get(shortened, aliases.get(normalized)))
    if isinstance(target, str):
        return normalize_product_name(target)
    if target:
        targets = {normalize_product_name(item) for item in target}
        if len(targets) == 1:
            return next(iter(targets))
    return normalized


def _matching_product_name(value: object) -> str:
    name = normalize_product_name(value)
    for prefix in BALL_IRON_PREFIXES:
        if name.startswith(prefix):
            return normalize_product_name(name.removeprefix(prefix))
    return name


def _ui_category(price: PriceRecord) -> str:
    if price.category == "玛钢" and clean_text(price.product_name).startswith(BALL_IRON_PREFIXES):
        return "球铁"
    return price.category


def normalize_price_marker(value: object, grade_markers: dict[str, str] | None = None) -> str:
    marker = clean_text(value)
    if marker in {"a", "A", "b"}:
        return marker
    if marker == "B":
        return "b"
    # Only an extracted grade field reaches this mapping, never a product name.
    marker = {"黑": "黑袋", "蓝": "蓝袋"}.get(marker, marker)
    return (grade_markers or {}).get(marker, "")


def _inquiry_fields(product: ProductRecord) -> tuple[str, str, str, int, str]:
    if product.extraction_method == "待人工处理":
        raise ValueError("识别失败占位行仅填写五列不能确认，请删除后人工新增或重新识别")
    category = clean_text(product.category)
    name = normalize_product_name(product.product_name)
    specification = clean_text(product.specification)
    if category not in UI_CATEGORIES:
        raise ValueError("分类必须选择沟槽、玛钢或球铁")
    if not name or not specification:
        raise ValueError("产品名和规格不能为空")
    raw_marker = clean_text(product.price_marker)
    marker = normalize_price_marker(raw_marker)
    if raw_marker and not marker:
        raise ValueError("价格标记只能是 a、A、b 或空白")
    if (clean_text(product.original_grade_text) or product.grade_review_required) and not raw_marker:
        raise ValueError("已识别等级/袋色原文但未确认合法 a/A/b")
    return category, name, specification, positive_integer(product.quantity_cartons, "箱数"), marker or "b"


def inquiry_confirmation_is_current(product: ProductRecord) -> bool:
    if not product.manually_confirmed or product.confirmed_inquiry is None:
        return False
    try:
        return product.confirmed_inquiry == _inquiry_fields(product)
    except ValueError:
        return False


def positive_integer(value: object, label: str) -> int:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是正整数") from exc
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise ValueError(f"{label}必须是正整数")
    return int(number)


def positive_money(value: object, label: str = "单价") -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是有限正数") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"{label}必须是有限正数")
    return number


def money_total(quantity: object, packing: object, unit_price: object) -> Decimal:
    cartons = positive_integer(quantity, "箱数")
    package_size = positive_integer(packing, "包装数")
    price = positive_money(unit_price)
    multiplier_digits = len(str(cartons)) + len(str(package_size))
    precision = max(28, len(price.as_tuple().digits) + multiplier_digits, max(0, price.adjusted()) + multiplier_digits + 3)
    try:
        with localcontext() as context:
            context.prec = precision
            result = Decimal(cartons) * Decimal(package_size) * price
            total = result.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    except (InvalidOperation, OverflowError) as exc:
        raise ValueError("行总价计算失败") from exc
    if not total.is_finite() or total <= 0:
        raise ValueError("行总价必须是有限正数")
    return total


def money_sum(values: Iterable[Decimal]) -> Decimal:
    values = list(values)
    if not values:
        return Decimal("0.00")
    precision = max(28, max(max(len(value.as_tuple().digits), value.adjusted() + 3) for value in values) + len(str(len(values))) + 1)
    with localcontext() as context:
        context.prec = precision
        return sum(values, Decimal("0.00"))


def discover_master_data(directory: str | Path = MASTER_DATA_DIR) -> Path:
    directory = Path(directory)
    workbooks = sorted(path for path in directory.glob("*.xlsx") if not path.name.startswith("~$") and path.is_file())
    if len(workbooks) != 1:
        raise ValueError("主数据目录根层必须且只能包含一张非临时 xlsx")
    return workbooks[0]


def master_data_fingerprint(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_price_records(paths: Iterable[str | Path]) -> list[PriceRecord]:
    records: list[PriceRecord] = []
    for path in paths:
        workbook = load_workbook(path, data_only=True, read_only=True)
        for sheet in workbook.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            header_index = next(
                (i for i, row in enumerate(rows[:20]) if "品名" in row and "规格" in row and "单价" in row),
                None,
            )
            if header_index is None:
                continue
            header = {clean_text(value): i for i, value in enumerate(rows[header_index]) if value is not None}
            required = {"品名", "规格", "单价", "包装数"}
            if not required.issubset(header):
                continue
            for source_row, row in enumerate(rows[header_index + 1 :], header_index + 2):
                name, spec = row[header["品名"]], row[header["规格"]]
                price, pack = row[header["单价"]], row[header["包装数"]]
                if not name or not spec or price in (None, ""):
                    continue
                marker = row[header["价格标记"]] if "价格标记" in header else ""
                aliases = row[header["别名"]] if "别名" in header else ""
                records.append(
                    PriceRecord(
                        str(path),
                        sheet.title,
                        source_row,
                        clean_text(name),
                        clean_text(spec),
                        normalize_price_marker(marker),
                        positive_money(price),
                        None if pack in (None, "") else positive_integer(pack, "包装数"),
                        tuple(filter(None, re.split(r"[；;]", clean_text(aliases)))),
                    )
                )
        workbook.close()
    return records


MASTER_DATA_HEADERS = {
    "分类", "源代码", "商标", "品名", "规格", "别名", "价格标记", "轻重", "单价", "包装数",
    "毛坯单重", "成品单重", "纯毛单重", "启用状态", "记录键", "备注",
}
MASTER_DATA_STATUSES = {"启用", "待补充", "停用"}
GRADE_MAPPING_HEADERS = {"客户写法", "价格标记", "启用状态", "备注"}
MULTI_ALIAS_MAPPING_HEADERS = {"分类", "别名", "正式品名", "启用状态", "备注"}
MASTER_PRODUCT_SHEETS = {"沟槽数据": "沟槽", "玛钢数据": "玛钢"}


def _optional_positive_decimal(value: object, label: str) -> Decimal | None:
    if value in (None, ""):
        return None
    return positive_money(value, label)


def load_master_price_records(path: str | Path, workbook_bytes: bytes | None = None) -> list[PriceRecord]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"主数据文件不存在：{path}")
    workbooks = sorted(candidate.resolve() for candidate in path.parent.glob("*.xlsx") if not candidate.name.startswith("~$"))
    if workbooks != [path.resolve()]:
        raise ValueError("主数据目录根层必须且只能包含一张非临时 xlsx")
    workbook = load_workbook(BytesIO(workbook_bytes) if workbook_bytes is not None else path, data_only=True, read_only=True)
    try:
        if workbook.sheetnames != ["沟槽数据", "玛钢数据", "等级映射"]:
            raise ValueError("主数据文件必须且只能按顺序包含“沟槽数据”“玛钢数据”“等级映射”")
        _grade_markers_from_workbook(workbook)
        multi_target_aliases = _multi_alias_targets_from_workbook(workbook)
        records: list[PriceRecord] = []
        business_keys: set[tuple[str, str, str, str]] = set()
        record_keys: set[str] = set()
        for sheet_name, expected_category in MASTER_PRODUCT_SHEETS.items():
            sheet = workbook[sheet_name]
            rows = list(sheet.iter_rows(values_only=True))
            header_index = next(
                (i for i, row in enumerate(rows[:20]) if MASTER_DATA_HEADERS.issubset({clean_text(value) for value in row if value is not None})),
                None,
            )
            if header_index is None:
                raise ValueError(f"{sheet_name}缺少标准表头")
            header = {clean_text(value): i for i, value in enumerate(rows[header_index]) if value is not None}
            for source_row, row in enumerate(rows[header_index + 1 :], header_index + 2):
                if not any(value not in (None, "") for value in row):
                    continue
                category = clean_text(row[header["分类"]])
                source_code = clean_text(row[header["源代码"]])
                trademark = clean_text(row[header["商标"]])
                status = clean_text(row[header["启用状态"]])
                name = clean_text(row[header["品名"]])
                spec = clean_text(row[header["规格"]])
                marker = clean_text(row[header["价格标记"]])
                price = row[header["单价"]]
                pack = row[header["包装数"]]
                light_heavy = clean_text(row[header["轻重"]])
                record_key = clean_text(row[header["记录键"]])
                notes = clean_text(row[header["备注"]])
                if category != expected_category:
                    raise ValueError(f"{sheet_name}第 {source_row} 行分类必须是{expected_category}")
                if not source_code or not trademark:
                    raise ValueError(f"{sheet_name}第 {source_row} 行缺少源代码或商标")
                if status not in MASTER_DATA_STATUSES:
                    raise ValueError(f"{sheet_name}第 {source_row} 行启用状态非法")
                if not name or not spec:
                    raise ValueError(f"{sheet_name}第 {source_row} 行缺少品名或规格")
                if marker not in {"a", "A", "b"}:
                    raise ValueError(f"{sheet_name}第 {source_row} 行价格标记只能是 a、A 或 b")
                expected_light_heavy = {"a": "轻货", "A": "", "b": "重货"}[marker]
                if light_heavy != expected_light_heavy:
                    raise ValueError(f"{sheet_name}第 {source_row} 行轻重必须与价格标记一致")
                expected_key = f"{category}｜{name}｜{normalize_specification(spec)}｜{marker}"
                if record_key != expected_key:
                    raise ValueError(f"{sheet_name}第 {source_row} 行记录键必须是：{expected_key}")
                business_key = (category, clean_text(name).lower(), normalize_specification(spec), marker)
                if business_key in business_keys:
                    raise ValueError(f"{sheet_name}第 {source_row} 行业务键重复：{expected_key}")
                business_keys.add(business_key)
                if record_key in record_keys:
                    raise ValueError(f"{sheet_name}第 {source_row} 行记录键重复：{record_key}")
                record_keys.add(record_key)
                try:
                    price_decimal = None if price in (None, "") else positive_money(price)
                    pack_integer = None if pack in (None, "") else positive_integer(pack, "包装数")
                    blank_weight = _optional_positive_decimal(row[header["毛坯单重"]], "毛坯单重")
                    finished_weight = _optional_positive_decimal(row[header["成品单重"]], "成品单重")
                    pure_weight = _optional_positive_decimal(row[header["纯毛单重"]], "纯毛单重")
                except ValueError as exc:
                    raise ValueError(f"{sheet_name}第 {source_row} 行：{exc}") from exc
                if status == "启用" and price_decimal is None:
                    raise ValueError(f"{sheet_name}第 {source_row} 行启用记录缺少单价")
                if status == "待补充" and (price_decimal is not None or "禁止报价" not in notes):
                    raise ValueError(f"{sheet_name}第 {source_row} 行待补充记录必须缺单价并备注禁止报价")
                missing_notes = [
                    (pack_integer, "缺包装数"),
                    (blank_weight, "缺毛坯单重"),
                    (finished_weight, "缺成品单重"),
                ]
                if any(value is None and phrase not in notes for value, phrase in missing_notes):
                    raise ValueError(f"{sheet_name}第 {source_row} 行缺失字段必须在备注中逐项说明")
                if pure_weight is None and not (
                    "源表未提供纯毛单重" in notes if category == "玛钢" else "缺纯毛单重" in notes
                ):
                    raise ValueError(f"{sheet_name}第 {source_row} 行缺纯毛单重必须在备注中说明")
                if status == "停用":
                    continue
                aliases = tuple(filter(None, re.split(r"[；;]", clean_text(row[header["别名"]]))))
                records.append(
                    PriceRecord(
                        source_file=str(path), sheet_name=sheet.title, source_row=source_row,
                        product_name=name, specification=spec, price_marker=marker,
                        unit_price=price_decimal, packing_quantity=pack_integer, aliases=aliases,
                        category=category, blank_weight=blank_weight, finished_weight=finished_weight,
                        pure_weight=pure_weight, notes=notes, multi_target_aliases=multi_target_aliases, status=status,
                    )
                )
        if not records:
            raise ValueError("主数据没有可检索的启用或待补充记录")
        return records
    finally:
        workbook.close()


def _grade_markers_from_workbook(workbook) -> dict[str, str]:
    if "等级映射" not in workbook.sheetnames:
        raise ValueError("主数据缺少工作表“等级映射”")
    rows = list(workbook["等级映射"].iter_rows(values_only=True))
    header_index = next(
        (i for i, row in enumerate(rows[:20]) if GRADE_MAPPING_HEADERS.issubset({clean_text(value) for value in row if value is not None})),
        None,
    )
    if header_index is None:
        raise ValueError("等级映射缺少标准表头")
    header = {clean_text(value): i for i, value in enumerate(rows[header_index]) if value is not None}
    mappings: dict[str, str] = {}
    for source_row, row in enumerate(rows[header_index + 1 :], header_index + 2):
        if not any(header[field] < len(row) and row[header[field]] not in (None, "") for field in GRADE_MAPPING_HEADERS):
            continue
        status = clean_text(row[header["启用状态"]])
        if status not in {"启用", "停用"}:
            raise ValueError(f"等级映射第 {source_row} 行启用状态非法")
        if status == "停用":
            continue
        phrase = clean_text(row[header["客户写法"]])
        marker = clean_text(row[header["价格标记"]])
        if not phrase or marker not in {"a", "b"}:
            raise ValueError(f"等级映射第 {source_row} 行客户写法或价格标记非法")
        if phrase in mappings:
            raise ValueError(f"等级映射第 {source_row} 行客户写法重复：{phrase}")
        mappings[phrase] = marker
    return mappings


def _multi_alias_targets_from_workbook(workbook) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    rows = list(workbook["等级映射"].iter_rows(values_only=True))
    header_index = next(
        (i for i, row in enumerate(rows[:20]) if MULTI_ALIAS_MAPPING_HEADERS.issubset({clean_text(value) for value in row if value is not None})),
        None,
    )
    if header_index is None:
        raise ValueError("等级映射缺少多目标别名表头")
    header = {clean_text(value): i for i, value in enumerate(rows[header_index]) if value is not None}
    mappings: dict[tuple[str, str], list[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    for source_row, row in enumerate(rows[header_index + 1 :], header_index + 2):
        category = clean_text(row[header["分类"]])
        alias = clean_text(row[header["别名"]])
        target = clean_text(row[header["正式品名"]])
        status = clean_text(row[header["启用状态"]])
        if not any((category, alias, target, status)):
            continue
        if category not in {"沟槽", "玛钢"}:
            raise ValueError(f"等级映射第 {source_row} 行多目标别名分类非法")
        if not alias or not target or status not in {"启用", "停用"}:
            raise ValueError(f"等级映射第 {source_row} 行多目标别名字段非法")
        if status == "停用":
            continue
        key = (category, alias.lower(), target.lower())
        if key in seen:
            raise ValueError(f"等级映射第 {source_row} 行多目标别名重复：{category}｜{alias}｜{target}")
        seen.add(key)
        mappings.setdefault((category, alias), []).append(target)
    if not mappings or any(len(set(targets)) < 2 for targets in mappings.values()):
        raise ValueError("多目标别名必须为同一分类和别名保留至少两个正式品名")
    return tuple((category, alias, tuple(targets)) for (category, alias), targets in mappings.items())


def load_master_grade_markers(path: str | Path, workbook_bytes: bytes | None = None) -> dict[str, str]:
    workbook = load_workbook(BytesIO(workbook_bytes) if workbook_bytes is not None else path, data_only=True, read_only=True)
    try:
        return _grade_markers_from_workbook(workbook)
    finally:
        workbook.close()


def load_master_data(path: str | Path, expected_fingerprint: str | None = None) -> tuple[list[PriceRecord], dict[str, str], str]:
    path = Path(path)
    data = path.read_bytes()
    fingerprint = sha256(data).hexdigest()
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("主数据内容在加载前已变化，请重新读取")
    return load_master_price_records(path, data), load_master_grade_markers(path, data), fingerprint


def price_aliases(prices: Iterable[PriceRecord], category: str = "") -> dict[str, frozenset[str]]:
    prices = list(prices)
    targets: dict[str, set[str]] = {}
    for price in prices:
        if category and _ui_category(price) != category:
            continue
        canonical = _matching_product_name(price.product_name)
        for alias in price.aliases:
            targets.setdefault(clean_text(alias).lower(), set()).add(canonical)
    explicit_targets: dict[str, set[str]] = {}
    for price in prices:
        for mapping_category, alias, names in price.multi_target_aliases:
            if mapping_category != price.category or (category and _ui_category(price) != category):
                continue
            explicit_targets.setdefault(clean_text(alias).lower(), set()).update(_matching_product_name(name) for name in names)
    targets.update(explicit_targets)
    return {alias: frozenset(names) for alias, names in targets.items()}


def price_product_terms(prices: Iterable[PriceRecord]) -> set[str]:
    prices = list(prices)
    multi_terms = {
        term
        for price in prices
        for _category, alias, targets in price.multi_target_aliases
        for term in (alias, *targets)
    }
    return {price.product_name for price in prices} | {alias for price in prices for alias in price.aliases} | multi_terms


def _name_candidates(name: str, specification: str = "", aliases: dict[str, object] | None = None) -> set[str]:
    raw = clean_text(name).lower()
    shortened = raw.removeprefix("demo")
    normalized = normalize_product_name(shortened)
    candidates = {normalized}
    if aliases:
        target = aliases.get(raw, aliases.get(shortened, aliases.get(normalized)))
        if isinstance(target, str):
            candidates.add(normalize_product_name(target))
        elif target:
            candidates.update(normalize_product_name(item) for item in target)
    return candidates


def _legal_candidates(product: ProductRecord, prices: Iterable[PriceRecord]) -> list[PriceRecord]:
    prices = list(prices)
    category = clean_text(product.category)
    input_name = clean_text(product.product_name)
    specification = normalize_specification(product.specification)
    categories = {_ui_category(price) for price in prices}
    aliases_by_category = {candidate_category: price_aliases(prices, candidate_category) for candidate_category in categories}
    formal_targets_by_category = {candidate_category: set() for candidate_category in categories}
    mapped_aliases_by_category = {candidate_category: set() for candidate_category in categories}
    for price in prices:
        price_category = _ui_category(price)
        for mapping_category, alias, targets in price.multi_target_aliases:
            if mapping_category != price.category:
                continue
            mapped_aliases_by_category[price_category].add(_matching_product_name(alias))
            formal_targets_by_category[price_category].update(_matching_product_name(target) for target in targets)
    candidates: list[PriceRecord] = []
    seen: set[tuple[str, str, int, str, str]] = set()
    for price in prices:
        price_category = _ui_category(price)
        if (category and price_category != category) or normalize_specification(price.specification) != specification:
            continue
        names = {
            _matching_product_name(name)
            for name in _name_candidates(input_name, aliases=aliases_by_category[price_category])
        }
        formal_canonical = normalize_product_name(price.product_name)
        canonical = _matching_product_name(price.product_name)
        source_terms = {canonical, *(_matching_product_name(alias) for alias in price.aliases)}
        expansions: set[str] = set()
        for mapping_category, alias, targets in price.multi_target_aliases:
            if mapping_category == price.category and _matching_product_name(alias) in source_terms:
                expansions.update(_matching_product_name(target) for target in targets)
        direct_targets = (
            set()
            if formal_canonical in mapped_aliases_by_category[price_category]
            and canonical not in formal_targets_by_category[price_category]
            else {canonical}
        )
        for target in (direct_targets | expansions) & names:
            key = (price.source_file, price.sheet_name, price.source_row, price.price_marker, target)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(price if target == canonical else replace(price, product_name=target))
    return candidates


def required_categories(product: ProductRecord, prices: Iterable[PriceRecord]) -> set[str]:
    prices = list(prices)
    explicit_category = (
        "球铁" if clean_text(product.product_name).lower().startswith(BALL_IRON_PREFIXES) else None
    )
    candidate_sources = {
        (candidate.source_file, candidate.sheet_name, candidate.source_row)
        for candidate in _legal_candidates(replace(product, category=explicit_category), prices)
    }
    return {
        _ui_category(price)
        for price in prices
        if (price.source_file, price.sheet_name, price.source_row) in candidate_sources and _ui_category(price)
    }


def _clear_price_match(product: ProductRecord) -> None:
    product.unit_price = None
    product.master_packing_quantity = None
    product.packing_quantity = None
    product.total_price = None
    product.price_source_file = None
    product.price_source_sheet = None
    product.price_source_row = None
    product.master_fingerprint = None


def confirm_inquiry(product: ProductRecord) -> None:
    category, name, specification, quantity, marker = _inquiry_fields(product)
    product.category = category
    product.product_name = name
    product.specification = specification
    product.quantity_cartons = quantity
    product.price_marker = marker or None
    product.confirmed_inquiry = (category, name, specification, quantity, marker)
    product.manually_confirmed = True
    product.match_status = "pending"
    product.match_message = ""
    _clear_price_match(product)


def match_price(product: ProductRecord, prices: Iterable[PriceRecord], master_fingerprint: str) -> list[PriceRecord]:
    _clear_price_match(product)
    if not inquiry_confirmation_is_current(product):
        product.manually_confirmed = False
        product.match_status = "unconfirmed"
        product.match_message = "第一次人工确认不存在或询价字段已修改，禁止匹配价格"
        return []
    try:
        product.quantity_cartons = positive_integer(product.quantity_cartons, "箱数")
    except ValueError as exc:
        product.match_status = "invalid_quantity"
        product.match_message = str(exc)
        return []
    candidates = _legal_candidates(product, prices)
    raw_marker = clean_text(product.price_marker)
    marker = normalize_price_marker(product.price_marker)
    if raw_marker and not marker:
        product.match_status = "invalid_marker"
        product.match_message = "价格标记只能是 a、A、b 或空白"
        return candidates
    if marker:
        candidates = [price for price in candidates if price.price_marker == marker]
    elif any(price.price_marker for price in candidates):
        product.match_status = "grade_required"
        product.match_message = "合法候选带 a/A/b，必须在第一次确认中人工选择"
        return candidates
    else:
        candidates = [price for price in candidates if not price.price_marker]
    if len(candidates) == 1:
        apply_price(product, candidates[0], master_fingerprint)
    elif candidates and len({
        (
            price.source_file, price.sheet_name, price.source_row, clean_text(price.category),
            normalize_specification(price.specification), price.price_marker, price.unit_price,
            price.packing_quantity, price.status, price.blank_weight, price.finished_weight,
            price.pure_weight, price.notes,
        )
        for price in candidates
    }) == 1:
        candidates = [replace(candidates[0], product_name=clean_text(product.product_name))]
        apply_price(product, candidates[0], master_fingerprint)
    elif candidates:
        product.match_status = "multiple_candidates"
        product.match_message = "存在多个候选，必须人工选择"
    else:
        product.match_status = "not_found"
        product.match_message = "主数据中未找到匹配价格，必须补充主数据后重试"
    return candidates


def apply_price(product: ProductRecord, price: PriceRecord, master_fingerprint: str) -> None:
    product.price_marker = price.price_marker or product.price_marker
    product.unit_price = price.unit_price
    product.master_packing_quantity = price.packing_quantity
    product.packing_quantity = price.packing_quantity
    product.price_source_file = price.source_file
    product.price_source_sheet = price.sheet_name
    product.price_source_row = price.source_row
    product.master_fingerprint = master_fingerprint
    if price.unit_price is None:
        product.total_price = None
        product.match_status = "pending"
        product.match_message = "面价待补充；该行不计算金额且不写入报价"
        return
    product.total_price = None if price.packing_quantity is None else money_total(product.quantity_cartons, price.packing_quantity, price.unit_price)
    product.match_status = "missing_packing" if price.packing_quantity is None else "matched"
    product.match_message = "主数据无包装数；按缺包装规则留空生成" if price.packing_quantity is None else "已匹配"


def apply_packing_override(product: ProductRecord, packing: object) -> None:
    if product.unit_price is None:
        raise ValueError("面价待补充，不能计算金额")
    if product.master_packing_quantity is None:
        if packing not in (None, ""):
            raise ValueError("主数据无包装数，不得填写本单包装数")
        product.packing_quantity = None
        product.total_price = None
        return
    packing = positive_integer(packing, "本单包装数")
    if packing != product.master_packing_quantity:
        raise ValueError("本单包装数必须与主数据包装数一致")
    product.packing_quantity = product.master_packing_quantity
    product.total_price = money_total(product.quantity_cartons, product.packing_quantity, product.unit_price)


def normalize_customer_name(value: object) -> str:
    name = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    if not name:
        raise ValueError("单位名称不能为空")
    return name


def load_customer_names(path: str | Path | None = None) -> list[str]:
    path = Path(path or CUSTOMER_HISTORY_PATH)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("单位名称历史无法读取") from exc
    if not isinstance(payload, list):
        raise ValueError("单位名称历史格式非法")
    names: list[str] = []
    for value in payload:
        if not isinstance(value, str):
            raise ValueError("单位名称历史中的每一项都必须是字符串")
        try:
            name = normalize_customer_name(value)
        except ValueError as exc:
            raise ValueError("单位名称历史包含空白项") from exc
        if name not in names:
            names.append(name)
    return names


def remember_customer_name(value: object, path: str | Path | None = None) -> str:
    name = normalize_customer_name(value)
    path = Path(path or CUSTOMER_HISTORY_PATH)
    # shortcut: history locking is per process; use a file lock or DB for multiple workers.
    with CUSTOMER_HISTORY_LOCK:
        names = load_customer_names(path)
        if name in names:
            return name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as target:
                json.dump([*names, name], target, ensure_ascii=False, indent=2)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return name


def local_generation_date() -> date:
    return datetime.now(BUSINESS_TIMEZONE).date()


def generate_sales_id() -> str:
    code = "".join(secrets.choice(SALES_ID_ALPHABET) for _ in range(10))
    return f"QA-{code[:5]}-{code[5:]}"


def _format_decimal(value: Decimal | None, fixed: bool = False) -> str:
    if value is None:
        return ""
    return f"{value:.2f}" if fixed else format(value.normalize(), "f")


def _format_unit_price(value: Decimal | None) -> str:
    if value is None:
        return ""
    whole, _, fraction = format(value, "f").partition(".")
    fraction = fraction.rstrip("0").ljust(2, "0")
    return f"{whole}.{fraction}"


def _sales_specification(product: ProductRecord) -> str:
    specification = clean_text(product.specification)
    marker = normalize_price_marker(product.price_marker)
    return specification if marker not in {"a", "A"} or specification.endswith(marker) else f"{specification}{marker}"


def _chinese_money(value: Decimal) -> str:
    digits = "零壹贰叁肆伍陆柒捌玖"

    def group_text(number: int) -> str:
        result = ""
        pending_zero = False
        for divisor, unit in ((1000, "仟"), (100, "佰"), (10, "拾"), (1, "")):
            digit = number // divisor % 10
            if digit:
                if pending_zero and result:
                    result += "零"
                result += digits[digit] + unit
                pending_zero = False
            elif result and number % divisor:
                pending_zero = True
        return result

    amount = value.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    yuan = int(amount)
    groups = []
    while yuan:
        groups.append(yuan % 10000)
        yuan //= 10000
    if len(groups) > 4:
        raise ValueError("金额过大，无法安全排版")
    integer = ""
    zero_group = False
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if group:
            if integer and (zero_group or group < 1000):
                integer += "零"
            integer += group_text(group) + ("", "万", "亿", "兆")[index]
            zero_group = False
        elif integer:
            zero_group = True
    integer = integer or "零"
    jiao = int(amount * 10) % 10
    fen = int(amount * 100) % 10
    fraction = "整" if not jiao and not fen else (digits[jiao] + "角" if jiao else "零") + (digits[fen] + "分" if fen else "")
    return integer + "元" + fraction


def _draw_text(page, rect: fitz.Rect, value: object, size: float = 9, align: int = fitz.TEXT_ALIGN_CENTER, bold: bool = False) -> None:
    text = "" if value is None else str(value)
    if not text:
        return
    font_path = PDF_BOLD_FONT_PATH if bold else PDF_FONT_PATH
    font_name = "msyh-bold" if bold else "msyh"
    if font_path.is_file():
        page.insert_font(fontname=font_name, fontfile=str(font_path))
    else:
        # PyMuPDF's bundled CJK font keeps Linux/macOS demos self-contained.
        font_name = "demo-cjk"
        page.insert_font(fontname=font_name, fontbuffer=fitz.Font("cjk").buffer)
    for font_size in (size, size - 1, size - 2, 6):
        shape = page.new_shape()
        remaining = shape.insert_textbox(
            rect,
            text,
            fontname=font_name,
            fontsize=max(font_size, 6),
            align=align,
            lineheight=1.05,
        )
        if remaining >= -0.01:
            shape.commit()
            return
    raise ValueError(f"销售单字段过长，无法安全排版：{text}")


def _fill_rect(page, rect: fitz.Rect, color: tuple[float, float, float]) -> None:
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(color=None, fill=color)
    shape.commit()


def _draw_grid(page, xs: list[float], ys: list[float]) -> None:
    shape = page.new_shape()
    for x in xs[1:-1]:
        shape.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys[1:-1]:
        shape.draw_line((xs[0], y), (xs[-1], y))
    shape.finish(color=(0.655, 0.655, 0.655), width=0.35)
    shape.commit()
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(xs[0], ys[0], xs[-1], ys[-1]))
    shape.finish(color=(0.239, 0.239, 0.239), width=0.9)
    shape.commit()


def _render_sales_page(
    page,
    products: list[ProductRecord],
    customer_name: str,
    generated_on: date,
    sales_id: str,
    page_number: int,
    page_count: int,
    incomplete_total: bool,
    is_last_page: bool,
    all_products: list[ProductRecord],
) -> None:
    left, right = 28.0, 694.0
    widths = [45, 100, 88, 38, 40, 44, 38, 58, 56, 72, 87]
    xs = [left]
    for width in widths:
        xs.append(xs[-1] + width)
    _draw_text(page, fitz.Rect(xs[9], 28, right, 47), f"共{page_count}页    第{page_number}页", 9, bold=True)
    _draw_text(page, fitz.Rect(75, 44, 647, 76), "示例企业销售单 - 虚构数据", 17, bold=True)
    customer_start = xs[1] + 12
    _draw_text(page, fitz.Rect(left, 82, customer_start, 106), "单位名称：", 10, fitz.TEXT_ALIGN_RIGHT, True)
    _draw_text(page, fitz.Rect(customer_start, 82, xs[4], 106), customer_name, 10, fitz.TEXT_ALIGN_LEFT)
    _draw_text(
        page,
        fitz.Rect(xs[5], 82, xs[7], 106),
        f"日期：{generated_on.year}/{generated_on.month}/{generated_on.day}",
        10,
        fitz.TEXT_ALIGN_LEFT,
    )
    _draw_text(page, fitz.Rect(xs[8], 82, right, 106), f"销单号：{sales_id}", 10, fitz.TEXT_ALIGN_RIGHT)

    row_height = 21
    table_top = 110
    ys = [table_top + row_height * index for index in range(len(products) + 2)]
    _fill_rect(page, fitz.Rect(xs[0], ys[0], xs[-1], ys[1]), (0.91, 0.93, 0.94))
    _draw_grid(page, xs, ys)
    headers = ["商标", "品名", "规格", "单位", "箱数", "包装", "零个", "总个数", "单价", "金额", "备注"]
    for index, header in enumerate(headers):
        _draw_text(page, fitz.Rect(xs[index] + 4, ys[0] + 4, xs[index + 1] - 4, ys[1] - 1), header, 10, bold=True)
    for row_index, product in enumerate(products, 1):
        pieces = None if product.packing_quantity is None else product.quantity_cartons * product.packing_quantity
        values = [
            "", product.product_name, _sales_specification(product), "箱", product.quantity_cartons,
            product.packing_quantity, "", pieces, _format_unit_price(product.unit_price),
            _format_decimal(product.total_price, True), "",
        ]
        for column, value in enumerate(values):
            _draw_text(
                page,
                fitz.Rect(xs[column] + 4, ys[row_index] + 4, xs[column + 1] - 4, ys[row_index + 1] - 1),
                value,
                10,
            )

    if not is_last_page:
        return

    summary_top = ys[-1]
    summary_mid = summary_top + 23
    summary_bottom = summary_mid + 26
    _fill_rect(page, fitz.Rect(xs[0], summary_top, xs[-1], summary_mid), (0.91, 0.93, 0.94))
    _draw_grid(page, xs, [summary_top, summary_mid])
    summary_values = [
        "合计", "箱数", sum(product.quantity_cartons for product in all_products), "个数",
        "" if incomplete_total else sum(product.quantity_cartons * product.packing_quantity for product in all_products),
        "货款", _format_decimal(money_sum(product.total_price for product in all_products if product.total_price is not None), True),
        "下浮", "", "已算小计" if incomplete_total else "合计金额",
        _format_decimal(money_sum(product.total_price for product in all_products if product.total_price is not None), True),
    ]
    for column, value in enumerate(summary_values):
        _draw_text(page, fitz.Rect(xs[column] + 4, summary_top + 4, xs[column + 1] - 4, summary_mid - 1), value, 10, bold=True)

    shape = page.new_shape()
    for x in (xs[2], xs[8], xs[10]):
        shape.draw_line((x, summary_mid), (x, summary_bottom))
    shape.finish(color=(0.655, 0.655, 0.655), width=0.35)
    shape.commit()
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(left, summary_mid, right, summary_bottom))
    shape.finish(color=(0.239, 0.239, 0.239), width=0.9)
    shape.commit()
    page_total = money_sum(product.total_price for product in all_products if product.total_price is not None)
    uppercase_label = "已计算小计大写" if incomplete_total else "人民币大写"
    _draw_text(page, fitz.Rect(left + 4, summary_mid + 5, xs[2] - 4, summary_bottom - 1), uppercase_label, 10, bold=True)
    _draw_text(page, fitz.Rect(xs[2] + 5, summary_mid + 5, xs[8] - 5, summary_bottom - 1), _chinese_money(page_total), 10, bold=True)
    _draw_text(page, fitz.Rect(xs[8] + 4, summary_mid + 5, xs[10] - 4, summary_bottom - 1), "重量：", 10, bold=True)

    footer_top = summary_bottom + 12
    footer = ["负责人：", "提货人签字：", "保管员：", "开票员：", "车号：", "手机号："]
    footer_widths = [100, 130, 100, 100, 100, 136]
    x = left
    for label, width in zip(footer, footer_widths):
        _draw_text(page, fitz.Rect(x, footer_top, x + width, footer_top + 22), label, 10, fitz.TEXT_ALIGN_LEFT, True)
        x += width


def _render_quote_files(
    products: list[ProductRecord], customer_name: str, generated_on: date, sales_id: str, incomplete_total: bool
) -> tuple[bytes, list[bytes]]:
    pages = [products[index : index + QUOTE_ROWS_PER_PAGE] for index in range(0, len(products), QUOTE_ROWS_PER_PAGE)]
    document = fitz.open()
    try:
        for page_number, page_products in enumerate(pages, 1):
            page = document.new_page(width=722, height=394)
            _render_sales_page(
                page,
                page_products,
                customer_name,
                generated_on,
                sales_id,
                page_number,
                len(pages),
                incomplete_total,
                page_number == len(pages),
                products,
            )
        document.subset_fonts()
        pdf_bytes = document.tobytes(garbage=4, deflate=True)
    finally:
        document.close()
    reopened = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if reopened.page_count != len(pages):
            raise ValueError("PDF 页数回读不一致")
        png_bytes = [page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png") for page in reopened]
    finally:
        reopened.close()
    for content in png_bytes:
        with Image.open(BytesIO(content)) as image:
            image.verify()
    return pdf_bytes, png_bytes


def _prepare_render_products(products: Iterable[ProductRecord]) -> tuple[list[ProductRecord], list[Decimal], int]:
    quotable_products: list[ProductRecord] = []
    calculated_totals: list[Decimal] = []
    missing_packing = 0
    for product in products:
        if not inquiry_confirmation_is_current(product):
            product.manually_confirmed = False
            raise ValueError("存在未确认或确认后已修改的询价字段")
        if not product.price_source_file or not product.master_fingerprint:
            raise ValueError("存在没有主数据价格来源或内容指纹的行")
        if product.match_status == "pending" and product.unit_price is None:
            continue
        product.unit_price = positive_money(product.unit_price)
        apply_packing_override(product, product.packing_quantity)
        quotable_products.append(product)
        if product.total_price is None:
            missing_packing += 1
        else:
            calculated_totals.append(product.total_price)
    if not quotable_products:
        raise ValueError("没有可生成的有价产品")
    return quotable_products, calculated_totals, missing_packing


def _verify_products_against_current_master(
    products: Iterable[ProductRecord], master_data_dir: str | Path
) -> str:
    master_path = discover_master_data(master_data_dir)
    current_prices, _, current_fingerprint = load_master_data(master_path)
    for product in products:
        if not inquiry_confirmation_is_current(product):
            product.manually_confirmed = False
            raise ValueError("存在未确认或确认后已修改的询价字段")
        if not product.price_source_file or not product.master_fingerprint:
            raise ValueError("存在没有主数据价格来源或内容指纹的行")
        if product.master_fingerprint != current_fingerprint:
            raise ValueError("主数据内容已变化，旧价格已失效，必须重新匹配")
        product.quantity_cartons = positive_integer(product.quantity_cartons, "箱数")
        _verify_current_master_match(product, current_prices, current_fingerprint)
    return current_fingerprint


def render_quotation_preview(
    products: Iterable[ProductRecord],
    customer_name: object,
    master_data_dir: str | Path = MASTER_DATA_DIR,
    *,
    generated_on: date | None = None,
) -> list[bytes]:
    products = list(products)
    if not products:
        raise ValueError("没有可生成的产品")
    customer_name = normalize_customer_name(customer_name)
    _verify_products_against_current_master(products, master_data_dir)
    products, _, missing_packing = _prepare_render_products(products)
    _, png_bytes = _render_quote_files(
        products,
        customer_name,
        generated_on or local_generation_date(),
        "PREVIEW",
        bool(missing_packing),
    )
    return png_bytes


def _verify_current_master_match(product: ProductRecord, prices: list[PriceRecord], fingerprint: str) -> None:
    resolved = ProductRecord(
        source_type="generation_check",
        source_name=product.source_name,
        category=product.category,
        product_name=product.product_name,
        specification=product.specification,
        quantity_cartons=product.quantity_cartons,
        price_marker=product.price_marker,
    )
    confirm_inquiry(resolved)
    match_price(resolved, prices, fingerprint)
    if product.match_status == "pending" and product.unit_price is None:
        if resolved.match_status != "pending" or resolved.unit_price is not None:
            raise ValueError("当前主数据不再是面价待补充状态，必须重新匹配")
        current = (
            product.master_packing_quantity,
            product.packing_quantity,
            product.price_source_file,
            product.price_source_sheet,
            product.price_source_row,
        )
        expected = (
            resolved.master_packing_quantity,
            resolved.packing_quantity,
            resolved.price_source_file,
            resolved.price_source_sheet,
            resolved.price_source_row,
        )
        if current != expected:
            raise ValueError("待补充行的包装或价格来源与当前主数据不一致，必须重新匹配")
        return
    if resolved.match_status not in {"matched", "missing_packing"}:
        raise ValueError("当前主数据无法唯一重解该产品、规格和 a/A/b，必须重新匹配")
    current = (
        positive_money(product.unit_price),
        product.master_packing_quantity,
        product.packing_quantity,
        product.price_source_file,
        product.price_source_sheet,
        product.price_source_row,
    )
    expected = (
        resolved.unit_price,
        resolved.master_packing_quantity,
        resolved.packing_quantity,
        resolved.price_source_file,
        resolved.price_source_sheet,
        resolved.price_source_row,
    )
    if current != expected:
        raise ValueError("报价行的单价、包装或价格来源与当前主数据不一致，必须重新匹配")


def _quotation_record_path(customer_history_path: str | Path, created_at: datetime, sales_id: str) -> Path:
    return Path(customer_history_path).parent / "quotation_records" / f"{created_at:%Y}" / f"{created_at:%m}" / f"{sales_id}.json"


def _quotation_record_bytes(
    sales_id: str,
    generated_on: date,
    created_at: datetime,
    customer_name: str,
    output_paths: list[Path],
) -> bytes:
    record = {
        "sales_id": sales_id,
        "document_date": generated_on.isoformat(),
        "created_at": created_at.isoformat(timespec="seconds"),
        "customer_name": customer_name,
        "output_files": [path.name for path in output_paths],
        "category_keys": [],
    }
    return (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def generate_quotation(
    products: Iterable[ProductRecord],
    output_dir: str | Path,
    customer_name: object,
    master_data_dir: str | Path = MASTER_DATA_DIR,
    customer_history_path: str | Path | None = None,
    *,
    sales_id: str | None = None,
    generated_on: date | None = None,
) -> tuple[Path, dict[str, object]]:
    products = list(products)
    if not products:
        raise ValueError("没有可生成的产品")
    customer_name = normalize_customer_name(customer_name)
    created_at = datetime.now(BUSINESS_TIMEZONE)
    generated_on = generated_on or created_at.date()
    sales_id = sales_id or generate_sales_id()
    if SALES_ID_PATTERN.fullmatch(sales_id) is None:
        raise ValueError("销单号必须是 QA-XXXXX-XXXXX 格式的 10 位去歧义大写 Base32 随机码")
    output_dir = Path(output_dir).resolve()
    output_root = OUTPUT_ROOT.resolve()
    try:
        output_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("输出只能写入 data/output/") from exc
    current_fingerprint = _verify_products_against_current_master(products, master_data_dir)
    products, calculated_totals, missing_packing = _prepare_render_products(products)
    subtotal = money_sum(calculated_totals)
    page_count = (len(products) + QUOTE_ROWS_PER_PAGE - 1) // QUOTE_ROWS_PER_PAGE
    pdf_path = output_dir / f"{sales_id}.pdf"
    png_paths = [output_dir / f"{sales_id}_p{page}.png" for page in range(1, page_count + 1)]
    output_paths = [pdf_path, *png_paths]
    history_path = Path(customer_history_path or CUSTOMER_HISTORY_PATH)
    record_path = _quotation_record_path(history_path, created_at, sales_id)
    record_bytes = _quotation_record_bytes(sales_id, generated_on, created_at, customer_name, output_paths)
    expected_record = json.loads(record_bytes)
    existing = next((path for path in [*output_paths, record_path] if path.exists()), None)
    if existing:
        raise FileExistsError(f"输出文件已存在，拒绝覆盖：{existing.name}")
    pdf_bytes, png_bytes = _render_quote_files(products, customer_name, generated_on, sales_id, bool(missing_packing))
    if master_data_fingerprint(discover_master_data(master_data_dir)) != current_fingerprint:
        raise ValueError("主数据在生成过程中发生变化，已阻断输出")

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for path, content in zip(output_paths, [pdf_bytes, *png_bytes]):
            with path.open("xb") as target:
                created.append(path)
                target.write(content)
        reopened = fitz.open(pdf_path)
        try:
            if reopened.page_count != page_count:
                raise ValueError("PDF 输出回读页数不一致")
        finally:
            reopened.close()
        for path in png_paths:
            with Image.open(path) as image:
                image.verify()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("xb") as target:
            created.append(record_path)
            target.write(record_bytes)
        if json.loads(record_path.read_text(encoding="utf-8")) != expected_record:
            raise ValueError("报价后台记录回读不一致")
        if master_data_fingerprint(discover_master_data(master_data_dir)) != current_fingerprint:
            raise ValueError("主数据在生成过程中发生变化，已阻断输出")
        remember_customer_name(customer_name, history_path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return pdf_path, {
        "sales_id": sales_id,
        "generated_date": generated_on,
        "png_paths": png_paths,
        "page_count": page_count,
        "total_label": "已计算小计" if missing_packing else "完整总计",
        "total": subtotal,
        "missing_packing": missing_packing,
    }
