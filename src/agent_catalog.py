from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Collection, Iterable, Mapping

from src.core import (
    MASTER_DATA_DIR,
    UI_CATEGORIES,
    _matching_product_name,
    _ui_category,
    clean_text,
    discover_master_data,
    load_master_grade_markers,
    load_master_price_records,
    master_data_fingerprint,
    normalize_price_marker,
    normalize_product_name,
    normalize_specification,
    price_aliases,
)


CATALOG_RECORD_FIELDS = frozenset(
    {"catalog_id", "category", "product_name", "specification", "price_marker", "aliases", "status"}
)
CATALOG_STATUSES = frozenset({"启用", "待补充"})
CATALOG_ID_RE = re.compile(r"cat_[0-9a-f]{16}\Z")
MAX_QUERIES = 64
MAX_FILTER_VALUES = 32
MAX_PAGE_SIZE = 200
CATEGORY_EVIDENCE_RE = re.compile(r"沟槽|玛钢|球铁|球热|(?<![A-Za-z])QT(?![A-Za-z])", re.IGNORECASE)


def category_evidence(text: str) -> set[str]:
    return {match.group() if match.group() in {"沟槽", "玛钢"} else "球铁"
            for match in CATEGORY_EVIDENCE_RE.finditer(text)}


def inquiry_categories(document: Mapping[str, object]) -> set[str]:
    """A source-wide retrieval ceiling, never a row/title classification."""
    categories = {"玛钢"}
    for unit in document["units"]:
        for text in (unit["text"], unit.get("sheet", "")):
            categories.update(category_evidence(text))
    return categories


QUERY_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_QUERIES,
            "items": {
                "type": "object",
                "properties": {
                    "query_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "product_terms": {"type": "array", "maxItems": MAX_FILTER_VALUES, "items": {"type": "string"}},
                    "specifications": {"type": "array", "maxItems": MAX_FILTER_VALUES, "items": {"type": "string"}},
                    "categories": {"type": "array", "maxItems": MAX_FILTER_VALUES, "items": {"type": "string"}},
                    "price_markers": {"type": "array", "maxItems": MAX_FILTER_VALUES, "items": {"type": "string"}},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_PAGE_SIZE},
                },
                "required": ["query_id", "product_terms", "specifications", "categories", "price_markers"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class CatalogRecord:
    catalog_id: str
    category: str
    product_name: str
    specification: str
    price_marker: str
    aliases: tuple[str, ...]
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog_id": self.catalog_id,
            "category": self.category,
            "product_name": self.product_name,
            "specification": self.specification,
            "price_marker": self.price_marker,
            "aliases": list(self.aliases),
            "status": self.status,
        }


@dataclass(frozen=True)
class CatalogSnapshot:
    records: tuple[CatalogRecord, ...]
    grade_mappings: tuple[tuple[str, str], ...]
    catalog_fingerprint: str
    master_fingerprint: str | None = None
    master_verified: bool = False
    master_path: Path | None = None

    def __post_init__(self) -> None:
        ids = [record.catalog_id for record in self.records]
        # Full catalog loaders require records; a restricted source may have none.
        if len(ids) != len(set(ids)):
            raise ValueError("候选目录 catalog_id 必须唯一")

    @classmethod
    def from_master(cls, path: str | Path | None = None) -> "CatalogSnapshot":
        master_path = discover_master_data(MASTER_DATA_DIR) if path is None else Path(path)
        workbook_bytes = master_path.read_bytes()
        fingerprint = sha256(workbook_bytes).hexdigest()
        prices = load_master_price_records(master_path, workbook_bytes)
        grade_markers = load_master_grade_markers(master_path, workbook_bytes)
        if master_data_fingerprint(master_path) != fingerprint:
            raise ValueError("主数据内容在目录快照构造期间发生变化")
        catalog = _catalog_from_prices(prices, grade_markers, fingerprint)
        snapshot = cls.from_eval_catalog(catalog)
        return cls(
            snapshot.records,
            snapshot.grade_mappings,
            snapshot.catalog_fingerprint,
            master_fingerprint=fingerprint,
            master_verified=True,
            master_path=master_path.resolve(),
        )

    @classmethod
    def from_eval_catalog(
        cls, catalog: object, expected_catalog_fingerprint: str | None = None
    ) -> "CatalogSnapshot":
        if not isinstance(catalog, dict) or set(catalog) != {"products", "grade_mappings"}:
            raise ValueError("冻结目录必须且只能包含 products 和 grade_mappings")
        products = catalog["products"]
        grade_mappings = catalog["grade_mappings"]
        if not isinstance(products, list) or not isinstance(grade_mappings, list):
            raise ValueError("冻结目录 products 和 grade_mappings 必须是数组")
        records = tuple(_record_from_payload(value) for value in products)
        ids = [record.catalog_id for record in records]
        if not records or len(ids) != len(set(ids)):
            raise ValueError("冻结目录必须非空且 catalog_id 唯一")
        mappings = tuple(_grade_mapping_from_payload(value) for value in grade_mappings)
        if len({raw for raw, _ in mappings}) != len(mappings):
            raise ValueError("冻结目录等级映射客户写法重复")
        normalized = {
            "products": [record.to_dict() for record in records],
            "grade_mappings": [{"raw": raw, "price_marker": marker} for raw, marker in mappings],
        }
        fingerprint = sha256(_canonical_bytes(normalized)).hexdigest()
        if expected_catalog_fingerprint is not None and fingerprint != expected_catalog_fingerprint:
            raise ValueError("冻结目录指纹不匹配")
        return cls(records, mappings, fingerprint)

    @classmethod
    def from_payload(
        cls, catalog: object, expected_catalog_fingerprint: str | None = None
    ) -> "CatalogSnapshot":
        return cls.from_eval_catalog(catalog, expected_catalog_fingerprint)

    def to_payload(self) -> dict[str, object]:
        return {
            "products": [record.to_dict() for record in self.records],
            "grade_mappings": [
                {"raw": raw, "price_marker": marker} for raw, marker in self.grade_mappings
            ],
        }

    def restrict_categories(self, categories: Collection[str]) -> "CatalogSnapshot":
        if isinstance(categories, (str, bytes)) or not set(categories) <= UI_CATEGORIES:
            raise ValueError("目录范围只能包含 UI 分类")
        records = tuple(record for record in self.records if record.category in categories)
        payload = {**self.to_payload(), "products": [record.to_dict() for record in records]}
        return replace(self, records=records, catalog_fingerprint=sha256(_canonical_bytes(payload)).hexdigest())

    def assert_equivalent(self, other: "CatalogSnapshot") -> dict[str, object]:
        if not isinstance(other, CatalogSnapshot):
            raise TypeError("只能比较 CatalogSnapshot")
        if not (self.master_verified or other.master_verified):
            raise ValueError("两份冻结目录都未由当前主表构造，不能证明主表对应关系")
        if self.to_payload() != other.to_payload():
            raise ValueError("冻结目录与主表重建目录不一致")
        live = self if self.master_verified else other
        return {
            "status": "verified",
            "catalog_fingerprint": self.catalog_fingerprint,
            "master_fingerprint": live.master_fingerprint,
            "record_count": len(self.records),
        }

    def assert_current(self) -> str:
        if not self.master_verified or self.master_path is None or self.master_fingerprint is None:
            raise ValueError("该冻结目录未绑定已验证的当前主表")
        current_path = discover_master_data(self.master_path.parent).resolve()
        if current_path != self.master_path or master_data_fingerprint(current_path) != self.master_fingerprint:
            raise ValueError("当前主表内容已变化，请停止并重新构造目录快照")
        return self.master_fingerprint

    def index_payload(self) -> dict[str, object]:
        grouped: dict[tuple[str, str, str], set[str]] = {}
        for record in self.records:
            grouped.setdefault((record.category, record.product_name, record.status), set()).update(record.aliases)
        products = [
            {
                "category": category,
                "product_name": product_name,
                "aliases": sorted(aliases),
                "status": status,
            }
            for (category, product_name, status), aliases in sorted(grouped.items())
        ]
        return {
            "catalog_fingerprint": self.catalog_fingerprint,
            "master_verified": self.master_verified,
            "record_count": len(self.records),
            "products": products,
            "grade_mappings": [
                {"raw": raw, "price_marker": marker} for raw, marker in self.grade_mappings
            ],
        }

    def query_batch(self, queries: object) -> dict[str, object]:
        parsed = _validate_queries(queries)
        known_terms = {
            term
            for record in self.records
            for term in _record_product_terms(record)
        }
        known_specs = {normalize_specification(record.specification) for record in self.records}
        known_categories = {record.category for record in self.records}
        known_markers = {record.price_marker for record in self.records}
        results = []
        for query in parsed:
            product_terms = {_normalize_product_term(value) for value in query["product_terms"]}
            specifications = {normalize_specification(value) for value in query["specifications"]}
            categories = set(query["categories"])
            price_markers = set(query["price_markers"])
            matches = [
                record
                for record in self.records
                if (not product_terms or product_terms & _record_product_terms(record))
                and (not specifications or normalize_specification(record.specification) in specifications)
                and (not categories or record.category in categories)
                and (not price_markers or record.price_marker in price_markers)
            ]
            offset = query.get("offset", 0)
            limit = query.get("limit")
            page = matches[offset:] if limit is None else matches[offset : offset + limit]
            next_offset = offset + len(page) if offset + len(page) < len(matches) else None
            results.append(
                {
                    "query_id": query["query_id"],
                    "status": "matched" if matches else "no_matches",
                    "total_matches": len(matches),
                    "returned_count": len(page),
                    "offset": offset,
                    "has_more": next_offset is not None,
                    "next_offset": next_offset,
                    "unknown_filters": {
                        "product_terms": [value for value in query["product_terms"] if _normalize_product_term(value) not in known_terms],
                        "specifications": [value for value in query["specifications"] if normalize_specification(value) not in known_specs],
                        "categories": [value for value in query["categories"] if value not in known_categories],
                        "price_markers": [value for value in query["price_markers"] if value not in known_markers],
                    },
                    "candidates": [record.to_dict() for record in page],
                }
            )
        return {"catalog_fingerprint": self.catalog_fingerprint, "results": results}

    def validate_proposal(
        self, items: object, *, allowed_catalog_ids: Collection[str]
    ) -> dict[str, object]:
        return validate_proposal(items, self, allowed_catalog_ids=allowed_catalog_ids)


def _record_from_payload(value: object) -> CatalogRecord:
    if not isinstance(value, dict) or set(value) != CATALOG_RECORD_FIELDS:
        raise ValueError("冻结目录记录字段不符合安全白名单")
    scalar_fields = ("catalog_id", "category", "product_name", "specification", "price_marker", "status")
    if any(not isinstance(value[field], str) or not value[field] for field in scalar_fields):
        raise ValueError("冻结目录记录标量字段必须是非空字符串")
    if not CATALOG_ID_RE.fullmatch(value["catalog_id"]):
        raise ValueError("冻结目录 catalog_id 格式非法")
    if value["category"] not in UI_CATEGORIES:
        raise ValueError("冻结目录分类非法")
    if value["price_marker"] not in {"a", "A", "b"}:
        raise ValueError("冻结目录价格标记非法")
    if value["status"] not in CATALOG_STATUSES:
        raise ValueError("冻结目录状态非法")
    if normalize_specification(value["specification"]) != value["specification"]:
        raise ValueError("冻结目录规格未规范化")
    aliases = value["aliases"]
    if not isinstance(aliases, list) or any(not isinstance(alias, str) or not clean_text(alias) for alias in aliases):
        raise ValueError("冻结目录 aliases 必须是非空字符串数组")
    if aliases != sorted(set(aliases)):
        raise ValueError("冻结目录 aliases 必须去重排序")
    return CatalogRecord(
        value["catalog_id"], value["category"], value["product_name"], value["specification"],
        value["price_marker"], tuple(aliases), value["status"]
    )


def _grade_mapping_from_payload(value: object) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) != {"raw", "price_marker"}:
        raise ValueError("冻结目录等级映射字段非法")
    raw, marker = value["raw"], value["price_marker"]
    if not isinstance(raw, str) or not clean_text(raw) or marker not in {"a", "b"}:
        raise ValueError("冻结目录等级映射值非法")
    return raw, marker


def _catalog_from_prices(prices: Iterable[object], grade_markers: Mapping[str, str], fingerprint: str) -> dict[str, object]:
    prices = list(prices)
    categories = {_ui_category(price) for price in prices}
    aliases_by_category = {category: price_aliases(prices, category) for category in categories}
    mapped_aliases = {category: set() for category in categories}
    formal_targets = {category: set() for category in categories}
    target_display: dict[tuple[str, str], str] = {}
    for price in prices:
        category = _ui_category(price)
        for mapping_category, alias, targets in price.multi_target_aliases:
            if mapping_category != price.category:
                continue
            mapped_aliases[category].add(_matching_product_name(alias))
            for target in targets:
                normalized = _matching_product_name(target)
                formal_targets[category].add(normalized)
                target_display[(category, normalized)] = target

    projected: dict[tuple[object, ...], set[str]] = {}
    for price in prices:
        category = _ui_category(price)
        canonical = _matching_product_name(price.product_name)
        formal_canonical = normalize_product_name(price.product_name)
        source_terms = {canonical, *(_matching_product_name(alias) for alias in price.aliases)}
        expansions: set[str] = set()
        for mapping_category, alias, targets in price.multi_target_aliases:
            if mapping_category == price.category and _matching_product_name(alias) in source_terms:
                expansions.update(_matching_product_name(target) for target in targets)
        direct = set() if formal_canonical in mapped_aliases[category] and canonical not in formal_targets[category] else {canonical}
        for target in direct | expansions:
            name = price.product_name if target == canonical else target_display[(category, target)]
            key = (
                fingerprint, price.sheet_name, price.source_row, target, category, name,
                price.specification, price.price_marker, price.status,
            )
            projected.setdefault(key, set()).update(
                alias for alias, targets in aliases_by_category[category].items() if target in targets
            )

    products = []
    for key, aliases in projected.items():
        master_fingerprint, sheet, source_row, target, category, name, specification, marker, status = key
        identity = json.dumps(
            [master_fingerprint, sheet, source_row, target], ensure_ascii=False, separators=(",", ":")
        )
        products.append(
            {
                "catalog_id": "cat_" + sha256(identity.encode("utf-8")).hexdigest()[:16],
                "category": category,
                "product_name": name,
                "specification": normalize_specification(specification),
                "price_marker": marker,
                "aliases": sorted(aliases),
                "status": status,
            }
        )
    products.sort(
        key=lambda item: (
            item["category"], item["product_name"], item["specification"],
            item["price_marker"], item["status"],
        )
    )
    mappings = dict(grade_markers)
    for raw in ("黑", "蓝"):
        marker = normalize_price_marker(raw, grade_markers)
        if marker:
            mappings[raw] = marker
    mappings = [{"raw": raw, "price_marker": marker} for raw, marker in mappings.items()]
    mappings.sort(key=lambda item: (item["raw"], item["price_marker"]))
    return {"products": products, "grade_mappings": mappings}


def _normalize_product_term(value: str) -> str:
    return normalize_product_name(value).casefold()


def _record_product_terms(record: CatalogRecord) -> set[str]:
    return {
        _normalize_product_term(term)
        for term in (record.product_name, *record.aliases)
    }


def _validate_queries(queries: object) -> list[dict[str, object]]:
    if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_QUERIES:
        raise ValueError(f"queries 必须是 1 至 {MAX_QUERIES} 项数组")
    required = {"query_id", "product_terms", "specifications", "categories", "price_markers"}
    allowed = required | {"offset", "limit"}
    parsed: list[dict[str, object]] = []
    query_ids: set[str] = set()
    for query in queries:
        if not isinstance(query, dict) or not required.issubset(query) or not set(query).issubset(allowed):
            raise ValueError("query 字段非法")
        query_id = query["query_id"]
        if not isinstance(query_id, str) or not clean_text(query_id) or len(query_id) > 128 or query_id in query_ids:
            raise ValueError("query_id 必须是唯一的非空短字符串")
        query_ids.add(query_id)
        for field in ("product_terms", "specifications", "categories", "price_markers"):
            values = query[field]
            if not isinstance(values, list) or len(values) > MAX_FILTER_VALUES:
                raise ValueError(f"{field} 必须是至多 {MAX_FILTER_VALUES} 项的字符串数组")
            if any(not isinstance(value, str) or not clean_text(value) for value in values):
                raise ValueError(f"{field} 不能包含空值或非字符串")
        offset = query.get("offset", 0)
        limit = query.get("limit")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset 必须是非负整数")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_SIZE
        ):
            raise ValueError(f"limit 必须是 1 至 {MAX_PAGE_SIZE} 的整数")
        parsed.append(dict(query))
    return parsed


def validate_proposal(
    items: object,
    catalog: CatalogSnapshot,
    *,
    allowed_catalog_ids: Collection[str],
) -> dict[str, object]:
    if not isinstance(catalog, CatalogSnapshot):
        raise TypeError("catalog 必须是 CatalogSnapshot")
    if not isinstance(items, list):
        raise ValueError("items 必须是数组")
    if isinstance(allowed_catalog_ids, (str, bytes)):
        raise ValueError("allowed_catalog_ids 必须是 ID 集合")
    try:
        allowed = set(allowed_catalog_ids)
    except TypeError as exc:
        raise ValueError("allowed_catalog_ids 必须是 ID 集合") from exc
    by_id = {record.catalog_id: record for record in catalog.records}
    if any(not isinstance(catalog_id, str) or catalog_id not in by_id for catalog_id in allowed):
        raise ValueError("allowed_catalog_ids 含目录外 ID")
    grade_mappings = dict(catalog.grade_mappings)
    reports = []
    all_errors: list[dict[str, str]] = []
    seen_item_ids: set[str] = set()
    for index, item in enumerate(items):
        errors: list[dict[str, str]] = []
        unresolved: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []
        if not isinstance(item, dict):
            item_id = f"index:{index}"
            errors.append(_issue("item_not_object", "项目必须是对象"))
            item = {}
        else:
            item_id = item.get("item_id")
            if not isinstance(item_id, str) or not clean_text(item_id) or item_id in seen_item_ids:
                item_id = f"index:{index}"
                errors.append(_issue("invalid_item_id", "item_id 必须是唯一非空字符串"))
            else:
                seen_item_ids.add(item_id)

        candidate_ids = item.get("candidate_ids")
        if not isinstance(candidate_ids, list) or any(not isinstance(value, str) for value in candidate_ids):
            errors.append(_issue("invalid_candidate_ids", "candidate_ids 必须是字符串数组"))
            candidate_ids = []
        elif len(candidate_ids) != len(set(candidate_ids)):
            errors.append(_issue("duplicate_candidate_id", "candidate_ids 不得重复"))
        for candidate_id in candidate_ids:
            if candidate_id not in by_id:
                errors.append(_issue("candidate_id_not_found", f"候选 ID 不存在：{candidate_id}"))
            elif candidate_id not in allowed:
                errors.append(_issue("candidate_id_not_visible", f"候选 ID 未在本次 session 返回：{candidate_id}"))

        selected = item.get("selected_candidate_id")
        if selected is not None and not isinstance(selected, str):
            errors.append(_issue("invalid_selected_candidate_id", "selected_candidate_id 必须是字符串或 null"))
            selected = None
        selected_record = by_id.get(selected) if isinstance(selected, str) else None
        if selected is not None:
            if selected_record is None:
                errors.append(_issue("selected_id_not_found", f"所选 ID 不存在：{selected}"))
            if selected not in candidate_ids:
                errors.append(_issue("selected_not_candidate", "所选 ID 不属于该项 candidate_ids"))
            if selected not in allowed:
                errors.append(_issue("selected_id_not_visible", "所选 ID 未在本次 session 返回"))

        model_unresolved = item.get("unresolved", [])
        if not isinstance(model_unresolved, list) or any(not isinstance(value, str) for value in model_unresolved):
            errors.append(_issue("invalid_unresolved", "unresolved 必须是字符串数组"))
            model_unresolved = []
        if model_unresolved:
            unresolved.append(_issue("model_declared_unresolved", "模型为该项保留了未决信息"))

        marker = item.get("price_marker")
        grade_raw = item.get("grade_raw")
        if grade_raw is not None and not isinstance(grade_raw, str):
            errors.append(_issue("invalid_grade_raw", "grade_raw 必须是字符串或 null"))
            grade_raw = None
        raw_grade = clean_text(grade_raw)
        if raw_grade:
            expected_marker = normalize_price_marker(raw_grade, grade_mappings)
            if not expected_marker:
                unresolved.append(_issue("unknown_grade", "grade_raw 未映射到已知 a/A/b"))
                if marker is not None:
                    errors.append(_issue("unknown_grade_has_marker", "未知等级不能生成 price_marker"))
                if selected is not None:
                    errors.append(_issue("unknown_grade_selected", "未知等级不能选择目录 ID"))
            elif marker is None:
                marker = expected_marker
            elif marker != expected_marker:
                errors.append(_issue("grade_marker_mismatch", "price_marker 与 grade_raw 的已知映射不一致"))
        elif marker is None:
            marker = "b"
        elif marker not in {"a", "A", "b"}:
            errors.append(_issue("invalid_price_marker", "price_marker 必须是 a、A、b 或 null"))

        if "grade_raw" not in item:
            warnings.append(_issue("grade_raw_missing", "缺少 grade_raw，程序不能自动证明原始等级语义"))

        if selected_record is not None:
            _compare_selected_fields({**item, "price_marker": marker}, selected_record, errors)
            if selected_record.status == "待补充":
                unresolved.append(_issue("selected_record_pending", "所选目录记录为待补充，禁止据此自动报价"))
        elif selected is None and not unresolved:
            unresolved.append(_issue("selection_missing", "该项未选择目录记录"))

        all_errors.extend({"item_id": item_id, **error} for error in errors)
        reports.append(
            {
                "item_id": item_id,
                "errors": errors,
                "unresolved": unresolved,
                "warnings": warnings,
                "resolved_price_marker": marker,
                "selected_record": selected_record.to_dict() if selected_record else None,
            }
        )
    return {
        "proposal_valid": not all_errors,
        "quote_authorized": False,
        "catalog_fingerprint": catalog.catalog_fingerprint,
        "allowed_catalog_scope": "session",
        "errors": all_errors,
        "items": reports,
        "limitations": [
            "目录一致性校验不能自动证明标题继承、数量或全部原始等级语义正确",
            "结构一致或目录匹配均不授予自动报价许可",
        ],
    }


def _compare_selected_fields(
    item: Mapping[str, object], record: CatalogRecord, errors: list[dict[str, str]]
) -> None:
    category = item.get("category")
    if not isinstance(category, str) or category != record.category:
        errors.append(_issue("selected_category_mismatch", "所选记录分类与提案不一致或提案分类缺失"))
    product_name = item.get("product_name")
    names = _record_product_terms(record)
    if not isinstance(product_name, str) or _normalize_product_term(product_name) not in names:
        errors.append(_issue("selected_product_name_mismatch", "所选记录品名与提案正式名/已确认别名不一致"))
    specification = item.get("specification")
    if not isinstance(specification, str) or normalize_specification(specification) != record.specification:
        errors.append(_issue("selected_specification_mismatch", "所选记录规格与提案不一致"))
    if item.get("price_marker") != record.price_marker:
        errors.append(_issue("selected_price_marker_mismatch", "所选记录档位与提案不一致"))


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}
