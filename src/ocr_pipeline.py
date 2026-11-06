from __future__ import annotations

from difflib import SequenceMatcher
from html.parser import HTMLParser
from hashlib import sha256
import json
import os
import re
from pathlib import Path
import site
import stat
import sys
from tempfile import TemporaryDirectory
from threading import Lock

from .core import (
    PriceRecord,
    ProductRecord,
    _name_candidates,
    load_master_grade_markers,
    load_master_price_records,
    normalize_price_marker,
    normalize_product_name,
    normalize_specification,
    price_aliases,
)
from .input_router import CATEGORY_TITLE_RE, GRADE_TITLE_PREFIX_RE, _records_from_ai, _validate_ai_items, validate_image


QUOTE_FIELDS = (
    "product_name_raw",
    "specification_raw",
    "quantity",
    "grade_or_bag_raw",
    "source_text",
    "uncertain_fields",
)
# 正式别名完全来自主数据；多目标别名必须在候选阶段展开，不能在代码中唯一化。
BUSINESS_ALIASES: dict[str, str] = {}
GRADE_RE = re.compile(r"(?:蓝袋?|黑袋?|中|好|(?<![A-Za-z])[aAbB](?![A-Za-z]))")
SPEC_RE = re.compile(r"\d+(?:\s*[xX×*\-]\s*\d+[GSgs]?)*(?:[GSgs])?")
QUANTITY_RE = re.compile(
    r"(?P<quantity>\d+(?:\.\d+)?)\s*(?P<unit>件|箱)\s*"
    r"(?P<grade>[（(]?\s*(?:蓝袋?|黑袋?|中|好|(?<![A-Za-z])[aAbB](?![A-Za-z]))\s*[）)]?)?"
)
IGNORABLE_RE = re.compile(r"[\s|:：,，。;；()（）#=\-]+")


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _clean_grade(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().strip("()（） ").removesuffix("袋") or None


def _bbox_overlap(left: list[int] | None, right: list[int] | None) -> bool:
    if not left or not right:
        return False
    return max(left[0], right[0]) < min(left[2], right[2]) and max(left[1], right[1]) < min(left[3], right[3])


def _remove_once(text: str, value: str) -> str:
    index = text.find(value)
    return text if index < 0 else text[:index] + text[index + len(value) :]


class RuleParser:
    """Convert Paddle document blocks into auditable quotation fields."""

    def __init__(self, prices: list[PriceRecord], grade_markers: dict[str, str] | None = None) -> None:
        self.prices = list(prices)
        self.grade_markers = grade_markers or {}
        self.aliases = price_aliases(self.prices)
        self.price_pairs = {(normalize_product_name(row.product_name), normalize_specification(row.specification)) for row in self.prices}
        self.price_specs = {spec for _, spec in self.price_pairs}
        aliases = set(self.aliases) | set(BUSINESS_ALIASES)
        self.product_words = sorted({row.product_name for row in self.prices} | aliases, key=len, reverse=True)

    def parse_document(self, payload: dict) -> dict:
        blocks = self._blocks(payload)
        spatial_columns = self._spatial_columns(blocks)
        parsed: list[dict] = []
        block_audit: list[dict] = []
        contexts: dict[tuple[int, object], dict[str, str | None]] = {}
        used_paired_line_inference = False
        for block in blocks:
            block_id = block["block_id"]
            label = block["label"]
            content = block["content"]
            spatial_column = spatial_columns.get(block_id)
            block_column = spatial_column if spatial_column is not None else 0
            block_context_key: object = spatial_column if spatial_column is not None else block_id
            block_scope = (block["document_index"], block_context_key)
            rows = self._table_rows(content) if label == "table" else None
            lines = rows if rows is not None else [[line] for line in content.splitlines() if line.strip()]
            category_titles = [
                match["category"]
                for cells in lines
                if (match := CATEGORY_TITLE_RE.fullmatch(cells[0].strip()))
            ]
            grade_titles = [
                match["grade"]
                for cells in lines
                if (match := GRADE_TITLE_PREFIX_RE.match(cells[0].strip()))
            ]
            title_conflicts: list[str] = []
            if label in {"paragraph_title", "footer", "header"}:
                if label in {"paragraph_title", "header"} and category_titles and spatial_column is None:
                    for context_scope, existing_context in list(contexts.items()):
                        if context_scope[0] == block["document_index"]:
                            contexts[context_scope] = {**existing_context, "category": None}
                context = dict(contexts.get(block_scope, {"category": None, "grade": None}))
                if category_titles:
                    if len(set(category_titles)) == 1:
                        context["category"] = "球铁" if category_titles[0] == "球铁" else None
                    else:
                        context["category"] = None
                        title_conflicts.append("category")
                if grade_titles:
                    context["grade"] = grade_titles[0]
                contexts[block_scope] = context
                block_audit.append({**block, "status": "excluded_non_business_block", "title_context": context})
                continue
            line_candidates: list[list[dict]] = []
            for line_index, cells in enumerate(lines):
                original_line = cells[0].strip()
                if CATEGORY_TITLE_RE.fullmatch(original_line):
                    continue
                grade_title = GRADE_TITLE_PREFIX_RE.match(original_line)
                clean_line = original_line[grade_title.end() :].strip() if grade_title else original_line
                if not clean_line:
                    continue
                if rows is not None and len(cells) >= 2 and re.fullmatch(r"\d+(?:\.\d+)?", cells[-1].strip()):
                    candidates = [self._parse_segment(clean_line, block, line_index, quantity_override=float(cells[-1]), source_text=" | ".join(cells))]
                else:
                    candidates = [
                        self._parse_segment(
                            segment,
                            block,
                            line_index,
                            source_text=original_line if grade_title and segment_index == 0 else None,
                        )
                        for segment_index, segment in enumerate(self._split_line(clean_line))
                    ]
                line_candidates.append(candidates)

            paired = any(len(items) > 1 for items in line_candidates)
            used_paired_line_inference |= paired
            column_counts = [0, 0]
            flat: list[dict] = []
            for items in line_candidates:
                if paired and len(items) > 1:
                    for slot, item in enumerate(items):
                        item["audit"]["segment_index"] = slot
                        item["audit"]["column"] = slot
                        item["audit"]["steps"].append("column_from_repeated_line_slot")
                        column_counts[slot] += 1
                elif paired:
                    slot = 0 if column_counts[0] <= column_counts[1] else 1
                    items[0]["audit"]["column"] = slot
                    items[0]["audit"]["steps"].append("column_inferred_from_balanced_paired_lines")
                    column_counts[slot] += 1
                else:
                    items[0]["audit"]["column"] = block_column
                flat.extend(items)

            title_column = 0 if paired else block_column
            title_context_key: object = (block_id, title_column) if paired else block_context_key
            title_scope = (block["document_index"], title_context_key)
            title_context = dict(contexts.get(title_scope, {"category": None, "grade": None}))
            if category_titles:
                if len(set(category_titles)) == 1:
                    title_context["category"] = "球铁" if category_titles[0] == "球铁" else None
                else:
                    title_context["category"] = None
                    title_conflicts.append("category")
            if grade_titles:
                title_context["grade"] = grade_titles[0]
            contexts[title_scope] = title_context
            for candidate in flat:
                item, audit = candidate["item"], candidate["audit"]
                item_context_key: object = (block_id, audit["column"]) if paired else block_context_key
                item_scope = (block["document_index"], item_context_key)
                context = contexts.get(item_scope, {"category": None, "grade": None})
                audit["context_category"] = context["category"]
                audit["context_grade"] = context["grade"]
                audit["title_context_conflict"] = list(title_conflicts) if item_scope == title_scope else []
                if context["category"]:
                    audit["steps"].append("category_from_section_title")
                if context["grade"]:
                    explicit_grade = item["grade_or_bag_raw"]
                    if explicit_grade is None:
                        item["grade_or_bag_raw"] = context["grade"]
                        audit["steps"].append("grade_from_title_prefix")
                    else:
                        explicit_marker = normalize_price_marker(explicit_grade, self.grade_markers)
                        if not explicit_marker:
                            explicit_marker = normalize_price_marker(f"{explicit_grade}袋", self.grade_markers)
                        context_marker = normalize_price_marker(context["grade"], self.grade_markers)
                        if explicit_marker != context_marker:
                            item["uncertain_fields"] = list(dict.fromkeys(item["uncertain_fields"] + ["grade_or_bag_raw"]))
                            audit["uncertain_reasons"].append("row_grade_conflicts_with_title_grade")
                            audit["title_context_conflict"].append("grade")
            parsed.extend(flat)
            block_audit.append({**block, "status": "parsed", "candidate_count": len(flat), "title_context": title_context})

        parsed, duplicates = self._deduplicate(parsed)
        parsed.sort(key=lambda row: (row["audit"]["column"], row["audit"]["bbox"][1], row["audit"]["line_index"], row["audit"]["segment_index"]))
        return {
            "items": [row["item"] for row in parsed],
            "audit": {
                "blocks": block_audit,
                "items": [row["audit"] for row in parsed],
                "duplicates": duplicates,
                "price_record_count": len(self.prices),
                "reading_order": {
                    "method": "merged_line_pattern_only" if used_paired_line_inference else "bbox_columns" if any(spatial_columns.values()) else "source_order",
                    "exact_spatial_order_available": not used_paired_line_inference,
                },
            },
        }

    @staticmethod
    def _blocks(payload: dict) -> list[dict]:
        raw = payload.get("raw", {})
        documents = raw if isinstance(raw, list) else [raw]
        blocks = []
        for document_index, document in enumerate(documents):
            for index, block in enumerate(document.get("res", {}).get("parsing_res_list", [])):
                blocks.append(
                    {
                        "block_id": f"d{document_index}-b{index}",
                        "document_index": document_index,
                        "label": block.get("block_label"),
                        "content": block.get("block_content", ""),
                        "bbox": block.get("block_bbox"),
                        "order": block.get("block_order"),
                    }
                )
        return blocks

    @staticmethod
    def _table_rows(content: str) -> list[list[str]] | None:
        if "<table" not in content.lower():
            return None
        parser = _TableParser()
        parser.feed(content)
        return [row for row in parser.rows if any(cell for cell in row)]

    @staticmethod
    def _spatial_columns(blocks: list[dict]) -> dict[str, int]:
        columns: dict[str, int] = {}
        for document_index in {block["document_index"] for block in blocks}:
            text_blocks = [
                block
                for block in blocks
                if block["document_index"] == document_index and block["label"] == "text" and block["bbox"]
            ]
            title_blocks = [
                block
                for block in blocks
                if block["document_index"] == document_index
                and block["label"] in {"paragraph_title", "header"}
                and block["bbox"]
                and any(
                    CATEGORY_TITLE_RE.fullmatch(line.strip()) or GRADE_TITLE_PREFIX_RE.match(line.strip())
                    for line in block["content"].splitlines()
                )
            ]
            layout_blocks = text_blocks + title_blocks
            if len(layout_blocks) < 2:
                continue
            if max(block["bbox"][0] for block in layout_blocks) < min(block["bbox"][2] for block in layout_blocks):
                columns.update({block["block_id"]: 0 for block in layout_blocks})
                continue
            if len(text_blocks) < 2:
                continue
            centers = sorted(
                (((block["bbox"][0] + block["bbox"][2]) / 2, block) for block in text_blocks),
                key=lambda pair: pair[0],
            )
            gaps = [(centers[index + 1][0] - centers[index][0], index) for index in range(len(centers) - 1)]
            gap, split = max(gaps)
            typical_width = sorted(block["bbox"][2] - block["bbox"][0] for block in text_blocks)[len(text_blocks) // 2]
            if gap <= max(250, typical_width * 0.75):
                continue
            left_ids = {block["block_id"] for _, block in centers[: split + 1]}
            columns.update({block["block_id"]: 0 if block["block_id"] in left_ids else 1 for block in text_blocks})
            boundary = (centers[split][0] + centers[split + 1][0]) / 2
            for block in title_blocks:
                if block["bbox"][2] <= boundary:
                    columns[block["block_id"]] = 0
                elif block["bbox"][0] >= boundary:
                    columns[block["block_id"]] = 1
        return columns

    @staticmethod
    def _split_line(line: str) -> list[str]:
        matches = list(QUANTITY_RE.finditer(line))
        if not matches:
            return [line.strip()]
        segments = []
        start = 0
        for match in matches:
            segment = line[start : match.end()].strip()
            if segment:
                segments.append(segment)
            start = match.end()
        tail = line[start:].strip()
        if tail:
            segments.append(tail)
        return segments

    def _parse_segment(
        self,
        text: str,
        block: dict,
        line_index: int,
        *,
        quantity_override: float | None = None,
        source_text: str | None = None,
    ) -> dict:
        source = (source_text or text).strip()
        quantity_match = None if quantity_override is not None else next(iter(QUANTITY_RE.finditer(text)), None)
        quantity = quantity_override if quantity_override is not None else (float(quantity_match["quantity"]) if quantity_match else None)
        grade = _clean_grade(quantity_match["grade"] if quantity_match else None)
        prefix = text if quantity_match is None else text[: quantity_match.start()]

        angle_match = re.search(r"(?:45|90)\s*(?:°|度)", prefix)
        specs = [match for match in SPEC_RE.finditer(prefix) if not angle_match or match.span() != angle_match.span()]
        spec_match = max(specs, key=lambda match: ("x" in normalize_specification(match.group()), len(match.group())), default=None)
        specification = spec_match.group().replace(" ", "") if spec_match else None

        product_matches = []
        for word in self.product_words:
            position = prefix.find(word)
            if position >= 0:
                product_matches.append((len(word), position, word))
        _, product_position, product = max(product_matches, default=(0, -1, None))
        if product == "弯头" and angle_match:
            product = angle_match.group().replace("度", "°").replace(" ", "") + "弯头"

        leftover = prefix
        if spec_match:
            leftover = _remove_once(leftover, spec_match.group())
        if product_matches:
            original_product = max(product_matches)[2]
            leftover = _remove_once(leftover, original_product)
        if angle_match:
            leftover = _remove_once(leftover, angle_match.group())
        if grade is None:
            grade_match = GRADE_RE.search(leftover)
            if grade_match and grade_match.group() not in {"中"} or (grade_match and "中大" not in leftover):
                grade = _clean_grade(grade_match.group())
                leftover = _remove_once(leftover, grade_match.group())
        significant_leftover = IGNORABLE_RE.sub("", leftover)

        canonical_candidates = self._price_candidates(product, specification)
        uncertain: list[str] = []
        reasons: list[str] = []
        if product is None:
            uncertain.append("product_name_raw")
            reasons.append("product_not_in_formal_dictionary")
        if specification is None:
            uncertain.append("specification_raw")
            reasons.append("specification_not_found")
        elif normalize_specification(specification) not in self.price_specs:
            uncertain.append("specification_raw")
            reasons.append("specification_not_in_price_table")
        if quantity is None or quantity <= 0:
            uncertain.append("quantity")
            reasons.append("valid_quantity_with_unit_not_found")
        if product and specification:
            if not canonical_candidates:
                uncertain.extend(["product_name_raw", "specification_raw"])
                reasons.append("price_table_product_spec_not_found")
            elif len(canonical_candidates) > 1:
                uncertain.append("product_name_raw")
                reasons.append("price_table_product_candidate_not_unique")
        if significant_leftover:
            uncertain.extend(["source_text", "product_name_raw"])
            reasons.append("unconsumed_source_text")
        if grade is None and re.search(r"[（(][^）)]{1,4}[）)]", text):
            uncertain.append("grade_or_bag_raw")
            reasons.append("unrecognized_parenthesized_grade")
        uncertain = list(dict.fromkeys(uncertain))

        steps = ["quantity_from_table_cell" if quantity_override is not None else "quantity_before_unit"]
        if spec_match:
            steps.append("specification_regex")
        if product:
            steps.append("longest_formal_product_match")
        if product in BUSINESS_ALIASES:
            steps.append("confirmed_business_alias_mapping")
        if canonical_candidates:
            steps.append("price_table_product_spec_validation")
        return {
            "item": {
                "product_name_raw": product,
                "specification_raw": specification,
                "quantity": quantity,
                "grade_or_bag_raw": grade,
                "source_text": source,
                "uncertain_fields": uncertain,
            },
            "audit": {
                "block_id": block["block_id"],
                "bbox": block["bbox"],
                "block_order": block["order"],
                "column": 0,
                "line_index": line_index,
                "segment_index": 0,
                "source_text": source,
                "steps": steps,
                "canonical_product_candidates": canonical_candidates,
                "uncertain_reasons": list(dict.fromkeys(reasons)),
                "unconsumed_text": significant_leftover,
                "source_consumed": not significant_leftover,
            },
        }

    def _price_candidates(self, product: str | None, specification: str | None) -> list[str]:
        if not product or not specification:
            return []
        semantic_product = BUSINESS_ALIASES.get(product, product)
        names = _name_candidates(semantic_product, specification, self.aliases)
        normalized_spec = normalize_specification(specification)
        candidates = {name for name, spec in self.price_pairs if name in names and spec == normalized_spec}
        suffix = specification[-1:].upper()
        if suffix == "G":
            groove = {name for name in candidates if "沟槽" in name}
            candidates = groove or candidates
        elif suffix == "S":
            threaded = {name for name in candidates if "丝接" in name}
            candidates = threaded or candidates
        return sorted(candidates)

    def _deduplicate(self, rows: list[dict]) -> tuple[list[dict], list[dict]]:
        kept: list[dict] = []
        duplicates: list[dict] = []
        for row in rows:
            item = row["item"]
            key = (
                normalize_product_name(BUSINESS_ALIASES.get(item["product_name_raw"], item["product_name_raw"] or ""), self.aliases),
                normalize_specification(item["specification_raw"]),
                item["quantity"],
            )
            duplicate = next(
                (
                    existing
                    for existing in kept
                    if existing["audit"]["block_id"] != row["audit"]["block_id"]
                    and _bbox_overlap(existing["audit"]["bbox"], row["audit"]["bbox"])
                    and key == (
                        normalize_product_name(BUSINESS_ALIASES.get(existing["item"]["product_name_raw"], existing["item"]["product_name_raw"] or ""), self.aliases),
                        normalize_specification(existing["item"]["specification_raw"]),
                        existing["item"]["quantity"],
                    )
                    and all(key)
                ),
                None,
            )
            if duplicate is None:
                kept.append(row)
                continue
            if duplicate["item"]["grade_or_bag_raw"] != item["grade_or_bag_raw"]:
                duplicate["item"]["grade_or_bag_raw"] = None
                duplicate["item"]["uncertain_fields"] = list(dict.fromkeys(duplicate["item"]["uncertain_fields"] + ["grade_or_bag_raw"]))
                duplicate["audit"]["uncertain_reasons"].append("overlapping_duplicate_grade_conflict")
            row["audit"]["column"] = duplicate["audit"]["column"]
            duplicates.append(
                {
                    "kept_block_id": duplicate["audit"]["block_id"],
                    "removed_block_id": row["audit"]["block_id"],
                    "removed_source_text": row["item"]["source_text"],
                    "key": key,
                    "source_consumed_as_duplicate": True,
                }
            )
            for sibling in rows:
                if sibling["audit"]["block_id"] == row["audit"]["block_id"]:
                    sibling["audit"]["column"] = duplicate["audit"]["column"]
        return kept, duplicates


OCR_CANDIDATE_RELATIONS = {
    "占": "古",
    "头": "弯头",
    "斜头": "弯头",
    "字头": "弯头",
    "梦头": "弯头",
    "7": "T",
}
MIN_SCORE = 60.0
MIN_MARGIN = 20.0


def _text_only(value: object) -> str:
    return "".join(re.findall(r"[A-Za-z\u4e00-\u9fff]", "" if value is None else str(value)))


def _edit_distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for index, left_char in enumerate(left, 1):
        next_row = [index]
        for column, right_char in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[column] + 1, row[column - 1] + (left_char != right_char)))
        row = next_row
    return row[-1]


def _semantic_family(value: str, aliases: dict[str, object] | None = None) -> str:
    normalized = normalize_product_name(BUSINESS_ALIASES.get(value, value), aliases)
    return next((suffix for suffix in ("弯头", "三通", "四通", "管箍") if normalized.endswith(suffix)), normalized)


class CandidateRanker:
    """Resolve only product-name uncertainty that passes every frozen safety gate."""

    def __init__(self, prices: list[PriceRecord]) -> None:
        self.prices = list(prices)
        self.aliases = price_aliases(self.prices)
        self.surfaces = sorted(set(self.aliases) | set(BUSINESS_ALIASES) | {row.product_name for row in self.prices})

    def legal_products(self, raw_product: str, specification: str) -> list[str]:
        semantic = BUSINESS_ALIASES.get(raw_product, raw_product)
        normalized = normalize_product_name(semantic)
        allowed = _name_candidates(semantic, specification, self.aliases)
        if normalized in {"弯头", "三通", "四通", "管箍"}:
            allowed |= {normalize_product_name(row.product_name) for row in self.prices if row.product_name.endswith(normalized)}
        spec = normalize_specification(specification)
        return sorted({row.product_name for row in self.prices if normalize_product_name(row.product_name) in allowed and normalize_specification(row.specification) == spec})

    def _candidate_surfaces(self, item: dict, audit: dict) -> set[str]:
        unknown = audit["unconsumed_text"]
        spec = item["specification_raw"] or ""
        surfaces = {surface for surface in self.surfaces if self.legal_products(surface, spec)}
        explicit = {target for source, target in OCR_CANDIDATE_RELATIONS.items() if source in unknown}
        surfaces |= explicit
        compact = _text_only(unknown)
        surfaces |= {
            surface for surface in self.surfaces
            if compact and (_edit_distance(compact, _text_only(surface)) <= 1 or compact in _text_only(surface) or _text_only(surface) in compact)
        }
        grouped: dict[str, str] = {}
        for surface in surfaces:
            family = _semantic_family(surface, self.aliases)
            preferred = next((target for target in explicit if _semantic_family(target, self.aliases) == family), None)
            grouped[family] = preferred or (family if family in {"弯头", "三通", "四通", "管箍"} else min(surface, grouped.get(family, surface), key=len))
        return set(grouped.values())

    def _score(self, surface: str, item: dict, audit: dict) -> dict:
        unknown = audit["unconsumed_text"]
        compact = _text_only(unknown)
        candidate_text = _text_only(surface)
        relation_sources = [source for source, target in OCR_CANDIDATE_RELATIONS.items() if source in unknown and _semantic_family(target, self.aliases) == _semantic_family(surface, self.aliases)]
        comparison = max(relation_sources or [compact], key=len)
        similarity = SequenceMatcher(None, comparison, candidate_text).ratio() if comparison and candidate_text else 0.0
        maximum = max(len(comparison), len(candidate_text), 1)
        edit_similarity = 1.0 - _edit_distance(comparison, candidate_text) / maximum if comparison and candidate_text else 0.0
        characters = set(comparison) | set(candidate_text)
        character_similarity = len(set(comparison) & set(candidate_text)) / len(characters) if characters else 0.0
        legal = self.legal_products(surface, item["specification_raw"] or "")
        score = similarity * 25 + max(0.0, edit_similarity) * 15 + character_similarity * 10
        score += 40 if relation_sources else 0
        score += 10 if comparison and candidate_text and (comparison in candidate_text or candidate_text in comparison) else 0
        score += 20 if legal else 0
        score += 5 if comparison.endswith("头") and candidate_text.endswith("头") else 0
        score += 2 if surface in self.aliases or surface in BUSINESS_ALIASES else 0
        matched = max(relation_sources, key=len) if relation_sources else (compact if compact and compact in candidate_text else "")
        remaining_key = _text_only(unknown.replace(matched, "", 1) if matched else unknown)
        score -= 8 * len(remaining_key)
        return {"product_name": surface, "score": round(score, 2), "legal_products": legal, "remaining_key_text": remaining_key}

    def rank(self, item: dict, audit: dict) -> dict:
        candidates = sorted((self._score(surface, item, audit) for surface in self._candidate_surfaces(item, audit)), key=lambda row: (-row["score"], row["product_name"]))[:12]
        top = candidates[0] if candidates else None
        margin = round(top["score"] - candidates[1]["score"], 2) if len(candidates) > 1 else (top["score"] if top else 0.0)
        blockers = []
        if not top:
            blockers.append("无候选")
        elif not top["legal_products"]:
            blockers.append("Top1 无当前规格合法价格组合")
        if top and top["score"] < MIN_SCORE:
            blockers.append(f"Top1 分数低于 {MIN_SCORE:g}")
        if top and margin < MIN_MARGIN:
            blockers.append(f"Top1 与 Top2 差值低于 {MIN_MARGIN:g}")
        if top and top["remaining_key_text"]:
            blockers.append("仍有未消费关键字符")
        if item["quantity"] is None or "quantity" in item["uncertain_fields"]:
            blockers.append("数量未可靠确定")
        if item["specification_raw"] is None or "specification_raw" in item["uncertain_fields"]:
            blockers.append("规格未可靠确定")
        return {"decision": None if blockers else _semantic_family(top["product_name"], self.aliases), "candidates": candidates, "margin": margin, "blockers": blockers}

    def resolve(self, parsed: dict) -> dict:
        decisions = []
        for item, audit in zip(parsed["items"], parsed["audit"]["items"]):
            if "product_name_raw" not in item["uncertain_fields"]:
                continue
            decision = self.rank(item, audit)
            decisions.append({"source_text": item["source_text"], **decision})
            if decision["decision"]:
                item["product_name_raw"] = decision["decision"]
                item["uncertain_fields"] = [field for field in item["uncertain_fields"] if field not in {"product_name_raw", "source_text"}]
        parsed["audit"]["ranking"] = decisions
        return parsed


def _prepare_paddle_cache(runtime_root: Path) -> str:
    """Keep the chosen cache, giving Windows native inference an ASCII path."""
    cache = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", str(runtime_root / "cache" / "paddleocr-vl-1.6"))).expanduser().absolute()
    frozen = sys.modules.get("paddlex.utils.cache")
    if frozen is not None:
        loaded = Path(frozen.CACHE_DIR).absolute()
        if (os.name == "nt" and not str(loaded).isascii()) or loaded.resolve() != cache.resolve():
            raise RuntimeError("PaddleX 已载入旧模型缓存路径，请重新启动询价助手后再识别")
    if os.name == "nt" and not str(cache).isascii():
        target = cache.resolve(strict=True)
        if not target.is_dir():
            raise RuntimeError("本地 Paddle 模型缓存目录不存在，请检查 runtime 配置")
        # Keep aliases beside the explicitly selected runtime, never at a drive root.
        scope = runtime_root.absolute().parent
        if scope == Path(scope.anchor) or not str(scope).isascii():
            raise RuntimeError("项目内没有可用的纯英文路径，请配置英文 PADDLE_PDX_CACHE_HOME")
        parent = next((path for path in target.parents if path.is_relative_to(scope) and str(path).isascii()), scope)
        suffix = sha256(os.path.normcase(str(target)).encode("utf-8")).hexdigest()[:12]
        alias = parent / ("paddle-cache-" + suffix)
        if not os.path.lexists(alias):
            # CPython's native junction API avoids shell parsing and model copies.
            import _winapi
            try:
                _winapi.CreateJunction(str(target), str(alias))
            except FileExistsError:
                pass  # Another initializer may have created the same junction.
        if (getattr(alias.lstat(), "st_reparse_tag", None) != stat.IO_REPARSE_TAG_MOUNT_POINT
                or alias.resolve(strict=True) != target):
            raise RuntimeError("Paddle 英文缓存访问路径已被其他目录占用，未修改该路径")
        cache = alias  # Do not resolve: native inference must receive the ASCII name.
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache)
    return str(cache)


# shortcut: OCR is serialized per process; use isolated workers for parallel OCR.
_OCR_LOCK = Lock()


class LocalOCRExtractor:
    """PaddleOCR-VL -> deterministic parser -> candidate ranker."""

    def __init__(self, master_data_path: Path, runtime_root: Path | None = None, device: str | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        self.runtime_root = runtime_root or root / "runtime"
        self.device = device or os.environ.get("CRP_PADDLE_DEVICE", "cpu")
        prices = load_master_price_records(master_data_path)
        self.grade_markers = load_master_grade_markers(master_data_path)
        self.parser = RuleParser(prices, self.grade_markers)
        self.ranker = CandidateRanker(prices)
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        _prepare_paddle_cache(self.runtime_root)
        try:
            from paddleocr import PaddleOCRVL
        except ImportError:
            packages = self.runtime_root / "paddleocr" / "Lib" / "site-packages"
            if not packages.is_dir():
                raise RuntimeError("缺少本地 PaddleOCR 运行环境：runtime/paddleocr") from None
            site.addsitedir(str(packages))
            from paddleocr import PaddleOCRVL
        import paddle
        with paddle.base.dygraph.guard(self.device):
            self._pipeline = PaddleOCRVL(
                pipeline_version="v1.6",
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_layout_detection=True,
                device=self.device,
                engine="paddle",
                enable_mkldnn=False,
                use_queues=False,
            )
        return self._pipeline

    def extract_raw(self, image_bytes: bytes) -> list[dict]:
        """Return local OCR evidence before product parsing or candidate ranking."""
        image_bytes = validate_image(image_bytes)
        with TemporaryDirectory(prefix="crp-ocr-") as directory:
            image_path = Path(directory) / "input.jpg"
            image_path.write_bytes(image_bytes)
            raw = []
            with _OCR_LOCK:
                pipeline = self._load_pipeline()
                import paddle
                # Streamlit reruns use new threads; the importing thread's mode can expire.
                with paddle.base.dygraph.guard(self.device):
                    for result in pipeline.predict(str(image_path)):
                        value = result.json
                        raw.append(json.loads(value) if isinstance(value, str) else value)
        return raw

    def extract_products(self, image_bytes: bytes, context: str | None = None) -> list[ProductRecord]:
        raw = self.extract_raw(image_bytes)
        parsed = self.ranker.resolve(self.parser.parse_document({"raw": raw}))
        _validate_ai_items({"items": parsed["items"]})
        records = _records_from_ai(
            {"items": parsed["items"]},
            context,
            "PaddleOCR-VL-1.6 + 规则 + 候选排序",
            self.grade_markers,
        )
        decisions = {decision["source_text"]: decision for decision in parsed["audit"].get("ranking", [])}
        for record, audit in zip(records, parsed["audit"]["items"]):
            record.category = audit.get("context_category") or None
            conflicts = audit.get("title_context_conflict", [])
            if "grade" in conflicts:
                record.price_marker = None
            if conflicts:
                record.warnings.append("行级证据与标题上下文冲突，已阻断自动确定")
            decision = decisions.get(record.original_text)
            if not decision:
                continue
            if decision["blockers"]:
                record.warnings.append("候选排序未自动确定：" + "、".join(decision["blockers"]))
        return records


