"""Title evidence for safe UI defaults; never parses or removes product rows."""
from collections import defaultdict
import re

from .agent_catalog import category_evidence
from .input_router import CATEGORY_TITLE_RE, GRADE_TITLE_PREFIX_RE
from .ocr_pipeline import RuleParser


def _category_title(text):
    title = CATEGORY_TITLE_RE.fullmatch(text.strip())
    if title:
        return title["category"]
    return "球铁" if re.fullmatch(r"(?:QT|球热)\s*[:：]?", text.strip(), re.IGNORECASE) else None


def inquiry_contexts(document):
    """Keep source/page/reading-space/sheet/column boundaries on title hints."""
    groups = defaultdict(list)
    for unit in document["units"]:
        groups[(unit.get("page"), unit.get("ocr_page"), unit.get("reading_method"), unit.get("sheet"))].append(unit)
    contexts = {}
    for units in groups.values():
        spatial = any("bbox" in unit or unit.get("reading_method") in {"local_ocr", "pdf_text"} for unit in units)
        blocks = [{"block_id": unit["unit_id"], "document_index": 0,
                   "label": unit.get("block_label") or "text", "bbox": unit.get("bbox"),
                   "content": "\n".join("球铁" if _category_title(line) == "球铁" else line
                                         for line in unit["text"].splitlines())} for unit in units]
        columns = RuleParser._spatial_columns(blocks) if spatial else {}
        if spatial and all(unit.get("bbox") for unit in units):
            units = sorted(units, key=lambda unit: (unit["bbox"][1], unit["bbox"][0]))
        states = {}
        uncertain_scopes = set()
        has_category_titles = any(_category_title(line) for unit in units for line in unit["text"].splitlines())
        has_grade_titles = any(GRADE_TITLE_PREFIX_RE.match(line.strip())
                               for unit in units for line in unit["text"].splitlines())
        for unit in units:
            # Unassigned geometry cannot borrow a heading from another block.
            scope = columns.get(unit["unit_id"], unit["unit_id"]) if spatial else "text"
            category, grade = states.get(scope, (None, None))
            table = RuleParser._table_rows(unit["text"])
            unassigned = spatial and unit["unit_id"] not in columns and len(units) > 1
            category_ambiguous = bool(unassigned and has_category_titles
                                      or table and any(category_evidence(cell) for row in table for cell in row))
            grade_ambiguous = bool(unassigned and has_grade_titles
                                   or table and any(GRADE_TITLE_PREFIX_RE.match(cell.strip()) for row in table for cell in row))
            lines = unit["text"].splitlines()
            if unit.get("sheet") is not None:
                # Excel cells carry their actual column; a heading never crosses cells.
                cells = [cell for line in lines for cell in line.split("\t")]
                entries = []
                for cell in cells:
                    match = re.fullmatch(r"([A-Z]+|列\d+)=(.*)", cell)
                    column, text = (match[1], match[2]) if match else (unit["unit_id"], cell)
                    cell_category, cell_grade = states.get(column, (_category_title(unit["sheet"]), None))
                    title = _category_title(text)
                    level = GRADE_TITLE_PREFIX_RE.match(text.strip())
                    if title:
                        cell_category = title
                    if level:
                        cell_grade = level["grade"]
                    states[column] = (cell_category, cell_grade)
                    entries.append((cell, cell_category, cell_grade, not bool(match), not bool(match)))
                contexts[unit["unit_id"]] = entries
                continue
            if spatial:
                # Existing OCR rule: scan a block's headings before its product text.
                titles = {_category_title(line) for line in lines} - {None}
                levels = {match["grade"] for line in lines
                          if (match := GRADE_TITLE_PREFIX_RE.match(line.strip()))}
                if titles and unit["unit_id"] not in columns and unit.get("block_label") in {"header", "paragraph_title"}:
                    # A heading spanning unassigned columns ends proven local scopes.
                    uncertain_scopes.update(columns.values())
                if len(titles) > 1:
                    uncertain_scopes.add(scope)
                if len(titles) == 1:
                    category = next(iter(titles))
                    uncertain_scopes.discard(scope)
                if len(levels) == 1:
                    grade = next(iter(levels))
                category_ambiguous |= scope in uncertain_scopes
                grade_ambiguous |= len(levels) > 1
            entries = []
            for line in lines:
                title = _category_title(line)
                level = GRADE_TITLE_PREFIX_RE.match(line.strip())
                if not spatial:
                    if title:
                        category = title
                    if level:
                        grade = level["grade"]
                entries.append((line, category, grade, category_ambiguous, grade_ambiguous))
            states[scope] = (category, grade)
            contexts[unit["unit_id"]] = entries
    return contexts


def item_context(item, contexts):
    """Only quoted rows/cells and their local headings may supply a default."""
    categories, grades = set(), set()
    category_ambiguous = grade_ambiguous = False
    for evidence in item["evidence"]:
        quote = evidence["quote"]
        entries = [entry for entry in contexts[evidence["unit_id"]]
                   if entry[0].strip() and (entry[0] in quote or quote in entry[0])]
        product_entries = [entry for entry in entries if item["product_name_raw"]
                           and item["product_name_raw"] in entry[0]]
        if product_entries:
            entries = product_entries
        if not entries:
            category_ambiguous = grade_ambiguous = True
        for text, category, grade, category_uncertain, grade_uncertain in entries:
            category_ambiguous |= category_uncertain
            grade_ambiguous |= grade_uncertain
            direct = category_evidence(text)
            if direct and category and direct != {category}:
                category_ambiguous = True
            categories.update(direct or {category or "玛钢"})
            grades.add(grade)
    category_ambiguous |= len(categories) != 1
    grade_ambiguous |= len(grades) > 1
    return (next(iter(categories)) if len(categories) == 1 else None,
            next(iter(grades)) if len(grades) == 1 else None, category_ambiguous, grade_ambiguous)
