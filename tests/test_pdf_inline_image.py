from types import SimpleNamespace

import fitz

from src.input_router import extract_evidence_units


def _pdf_with_inline_image():
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((30, 40), "inline image text")
        content_xref = page.get_contents()[0]
        content = document.xref_stream(content_xref)
        inline_image = b"BI /W 1 /H 1 /CS /RGB /BPC 8 ID " + bytes((255, 0, 0)) + b" EI"
        document.update_stream(
            content_xref,
            content + b"\nq 20 0 0 20 50 50 cm " + inline_image + b" Q\n",
        )
        return document.tobytes()


def test_pdf_inline_image_keeps_text_and_reads_image_once():
    calls = []
    raw = [{"res": {"parsing_res_list": [{"block_label": "text", "block_content": "图片文字"}]}}]
    ocr = SimpleNamespace(extract_raw=lambda data: calls.append(data) or raw)

    units = extract_evidence_units("inline.pdf", _pdf_with_inline_image(), ocr)

    assert len(calls) == 1
    assert any(unit["reading_method"] == "pdf_text" and "inline image text" in unit["text"] for unit in units)
    assert [unit["text"] for unit in units if unit["reading_method"] == "local_ocr"] == ["图片文字"]
