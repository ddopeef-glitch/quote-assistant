from io import BytesIO
from types import SimpleNamespace
from pathlib import Path

import fitz
from PIL import Image, UnidentifiedImageError
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import input_router as router


def image_bytes(format_name="PNG", *, animated=False, exif=None):
    output = BytesIO()
    image = Image.new("RGB", (30, 20), "white")
    options = {"exif": exif} if exif else {}
    if animated:
        options.update(save_all=True, append_images=[Image.new("RGB", image.size, "black")])
    image.save(output, format_name, **options)
    return output.getvalue()


def test_image_validation_limits_decoder_and_pixels_and_preserves_orientation(monkeypatch):
    with pytest.raises(UnidentifiedImageError):
        router.validate_image(image_bytes("TIFF"))
    exif = Image.Exif()
    exif[274] = 6
    normalized = router.validate_image(image_bytes("JPEG", exif=exif))
    with Image.open(BytesIO(normalized)) as image:
        assert image.size == (20, 30) and image.format == "JPEG"
        assert not image.getexif()
    monkeypatch.setattr(router, "MAX_IMAGE_PIXELS", 599)
    with pytest.raises(ValueError, match="像素"):
        router.validate_image(image_bytes())


@pytest.mark.parametrize("kind", ["broken", "static", "animated", "renamed-tiff"])
def test_upload_preview_handles_bad_images_without_ocr_or_api(monkeypatch, kind):
    from src.ocr_pipeline import LocalOCRExtractor
    from src import agent_inquiry

    def forbidden(*args, **kwargs):
        pytest.fail("Preview must not run OCR or the API")

    content = (b"broken image" if kind == "broken" else
               image_bytes("TIFF" if kind == "renamed-tiff" else "PNG", animated=kind == "animated"))
    upload = SimpleNamespace(name="sample.png", type="application/octet-stream", getvalue=lambda: content)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: [upload])
    monkeypatch.setattr(LocalOCRExtractor, "extract_raw", forbidden)
    monkeypatch.setattr(agent_inquiry, "recognize_inquiry", forbidden)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert bool(app.error) == (kind != "static")
    assert len(app.get("image")) == (1 if kind == "static" else 0)
    if app.error:
        assert "重新上传静态图片" in app.error[0].value


def pdf_bytes(*, pages=1, width=595, height=842):
    with fitz.open() as document:
        for _ in range(pages):
            document.new_page(width=width, height=height)
        return document.tobytes()


@pytest.mark.parametrize("legacy", [False, True])
def test_pdf_bounds_are_checked_before_raster_allocation(monkeypatch, legacy):
    calls = []

    def raster(page, *, matrix, colorspace, alpha):
        assert page.rect.width * matrix.a <= router.MAX_IMAGE_SIDE
        assert page.rect.height * matrix.d <= router.MAX_IMAGE_SIDE
        assert colorspace == fitz.csRGB and not alpha
        calls.append(matrix)
        return SimpleNamespace(tobytes=lambda _: image_bytes())

    monkeypatch.setattr(fitz.Page, "get_pixmap", raster)
    raw = [{"res": {"parsing_res_list": [{"block_content": "synthetic", "block_label": "text"}]}}]
    ocr = SimpleNamespace(extract_raw=lambda _: raw, extract_products=lambda *args: [])

    def extract(data):
        return (router.extract_file("scan.pdf", data, ai_vision=ocr) if legacy
                else router.extract_evidence_units("scan.pdf", data, ocr))

    extract(pdf_bytes(width=14400, height=14400))
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(ValueError, match="超过.*页"):
        extract(pdf_bytes(pages=router.MAX_PDF_PAGES + 1))
    assert not calls
    monkeypatch.setattr(router, "MAX_PDF_PIXELS", 2400 * 2400)
    with pytest.raises(ValueError, match="累计渲染像素"):
        extract(pdf_bytes(pages=2, width=14400, height=14400))
    assert len(calls) == 1  # The second page is rejected before allocation.


def test_pdf_without_ocr_never_renders(monkeypatch):
    monkeypatch.setattr(fitz.Page, "get_pixmap", lambda *args, **kwargs: pytest.fail("unneeded raster"))
    data = pdf_bytes()
    with pytest.raises(ValueError, match="缺少本地 OCR"):
        router.extract_evidence_units("scan.pdf", data)
    assert router.extract_file("scan.pdf", data) == []
