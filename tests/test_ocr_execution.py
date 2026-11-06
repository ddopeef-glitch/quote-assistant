"""Offline checks for Paddle mode, cached pipelines and shared inference access."""
from concurrent.futures import ThreadPoolExecutor
import ast
from contextlib import contextmanager
from io import BytesIO
import json
import gc
from pathlib import Path
import sys
from threading import Barrier, local
import time
import weakref
from types import SimpleNamespace

from PIL import Image
import pytest

from config import MASTER_DATA_FILE
from src import ocr_pipeline


def test_changing_master_fingerprint_releases_evicted_ocr_models():
    import streamlit as st

    class Pipeline:
        pass

    class Extractor:
        def __init__(self, path):
            self._pipeline = Pipeline()

    tree = ast.parse((MASTER_DATA_FILE.parents[1] / "app.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "local_ocr_extractor")
    namespace = {"st": st, "Path": Path, "LocalOCRExtractor": Extractor}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
    cached = namespace["local_ocr_extractor"]
    cached.clear()
    references = []
    try:
        for fingerprint in range(5):
            extractor = cached(str(MASTER_DATA_FILE), str(fingerprint))
            assert cached(str(MASTER_DATA_FILE), str(fingerprint)) is extractor
            references.append(weakref.ref(extractor._pipeline))
            del extractor
            gc.collect()
            assert sum(reference() is not None for reference in references) == 1
    finally:
        cached.clear()
    gc.collect()
    assert all(reference() is None for reference in references)


def test_cached_ocr_stays_dynamic_and_serial_across_threads(monkeypatch):
    state = local()
    constructors = []
    active = []
    calls = []
    fail_next = [True]

    @contextmanager
    def guard(device):
        assert device == "cpu"
        assert not getattr(state, "dynamic", False)
        assert not active, "Paddle mode contexts must not overlap across instances"
        active.append(True)
        state.dynamic = True
        try:
            time.sleep(0.005)  # Let another caller contend for the shared guard.
            yield
        finally:
            state.dynamic = False
            active.pop()

    class Pipeline:
        def __init__(self, **kwargs):
            assert state.dynamic
            assert kwargs["use_queues"] is False
            assert kwargs["device"] == "cpu"
            constructors.append(self)

        def predict(self, path):
            assert state.dynamic and Path(path).is_file()
            calls.append(self)
            time.sleep(0.005)
            if fail_next[0]:
                fail_next[0] = False
                raise RuntimeError("synthetic OCR failure")
            yield SimpleNamespace(json=json.dumps({"res": {"parsing_res_list": []}}))
            assert state.dynamic, "The generator must be consumed inside the guard"
            yield SimpleNamespace(json={"res": {"parsing_res_list": []}})

    monkeypatch.setattr(ocr_pipeline, "_prepare_paddle_cache", lambda _: None)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=Pipeline))
    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(
        base=SimpleNamespace(dygraph=SimpleNamespace(guard=guard)),
    ))
    image = BytesIO()
    Image.new("RGB", (20, 20), "white").save(image, "PNG")
    extractors = [ocr_pipeline.LocalOCRExtractor(MASTER_DATA_FILE, device="cpu") for _ in range(2)]

    with pytest.raises(RuntimeError, match="synthetic OCR failure"):
        extractors[0].extract_raw(image.getvalue())
    assert not state.dynamic and not active

    def extract(extractor, barrier=None):
        if barrier is not None:
            barrier.wait(timeout=5)
        result = extractor.extract_raw(image.getvalue())
        assert not state.dynamic
        return result

    # A fresh script thread reuses the pipeline even after a failed invocation.
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert len(pool.submit(extract, extractors[0]).result(timeout=5)) == 2

    barrier = Barrier(3)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(extract, extractor, barrier) for extractor in (*extractors, extractors[1])]
        assert all(len(job.result(timeout=5)) == 2 for job in jobs)
    assert len(constructors) == 2 and len(calls) == 5
    assert not active
