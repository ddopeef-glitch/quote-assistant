"""Temporary local inquiry evidence: originals, metadata and append-only events."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import stat
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = ROOT / "data" / "usage_diagnostics"


def now():
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, (Decimal, Path)):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    # Do not serialize arbitrary object reprs, which may contain credentials.
    raise TypeError(f"unsupported diagnostic value: {type(value).__name__}")


def code_fingerprint(root=ROOT):
    paths = [root / name for name in ("app.py", "config.py", "VERSION", "requirements.txt")]
    paths += list((root / "src").rglob("*.py"))
    paths += list((root / "benchmarks" / "prompts").glob("*.txt"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _check_path(path, root):
    """Reject all reparse points, including Windows junctions, before reading."""
    path.relative_to(root)
    for item in [*(root / parent for parent in reversed(path.relative_to(root).parents)), path]:
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("诊断目录含链接，已拒绝访问")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("诊断路径越界")


class Diagnostics:
    def __init__(self, *, root=None):
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex
        self.root = Path(root or DIAGNOSTICS_DIR)
        self.path = self.root / self.id
        self.errors = []
        self.ready = False
        self._last = {}
        self._secrets = tuple(value for value in [os.environ.get("DEEPSEEK_API_KEY", "")] if value)
        self.revision = 0

    def clean(self, text):
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        return re.sub(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?|(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*)[^\s,;\"'}]+",
                      r"\1[REDACTED]", text)

    def failure(self, stage, error):
        message = self.clean(f"{stage}：{type(error).__name__}：{error}")
        if message not in self.errors:
            self.errors.append(message)
            self.revision += 1
            if self.ready:
                try:
                    self._append("diagnostic_failed", {"stage": stage, "error": message})
                except Exception:
                    pass  # The persistent page warning remains if the disk cannot be written.

    def _dumps(self, value):
        def scrub(item):
            if isinstance(item, str):
                return self.clean(item)
            if isinstance(item, dict):
                return {self.clean(str(key)): "[REDACTED]" if str(key).casefold() in
                        {"authorization", "api_key", "apikey", "access_token", "password", "deepseek_api_key"}
                        else scrub(child) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                return [scrub(child) for child in item]
            return item
        return json.dumps(scrub(value), ensure_ascii=False, allow_nan=False, default=_json_default)

    def start(self, files, text, *, version, master_fingerprint):
        try:
            parent = next(parent for parent in self.root.parents if parent.exists())
            _check_path(parent, parent)
            if self.root.exists():
                _check_path(self.root, parent)
            self.root.mkdir(parents=True, exist_ok=True)
            _check_path(self.root, self.root.parent)
            self.path.mkdir(exist_ok=False)
            (self.path / "events.jsonl").touch(exist_ok=False)
            self.ready = True
        except Exception as error:
            self.failure("初始化", error)
            return
        self.event("recognition_started", model_root="model", input_count=len(files), pasted_text=bool(text))
        originals = []
        inputs = [("粘贴文字.txt", text.encode("utf-8")), *files]
        for index, (name, data) in enumerate(inputs):
            suffix = Path(name).suffix.lower()
            suffix = suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) else ".bin"
            stored = f"input-{index:03d}{suffix}"
            row = {"source_id": f"source-{len(files) + 1}" if index == 0 and text.strip() else f"source-{index}" if index else None,
                   "kind": "paste" if index == 0 else "upload", "original_name": name,
                   "file": stored, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "saved": False}
            try:
                # A runtime credential accidentally present in an input must not be archived.
                if any(secret.encode() in data for secret in self._secrets):
                    raise ValueError("原始输入含运行凭据，未保存该原件")
                with (self.path / stored).open("xb") as handle:
                    handle.write(data)
                row["saved"] = True
            except Exception as error:
                self.failure("原始输入", error)
            originals.append(row)
            self.event("input_saved", **row)
        fingerprint = None
        try:
            fingerprint = code_fingerprint()
        except Exception as error:
            self.failure("代码指纹", error)
        try:
            with (self.path / "meta.json").open("x", encoding="utf-8") as handle:
                handle.write(self._dumps({"schema_version": 1, "run_id": self.id, "created_at": now(),
                                         "software_version": version, "code_sha256": fingerprint,
                                         "master_fingerprint": master_fingerprint, "model_root": "model",
                                         "originals": originals, "confirmation_is": "user_confirmed_version"}) + "\n")
        except Exception as error:
            self.failure("元信息", error)

    def _append(self, event, data):
        _check_path(self.path / "events.jsonl", self.root)
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(self._dumps({"event": event, "at": now(), "data": data}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.revision += 1

    def event(self, event, *, dedupe=None, dedupe_value=None, **data):
        if not self.ready:
            return
        try:
            content = self._dumps(data if dedupe_value is None else dedupe_value)
            if dedupe is not None and self._last.get(dedupe) == content:
                return
            self._append(event, data)
            if dedupe is not None:
                self._last[dedupe] = content
        except Exception as error:
            self.failure(event, error)

    def callback(self, event, data):
        self.event(event, **data)

    def package(self):
        """Build on demand; never include quotation files or follow links."""
        def unreadable(error):
            raise error
        try:
            _check_path(self.path, self.root)
            output = BytesIO()
            with ZipFile(output, "w", ZIP_DEFLATED) as archive:
                for folder, directories, filenames in os.walk(self.path, followlinks=False, onerror=unreadable):
                    for name in directories + filenames:
                        _check_path(Path(folder) / name, self.root)
                    for name in filenames:
                        path = Path(folder) / name
                        if not stat.S_ISREG(path.lstat().st_mode):
                            raise ValueError("诊断目录含非常规文件")
                        archive.writestr(path.relative_to(self.path).as_posix(), path.read_bytes())
                archive.writestr("diagnostic_status.json", self._dumps({"run_id": self.id, "recording_errors": self.errors,
                                  "note": "仅含已收到的操作；最后事件不代表全流程完成。下载不表示已发送。"}))
            return output.getvalue()
        except Exception as error:
            self.failure("诊断包", error)
            return None


def emit_diagnostic(callback, event, **data):
    """Optional observers must not change input parsing or OCR outcomes."""
    if callback:
        try:
            callback(event, data)
        except Exception:
            # The page's Diagnostics callback reports its own persistence failures.
            pass
