"""Validate an explicitly reviewed manifest; package only its exact bytes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "public-manifest.json"
BLOCKED_PARTS = {".git", ".venv", "runtime", "data", "主数据", "模板", "测试样本", "导出模板", "交付材料"}
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def read_regular(root, name):
    relative = PurePosixPath(name)
    parts = [part.casefold() for part in relative.parts]
    if (not name or "\\" in name or relative.is_absolute() or ":" in name
            or any(part in {".", ".."} for part in name.split("/"))
            or any(part in BLOCKED_PARTS for part in parts)
            or any(part != ".env.example" and (part == ".env" or part.startswith(".env.")) for part in parts)
            or any(parent == ".streamlit" and child == "secrets.toml" for parent, child in zip(parts, parts[1:]))):
        raise ValueError("Forbidden release path")
    target = root
    for part in relative.parts:
        target /= part
        info = target.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("Links are forbidden in a release")
    if not target.is_file() or info.st_size > 2_000_000:
        raise ValueError("Release entry must be a regular file below 2 MB")
    return target.read_bytes()


def checked_files(root=ROOT):
    root = Path(root).resolve()
    raw = read_regular(root, MANIFEST)
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Empty or invalid release manifest")
    if len({name.casefold() for name in manifest}) != len(manifest) or MANIFEST in manifest:
        raise ValueError("Duplicate or recursive release entry")
    contents = {}
    for name, expected in manifest.items():
        data = read_regular(root, name)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Changed since review: {name}")
        if Path(name).suffix.lower() in {".xlsx", ".xls", ".pdf", ".zip", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"Business/binary attachment is forbidden: {name}")
        if Path(name).suffix.lower() == ".png" and not name.startswith("docs/assets/"):
            raise ValueError("Only reviewed documentation PNGs are allowed")
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Possible credential in {name}; matched value withheld")
        contents[name] = data
    # Never inherit a parent repository's history or publish extra tracked files.
    if (root / ".git").exists():
        tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
        if set(filter(None, tracked)) - set(contents) - {MANIFEST}:
            raise ValueError("Git tracks files outside the reviewed release manifest")
    contents[MANIFEST] = raw
    return contents


def package(destination, root=ROOT):
    contents = checked_files(root)
    # Exclusive creation refuses to replace any previous archive.
    with Path(destination).open("xb") as handle, ZipFile(handle, "w", ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            archive.writestr("quote-assistant/" + name, data)
    return len(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Create a new source-only ZIP; omit to check")
    args = parser.parse_args()
    try:
        count = package(args.output) if args.output else len(checked_files())
        print(f"Verified {count} reviewed files. No Git history or runtime data included.")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release blocked: {error}\n")


if __name__ == "__main__":
    main()
