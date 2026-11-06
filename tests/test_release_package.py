import hashlib
import json
from zipfile import ZipFile

import pytest

from scripts.package_release import checked_files, package


def manifest(root, name, content):
    (root / "public-manifest.json").write_text(json.dumps({name: hashlib.sha256(content).hexdigest()}))


def test_package_excludes_unlisted_business_files_and_refuses_overwrite(tmp_path):
    (tmp_path / "README.md").write_bytes(b"synthetic demo")
    (tmp_path / "private.xlsx").write_bytes(b"not for publication")
    manifest(tmp_path, "README.md", b"synthetic demo")
    output = tmp_path / "release.zip"
    assert package(output, tmp_path) == 2
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {"quote-assistant/README.md", "quote-assistant/public-manifest.json"}
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        package(output, tmp_path)
    assert output.read_bytes() == before


def test_changed_or_credential_bearing_source_is_blocked(tmp_path):
    path = tmp_path / "app.py"
    path.write_bytes(b"changed")
    manifest(tmp_path, "app.py", b"reviewed")
    with pytest.raises(ValueError, match="Changed since review"):
        checked_files(tmp_path)
    synthetic_key = b"sk-" + b"A" * 32
    path.write_bytes(synthetic_key)
    manifest(tmp_path, "app.py", synthetic_key)
    with pytest.raises(ValueError, match="Possible credential"):
        checked_files(tmp_path)


@pytest.mark.parametrize("name", ["../outside.txt", "data/input.txt", "DATA/customer.txt", "Runtime/trace.txt",
                                 "主数据/a.xlsx", "C:/secret.txt", ".env", ".ENV.production",
                                 ".streamlit/secrets.toml", ".Streamlit/Secrets.TOML"])
def test_manifest_cannot_allow_private_or_outside_paths(tmp_path, name):
    manifest(tmp_path, name, b"private")
    with pytest.raises(ValueError, match="Forbidden release path"):
        checked_files(tmp_path)


@pytest.mark.parametrize("name", [".env.example", "src/example.py"])
def test_safe_documentation_and_source_remain_allowed(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"example")
    manifest(tmp_path, name, b"example")
    assert checked_files(tmp_path)[name] == b"example"
