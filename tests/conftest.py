import pytest
import socket
import sys


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    original_connect = socket.socket.connect

    def connect(sock, address):
        # Windows asyncio uses the stdlib socketpair's private loopback pipe.
        if (sys.platform == "win32" and sys._getframe(1).f_code is socket.socketpair.__code__
                and address[0] in {"127.0.0.1", "::1"}):
            return original_connect(sock, address)
        raise AssertionError("Tests must not open network connections")

    def reject(*args, **kwargs):
        raise AssertionError("Tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)


@pytest.fixture(autouse=True)
def isolate_usage_diagnostics(tmp_path, monkeypatch):
    from src import usage_diagnostics
    monkeypatch.setattr(usage_diagnostics, "DIAGNOSTICS_DIR", tmp_path / "usage_diagnostics")
