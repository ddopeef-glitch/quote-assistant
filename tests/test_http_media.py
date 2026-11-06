"""Exercise the HTTP layer that AppTest does not cover, without network I/O."""
import gzip
import asyncio
import socket

import pytest
from streamlit.web.server.starlette.starlette_gzip_middleware import SelectiveGZipMiddleware


def test_offline_guard_allows_event_loop_pipe_but_blocks_network_clients():
    loop = asyncio.new_event_loop()
    loop.close()
    for address in ("127.0.0.1", "192.0.2.1"):
        for method in ("connect", "connect_ex"):
            with socket.socket() as client, pytest.raises(AssertionError, match="network"):
                getattr(client, method)((address, 80))


@pytest.mark.parametrize("encoding", [b"gzip", b"identity"])
@pytest.mark.parametrize("content_type", [b"image/png", b"application/pdf"])
def test_media_http_response_starts_before_body(encoding, content_type):
    payload = b"synthetic-media-fixture" * 100
    messages = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", content_type), (b"content-length", str(len(payload)).encode())]})
        await send({"type": "http.response.body", "body": payload})

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "path": "/media/demo.png", "headers": [(b"accept-encoding", encoding)]}
    # These in-memory ASGI callbacks do not suspend or require an event loop.
    response = SelectiveGZipMiddleware(app)(scope, receive, send)
    with pytest.raises(StopIteration):
        response.send(None)
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 200
    body = b"".join(message.get("body", b"") for message in messages[1:])
    if dict(messages[0]["headers"]).get(b"content-encoding") == b"gzip":
        body = gzip.decompress(body)
    assert body == payload
