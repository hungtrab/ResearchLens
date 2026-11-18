"""Reader network boundaries; all publisher responses are simulated."""
from contextlib import contextmanager

import httpx
import pytest

from researchlens import pdf


URL = "https://arxiv.org/pdf/2301.12345"
CONTENT = b"%PDF-1.7\nexample fixture\n%%EOF"


class Response:
    def __init__(self, content=CONTENT, headers=None, status=200, chunks=None):
        self.headers = headers or {}
        self.status = status
        self.is_redirect = 300 <= status < 400
        self.chunks = [content] if chunks is None else chunks

    def raise_for_status(self):
        if self.status >= 400:
            raise httpx.HTTPStatusError("Publisher failure", request=httpx.Request("GET", URL), response=httpx.Response(self.status))

    def iter_bytes(self, chunk_size):
        yield from self.chunks


def publisher(monkeypatch, responses):
    state = {"urls": [], "options": None, "closed": False}
    class Client:
        def __init__(self, **kwargs):
            state["options"] = kwargs
        def __enter__(self):
            return self
        def __exit__(self, *args):
            state["closed"] = True
        @contextmanager
        def stream(self, method, url):
            assert method == "GET"
            state["urls"].append(url)
            response = responses[len(state["urls"]) - 1]
            if isinstance(response, Exception):
                raise response
            yield response
    monkeypatch.setattr(pdf.httpx, "Client", Client)
    return state


@pytest.mark.parametrize("url", [URL, "https://openreview.net/pdf?id=abc", "https://proceedings.mlr.press/v1/paper.pdf", "https://aclanthology.org/2020.paper.pdf", "https://arxiv.org:443/pdf/1234"])
def test_supported_public_repository_urls(url):
    assert pdf.supported_pdf(url)


@pytest.mark.parametrize("url", [None, "", 123, "http://arxiv.org/pdf/x", "https://127.0.0.1/x", "https://localhost/x", "https://169.254.169.254/latest/meta-data", "https://arxiv.org.evil.example/x", "https://evil.arxiv.org/x", "https://arxiv.org@evil.example/x", "https://user:pass@arxiv.org/x", "https://arxiv.org:444/x", "https://arxiv.org:bad/x", "https://arxiv.org/x#fragment", "https://arxiv.org/x\nInjected", "https://arxiv.org/x\tbad", "https://arxiv.org\\evil.example/x", "file:///etc/passwd", "https://[::1]/x", "https://arxiv.org/" + "x" * 2000])
def test_pdf_url_security_boundaries(url):
    assert not pdf.supported_pdf(url)


def test_unsupported_url_is_rejected_before_network(monkeypatch):
    state = publisher(monkeypatch, [])
    with pytest.raises(ValueError, match="cannot be embedded"):
        pdf.download_pdf("https://127.0.0.1/private")
    assert state["options"] is None and state["urls"] == []


def test_pdf_download_uses_bounded_no_proxy_no_auto_redirect_client(monkeypatch):
    state = publisher(monkeypatch, [Response()])
    assert pdf.download_pdf(URL) == CONTENT
    assert state["options"]["follow_redirects"] is False
    assert state["options"]["trust_env"] is False
    assert state["options"]["timeout"].connect == 8
    assert state["options"]["headers"]["Accept"] == "application/pdf"
    assert state["closed"]


def test_relative_and_supported_cross_repository_redirects(monkeypatch):
    state = publisher(monkeypatch, [Response(status=302, headers={"location": "/pdf/next"}), Response(status=301, headers={"location": "https://openreview.net/pdf?id=x"}), Response()])
    assert pdf.download_pdf(URL) == CONTENT
    assert state["urls"] == [URL, "https://arxiv.org/pdf/next", "https://openreview.net/pdf?id=x"]


@pytest.mark.parametrize("location", ["http://arxiv.org/pdf/x", "http://127.0.0.1/private", "https://attacker.example/a", "//169.254.169.254/secret", "https://user@arxiv.org/a"])
def test_redirects_cannot_escape_allowlist(monkeypatch, location):
    state = publisher(monkeypatch, [Response(status=302, headers={"location": location})])
    with pytest.raises(ValueError, match="redirects outside"):
        pdf.download_pdf(URL)
    assert state["urls"] == [URL] and state["closed"]


def test_redirect_loop_is_bounded(monkeypatch):
    state = publisher(monkeypatch, [Response(status=302, headers={"location": URL})] * 4)
    with pytest.raises(OSError, match="Too many"):
        pdf.download_pdf(URL)
    assert len(state["urls"]) == 4 and state["closed"]


def test_declared_and_streamed_size_limits(monkeypatch):
    monkeypatch.setattr(pdf, "MAX_PDF_BYTES", 12)
    publisher(monkeypatch, [Response(headers={"content-length": "13"})])
    with pytest.raises(ValueError, match="limit"):
        pdf.download_pdf(URL)
    publisher(monkeypatch, [Response(chunks=[b"%PDF-1.7", b"12345"])])
    with pytest.raises(ValueError, match="limit"):
        pdf.download_pdf(URL)
    publisher(monkeypatch, [Response(content=b"%PDF-1.7abcd")])
    assert pdf.download_pdf(URL) == b"%PDF-1.7abcd"


@pytest.mark.parametrize("body", [b"", b"<html>Sign in</html>", b"not a PDF", b"<!DOCTYPE html>%PDF-"])
def test_content_signature_rejects_non_pdf_even_with_pdf_mime(monkeypatch, body):
    publisher(monkeypatch, [Response(content=body, headers={"content-type": "application/pdf"})])
    with pytest.raises(OSError, match="did not return a PDF"):
        pdf.download_pdf(URL)


def test_wall_clock_deadline(monkeypatch):
    publisher(monkeypatch, [Response()])
    times = iter([0, 26])
    monkeypatch.setattr(pdf.time, "monotonic", lambda: next(times))
    with pytest.raises(OSError, match="too long"):
        pdf.download_pdf(URL)


@pytest.mark.parametrize("failure", [Response(status=404), Response(status=503), httpx.ConnectError("down"), httpx.ReadTimeout("slow")])
def test_upstream_errors_are_reader_errors(monkeypatch, failure):
    state = publisher(monkeypatch, [failure])
    with pytest.raises(OSError, match="unavailable"):
        pdf.download_pdf(URL)
    assert state["closed"]


def test_pdf_api_info_and_content(client, monkeypatch):
    client.app.state.engine.by_id["P1"]["pdf_url"] = URL
    info = client.get("/api/papers/P1/pdf-info").json()
    assert info["supported"] and info["url"] == URL
    publisher(monkeypatch, [Response()])
    response = client.get("/api/papers/P1/pdf")
    assert response.status_code == 200 and response.content == CONTENT
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_pdf_api_unavailable_unsupported_and_missing(client, monkeypatch):
    assert not client.get("/api/papers/P1/pdf-info").json()["supported"]
    assert client.get("/api/papers/P1/pdf").status_code == 422
    assert client.get("/api/papers/missing/pdf-info").status_code == 404
    assert client.get("/api/papers/missing/pdf").status_code == 404
    client.app.state.engine.by_id["P1"]["pdf_url"] = URL
    publisher(monkeypatch, [Response(status=503)])
    assert client.get("/api/papers/P1/pdf").status_code == 502


@pytest.mark.parametrize("url", [
    "https://openaccess.thecvf.com/content/CVPR2024/papers/paper.pdf",
    "https://proceedings.iclr.cc/paper_files/paper/2025/file/paper.pdf",
    "https://raw.githubusercontent.com/mlresearch/v235/main/assets/paper.pdf",
])
def test_official_discovery_pdf_sources_are_supported(url):
    assert pdf.supported_pdf(url)


@pytest.mark.parametrize("url", [
    "https://raw.githubusercontent.com/other-org/v235/main/paper.pdf",
    "https://raw.githubusercontent.com/mlresearch/v235/main/page.html",
    "https://raw.githubusercontent.com/mlresearch/v235/main/code.py",
    "https://raw.githubusercontent.com/mlresearch/../other-org/paper.pdf",
    "https://raw.githubusercontent.com/mlresearch/%2e%2e/other-org/paper.pdf",
])
def test_raw_github_reader_access_is_restricted_to_mlresearch_pdf_paths(url):
    assert not pdf.supported_pdf(url)
