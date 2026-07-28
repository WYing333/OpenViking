from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.parse.parsers.media import utils as media_utils


class _MemoryFS:
    def __init__(self, content: bytes):
        self.content = content
        self.read_calls = 0

    async def stat(self, _path: str, ctx=None):
        return {"size": len(self.content)}

    async def read(self, _path: str, offset: int, size: int, ctx=None):
        self.read_calls += 1
        return self.content[offset : offset + size]


class _MediaVLM:
    model = "media-vlm"

    def __init__(self, supported: bool = True):
        self.supported = supported
        self.support_calls = []
        self.completion_calls = []

    def supports_media(self, *, media_type: str, filename: str, size_bytes: int):
        self.support_calls.append((media_type, filename, size_bytes))
        return self.supported

    async def get_media_completion_async(
        self,
        prompt: str,
        media_path: str,
        filename: str,
        media_type: str,
    ):
        with open(media_path, "rb") as file:
            content = file.read()
        self.completion_calls.append((prompt, filename, media_type, content))
        return "# Audio summary\n\nA concise summary of the recording."


@pytest.mark.asyncio
async def test_audio_summary_routes_through_vlm(monkeypatch):
    fs = _MemoryFS(b"audio bytes")
    vlm = _MediaVLM()
    config = SimpleNamespace(
        vlm=vlm,
        semantic=SimpleNamespace(
            overview_max_chars=1000,
            abstract_max_chars=200,
        ),
        output_language_override=None,
    )
    monkeypatch.setattr(media_utils, "get_openviking_config", lambda: config)
    monkeypatch.setattr(media_utils, "get_viking_fs", lambda: fs)
    monkeypatch.setattr(media_utils, "render_prompt", lambda *_args, **_kwargs: "prompt")

    summary = await media_utils.generate_audio_summary(
        "/audio/sample.mp3",
        "sample.mp3",
        llm_sem=None,
    )

    assert summary["summary"]
    assert vlm.support_calls == [("audio", "sample.mp3", len(b"audio bytes"))]
    assert len(vlm.completion_calls) == 1
    assert vlm.completion_calls[0][1:] == (
        "sample.mp3",
        "audio",
        b"audio bytes",
    )


@pytest.mark.asyncio
async def test_unsupported_media_skips_file_read(monkeypatch):
    fs = _MemoryFS(b"audio bytes")
    vlm = _MediaVLM(supported=False)
    config = SimpleNamespace(
        vlm=vlm,
        semantic=SimpleNamespace(
            overview_max_chars=1000,
            abstract_max_chars=200,
        ),
        output_language_override=None,
    )
    monkeypatch.setattr(media_utils, "get_openviking_config", lambda: config)
    monkeypatch.setattr(media_utils, "get_viking_fs", lambda: fs)
    monkeypatch.setattr(media_utils, "render_prompt", lambda *_args, **_kwargs: "prompt")

    summary = await media_utils.generate_audio_summary(
        "/audio/sample.mp3",
        "sample.mp3",
        llm_sem=None,
    )

    assert summary["summary"] == ""
    assert fs.read_calls == 0
    assert vlm.completion_calls == []


@pytest.mark.asyncio
async def test_local_tempfile_cleanup_error_does_not_replace_success(monkeypatch, caplog):
    fs = _MemoryFS(b"audio bytes")
    vlm = _MediaVLM()
    config = SimpleNamespace(
        vlm=vlm,
        semantic=SimpleNamespace(
            overview_max_chars=1000,
            abstract_max_chars=200,
        ),
        output_language_override=None,
    )
    monkeypatch.setattr(media_utils, "get_openviking_config", lambda: config)
    monkeypatch.setattr(media_utils, "get_viking_fs", lambda: fs)
    monkeypatch.setattr(media_utils, "render_prompt", lambda *_args, **_kwargs: "prompt")

    def fail_unlink(_path, *, missing_ok=False):
        del missing_ok
        raise OSError("sensitive temporary path")

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    summary = await media_utils.generate_audio_summary(
        "/audio/sample.mp3",
        "sample.mp3",
        llm_sem=None,
    )

    assert summary["summary"]
    assert "sensitive temporary path" not in caplog.text
