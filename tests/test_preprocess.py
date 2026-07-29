"""Tests for decode-friendly clip preprocessing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from marineworld.data.preprocess import PreprocessSpec, build_preprocessed, preprocess_clip


def _touch_mp4(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")


def _noop_preprocess(src: Path, dst: Path, spec: PreprocessSpec) -> None:
    _touch_mp4(dst)


class TestPreprocessCommand:
    """The ffmpeg invocation encodes the small-frame, short-GOP contract."""

    def test_command_downscales_pins_gop_and_drops_audio(self, tmp_path: Path) -> None:
        cmd = PreprocessSpec(height=256, gop=10).ffmpeg_command(
            tmp_path / "a.mp4", tmp_path / "b.mp4"
        )
        assert "scale=-2:256" in cmd
        assert cmd[cmd.index("-g") + 1] == "10"
        assert cmd[cmd.index("-keyint_min") + 1] == "10"
        assert cmd[cmd.index("-sc_threshold") + 1] == "0"  # scene cuts cannot lengthen the GOP
        assert "-an" in cmd and "-sn" in cmd
        assert cmd[-1] == str(tmp_path / "b.mp4")


class TestBuildPreprocessed:
    """Walk source clips into a mirrored output tree, idempotently."""

    def test_mirrors_structure_and_skips_existing_on_a_second_run(self, tmp_path: Path) -> None:
        source_root, out_root = tmp_path / "src", tmp_path / "out"
        clips = [source_root / "Clip-10" / f"clip-{i}" / "v.mp4" for i in (1, 2)]
        for clip in clips:
            _touch_mp4(clip)

        written = build_preprocessed(source_root, out_root, clips, preprocess=_noop_preprocess)

        assert [dst.relative_to(out_root) for dst in written] == [
            clip.relative_to(source_root) for clip in clips
        ]
        assert build_preprocessed(source_root, out_root, clips, preprocess=_noop_preprocess) == []

    def test_force_reruns_existing_outputs(self, tmp_path: Path) -> None:
        source_root, out_root = tmp_path / "src", tmp_path / "out"
        clip = source_root / "v.mp4"
        _touch_mp4(clip)
        _touch_mp4(out_root / "v.mp4")

        assert build_preprocessed(source_root, out_root, [clip], preprocess=_noop_preprocess) == []
        forced = build_preprocessed(
            source_root, out_root, [clip], preprocess=_noop_preprocess, force=True
        )
        assert forced == [out_root / "v.mp4"]


def test_preprocess_clip_raises_and_removes_partial_output_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dst = tmp_path / "out" / "b.mp4"

    def failing_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"partial")  # ffmpeg's truncated temp output
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(subprocess, "run", failing_run)

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        preprocess_clip(tmp_path / "a.mp4", dst, PreprocessSpec())
    assert not dst.exists()  # no output at the final path
    assert not dst.with_suffix(f".tmp{dst.suffix}").exists()  # temp cleaned up too


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg binary not installed")
def test_preprocess_produces_a_downscaled_short_gop_clip(tmp_path: Path) -> None:
    import av

    src = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=640x480:rate=25",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
    )
    dst = tmp_path / "out.mp4"

    preprocess_clip(src, dst, PreprocessSpec(height=64, gop=5))

    with av.open(str(dst)) as container:
        stream = container.streams.video[0]
        height = stream.codec_context.height
        packets = [packet for packet in container.demux(stream) if packet.size > 0]
    keyframes = sum(1 for packet in packets if packet.is_keyframe)
    assert height == 64  # downscaled to the requested height
    # a keyframe at least every `gop` frames is what lets a random-window seek stay cheap
    assert keyframes * 5 >= len(packets) - 5
    assert not dst.with_suffix(f".tmp{dst.suffix}").exists()  # temp renamed away on success
