"""Transcode source clips into decode-friendly training proxies.

The FVessel Clip-10 clips are 2560x1440 long-GOP H.264. Random-access seeks --
which the clip sampler makes for every training window -- fall back to a full
sequential decode of 2K frames, slow even though the streams are not damaged
(decord cannot seek between distant keyframes, so it errors and PyAV walks the
stream instead). Transcoding once to a downscaled, short-GOP proxy caps that
walk at one GOP and lets random seeks succeed, so training reads the real
footage at full speed.

Proxies are frames only -- no ``ais/`` or ``gt/`` -- so self-supervised
pretraining points ``data.root`` at the proxy root, while any eval/probe path
that needs AIS or targets must keep resolving them against the source root.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.progress import Progress

from marineworld.data.fvessel import FVesselAdapter

__all__ = ["ProxySpec", "build_proxies", "transcode_clip", "app"]

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Build decode-friendly training proxies from source clips."""


@dataclass(frozen=True)
class ProxySpec:
    """Encoding for a training proxy: small frames and frequent keyframes.

    The short GOP is the point -- it caps the sequential decode any random-window
    seek must do, which is exactly what makes the 2K long-GOP source slow to sample.
    """

    height: int = 256
    gop: int = 10
    crf: int = 23
    preset: str = "veryfast"

    def ffmpeg_command(self, src: Path, dst: Path) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            # fix height, keep aspect, force even width (-2); assumes landscape
            # (source is 2560x1440) so height is the short side and stays >= the 224 crop
            "-vf",
            f"scale=-2:{self.height}",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            # pin the keyframe cadence so scene-cut detection cannot lengthen the GOP
            "-g",
            str(self.gop),
            "-keyint_min",
            str(self.gop),
            "-sc_threshold",
            "0",
            "-an",
            "-sn",  # training reads frames only; drop audio and subtitles
            str(dst),
        ]


def transcode_clip(src: Path, dst: Path, spec: ProxySpec) -> None:
    """Transcode one clip to its proxy, writing via a temp file renamed on success.

    ffmpeg writes a sibling ``.tmp`` that is ``os.replace``-d into place only when
    it exits cleanly, so a file at the final path always means a complete proxy.
    An interrupt or kill mid-transcode leaves only the temp file, which the
    existence-based skip in ``build_proxies`` ignores -- no truncated proxy is
    ever mistaken for done.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(f".tmp{dst.suffix}")  # keep the real extension so ffmpeg picks the muxer
    result = subprocess.run(spec.ffmpeg_command(src, tmp), capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {src}: {result.stderr.strip()}")
    os.replace(tmp, dst)


def build_proxies(
    source_root: Path,
    proxy_root: Path,
    clips: Iterable[Path],
    *,
    spec: ProxySpec = ProxySpec(),
    transcode: Callable[[Path, Path, ProxySpec], None] = transcode_clip,
    force: bool = False,
) -> list[Path]:
    """Mirror each clip into proxy_root at its source-relative path, skipping done ones."""
    pending = sorted(clips)
    written: list[Path] = []
    with Progress() as progress:
        task = progress.add_task("transcoding", total=len(pending))
        for src in pending:
            dst = proxy_root / src.relative_to(source_root)
            # Existence means done (transcode_clip renames atomically); a changed
            # source is not detected, which is safe for the download-once corpus.
            # Use --force to rebuild after editing sources.
            if force or not dst.exists():
                transcode(src, dst, spec)
                written.append(dst)
            progress.advance(task)
    return written


@app.command("fvessel")
def fvessel(
    source_root: Path = typer.Argument(Path("data/raw/fvessel"), help="FVessel source root."),
    proxy_root: Path = typer.Argument(Path("data/proxy/fvessel"), help="Proxy destination root."),
    height: int = typer.Option(256, help="Proxy frame height in px (width keeps aspect)."),
    gop: int = typer.Option(10, help="Keyframe interval; smaller means faster random seeks."),
    crf: int = typer.Option(23, help="libx264 quality (lower is better and larger)."),
    force: bool = typer.Option(False, help="Re-transcode clips whose proxy already exists."),
) -> None:
    """Transcode FVessel source clips (excluding gt/ overlays) into training proxies."""
    clips = FVesselAdapter.source_clips(source_root)
    written = build_proxies(
        source_root,
        proxy_root,
        clips,
        spec=ProxySpec(height=height, gop=gop, crf=crf),
        force=force,
    )
    typer.echo(f"wrote {len(written)} proxy clip(s) under {proxy_root}")


if __name__ == "__main__":
    app()
