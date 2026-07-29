"""Offline preprocessing of source clips into decode-friendly training clips.

The FVessel Clip-10 clips are 2560x1440 long-GOP H.264. Random-access seeks --
which the clip sampler makes for every training window -- fall back to a full
sequential decode of 2K frames, slow even though the streams are not damaged
(decord cannot seek between distant keyframes, so it errors and PyAV walks the
stream instead). Preprocessing once to a downscaled, short-GOP clip caps that
walk at one GOP and lets random seeks succeed, so training reads the real
footage at full speed.

Today the preprocessing step is a downscale + re-encode (`PreprocessSpec`); it is
the natural home for any further offline preparation (colour, cropping, ...) later.

Preprocessed clips are frames only -- no ``ais/`` or ``gt/`` -- so self-supervised
pretraining points ``data.root`` at the preprocessed root, while any eval/probe
path that needs AIS or targets must keep resolving them against the source root.
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

__all__ = ["PreprocessSpec", "build_preprocessed", "preprocess_clip", "app"]

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Preprocess source clips into decode-friendly training clips."""


@dataclass(frozen=True)
class PreprocessSpec:
    """Encoding for a preprocessed clip: small frames and frequent keyframes.

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


def preprocess_clip(src: Path, dst: Path, spec: PreprocessSpec) -> None:
    """Preprocess one clip, writing via a temp file renamed on success.

    ffmpeg writes a sibling ``.tmp`` that is ``os.replace``-d into place only when
    it exits cleanly, so a file at the final path always means a complete clip.
    An interrupt or kill mid-run leaves only the temp file, which the existence-based
    skip in ``build_preprocessed`` ignores -- no truncated clip is mistaken for done.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(f".tmp{dst.suffix}")  # keep the real extension so ffmpeg picks the muxer
    result = subprocess.run(spec.ffmpeg_command(src, tmp), capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {src}: {result.stderr.strip()}")
    os.replace(tmp, dst)


def build_preprocessed(
    source_root: Path,
    out_root: Path,
    clips: Iterable[Path],
    *,
    spec: PreprocessSpec = PreprocessSpec(),
    preprocess: Callable[[Path, Path, PreprocessSpec], None] = preprocess_clip,
    force: bool = False,
) -> list[Path]:
    """Mirror each clip into out_root at its source-relative path, skipping done ones."""
    pending = sorted(clips)
    written: list[Path] = []
    with Progress() as progress:
        task = progress.add_task("preprocessing", total=len(pending))
        for src in pending:
            dst = out_root / src.relative_to(source_root)
            # Existence means done (preprocess_clip renames atomically); a changed
            # source is not detected, which is safe for the download-once corpus.
            # Use --force to rebuild after editing sources.
            if force or not dst.exists():
                preprocess(src, dst, spec)
                written.append(dst)
            progress.advance(task)
    return written


@app.command("fvessel")
def fvessel(
    source_root: Path = typer.Argument(Path("data/raw/fvessel"), help="FVessel source root."),
    out_root: Path = typer.Argument(
        Path("data/preprocessed/fvessel"), help="Preprocessed destination root."
    ),
    height: int = typer.Option(256, help="Frame height in px (width keeps aspect)."),
    gop: int = typer.Option(10, help="Keyframe interval; smaller means faster random seeks."),
    crf: int = typer.Option(23, help="libx264 quality (lower is better and larger)."),
    force: bool = typer.Option(
        False, help="Re-run clips whose preprocessed output already exists."
    ),
) -> None:
    """Preprocess FVessel source clips (excluding gt/ overlays) into training clips."""
    clips = FVesselAdapter.source_clips(source_root)
    written = build_preprocessed(
        source_root,
        out_root,
        clips,
        spec=PreprocessSpec(height=height, gop=gop, crf=crf),
        force=force,
    )
    typer.echo(f"wrote {len(written)} preprocessed clip(s) under {out_root}")


if __name__ == "__main__":
    app()
