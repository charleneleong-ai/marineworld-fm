"""Safe acquisition commands for public maritime datasets."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath

import httpx
import typer

FVESSEL_CLIP10_URL = (
    "https://huggingface.co/datasets/gy65896/FVessel/resolve/main/Clip-10.zip?download=true"
)

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Download and validate public maritime datasets."""


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP only when every member remains under the destination."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        bad_member = handle.testzip()
        if bad_member is not None:
            raise ValueError(f"corrupt archive member: {bad_member}")
        for member in handle.infolist():
            normalized = PurePosixPath(member.filename.replace("\\", "/"))
            if (
                not normalized.parts
                or normalized.is_absolute()
                or ".." in normalized.parts
                or ":" in normalized.parts[0]
            ):
                raise ValueError(f"unsafe archive member: {member.filename}")
            target = (destination / Path(*normalized.parts)).resolve()
            try:
                target.relative_to(destination)
            except ValueError as error:
                raise ValueError(f"unsafe archive member: {member.filename}") from error
        handle.extractall(destination)


def download_fvessel_clip10(output: Path) -> Path:
    """Download, validate, and safely extract the official FVessel Clip-10 subset."""
    output = output.resolve()
    existing_videos = tuple(output.rglob("*.mp4")) if output.exists() else ()
    if existing_videos:
        return output
    if output.exists() and any(output.iterdir()):
        raise ValueError("FVessel output must be empty or contain an existing MP4 extraction")

    output.mkdir(parents=True, exist_ok=True)
    archive = output.parent / "Clip-10.zip"
    partial = archive.with_suffix(".zip.part")
    try:
        with httpx.stream(
            "GET", FVESSEL_CLIP10_URL, follow_redirects=True, timeout=None
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        with zipfile.ZipFile(partial) as handle:
            bad_member = handle.testzip()
            if bad_member is not None:
                raise ValueError(f"corrupt archive member: {bad_member}")
        partial.replace(archive)
        safe_extract_zip(archive, output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if not any(output.rglob("*.mp4")):
        shutil.rmtree(output)
        raise ValueError("FVessel Clip-10 archive contains no MP4 files")
    return output


@app.command("fvessel-clip10")
def fvessel_clip10(
    output: Path = typer.Option(Path("data/raw/fvessel"), help="Extraction destination."),
) -> None:
    """Download the 2.56 GB official FVessel Clip-10 archive."""
    destination = download_fvessel_clip10(output)
    typer.echo(f"FVessel Clip-10 ready at {destination}")


if __name__ == "__main__":
    app()
