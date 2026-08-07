"""Pre-cache decoded video frames as tensors to eliminate runtime video decode.

Usage:
    uv run python scripts/cache_video_frames.py \\
        --data-root /path/to/data \\
        --cache-dir /path/to/cache \\
        --config-name config_v2 \\
        --workers 8

Each clip (16 frames x 224 x 224 x 3 uint8) is saved as a single .pt file.
Cache structure:
    {cache_dir}/{split}/{record_id}__{start}.pt  ->  Tensor[16, 3, 224, 224]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from marineworld.data.adapters import build_data_adapter
from marineworld.data.clips import build_clip_index
from marineworld.data.manifest import manifest_checksum
from marineworld.data.video import decode_video

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def load_config(config_name: str) -> DictConfig:
    """Load a Hydra-composed config without running the main entrypoint."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if not GlobalHydra.instance().is_initialized():
        initialize_config_dir(config_dir=str(CONFIGS_DIR.resolve()), version_base=None)
    return compose(config_name=config_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-cache decoded video frames")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root directory")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Output cache directory")
    parser.add_argument("--config-name", default="config_v2", help="Hydra config name")
    parser.add_argument(
        "--image-size", type=int, default=224, help="Resize frames to this size (0=skip resize)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Which splits to cache",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel decode workers")
    return parser.parse_args()


def _decode_one(args_tuple: tuple) -> tuple[str, bool, str]:
    """Worker function: decode one clip and save to disk. Designed for multiprocessing."""
    record_dict, frame_indices, split, cache_dir, image_size = args_tuple
    record_id = record_dict["id"]

    try:
        from marineworld.data.contracts import VideoRecord

        record = VideoRecord(**record_dict)
        frames = decode_video(record, tuple(frame_indices))

        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(np.array(frames))
        if frames.ndim == 4 and frames.shape[1] not in (1, 3):
            frames = frames.permute(0, 3, 1, 2)

        if image_size > 0 and frames.shape[-1] != image_size:
            frames = torch.nn.functional.interpolate(
                frames.float(), size=(image_size, image_size), mode="bilinear", align_corners=False
            ).to(torch.uint8)

        out_path = Path(cache_dir) / split / f"{record_id}__{frame_indices[0]}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(frames, out_path)
        return record_id, True, ""
    except Exception as exc:
        return record_id, False, str(exc)[:200]


def main() -> None:
    args = parse_args()

    print(f"Loading config {args.config_name} ...")
    cfg = load_config(args.config_name)

    print(f"Building manifest from {args.data_root} ...")
    adapter = build_data_adapter(cfg.data)
    manifest = adapter.build_manifest(args.data_root)
    records = {r.id: r for r in manifest.records}
    fp = manifest_checksum(manifest)

    print("Building clip index ...")
    clips_by_split: dict[str, list[tuple[str, list[int], float]]] = {}
    for split in ("train", "val"):
        clips = build_clip_index(
            manifest,
            split=split,
            frames=int(cfg.model.num_frames),
            stride=1,
            seed=int(cfg.seed),
            fingerprint=fp,
        )
        clips_by_split[split] = [
            (clip.record_id, list(clip.frame_indices), records[clip.record_id].fps)
            for clip in clips
        ]

    total_clips = sum(len(v) for v in clips_by_split.values())
    print(f"Total clips to cache: {total_clips}")

    for split, entries in clips_by_split.items():
        split_dir = args.cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        cached = {f.stem for f in split_dir.glob("*.pt")}
        to_cache = [(rid, fi) for rid, fi, _fps in entries if f"{rid}__{fi[0]}" not in cached]

        if not to_cache:
            print(f"  {split}: all {len(entries)} clips already cached")
            continue

        print(f"  {split}: {len(to_cache)} to cache ({len(entries) - len(to_cache)} skipped)")

        work_items = [
            (asdict(records[rid]), fi, split, str(args.cache_dir), args.image_size)
            for rid, fi in to_cache
        ]

        t0 = time.monotonic()
        succeeded = 0
        failed = 0
        failed_ids: list[str] = []

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_decode_one, item): item for item in work_items}

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(f"  {split}", total=len(futures))

                for future in as_completed(futures):
                    record_id, ok, err = future.result()
                    if ok:
                        succeeded += 1
                    else:
                        failed += 1
                        if failed <= 10:
                            failed_ids.append(record_id)
                    progress.advance(task)

        elapsed = time.monotonic() - t0
        rate = succeeded / elapsed if elapsed > 0 else 0
        print(f"  {split}: {succeeded} cached, {failed} failed ({rate:.1f} clips/sec)")
        if failed_ids:
            print(f"    Failed: {', '.join(failed_ids[:10])}")

    total_cached = sum(1 for _ in args.cache_dir.rglob("*.pt"))
    total_size_mb = sum(f.stat().st_size for f in args.cache_dir.rglob("*.pt")) / 1e6
    print(f"\nDone: {total_cached} clips, {total_size_mb:.0f} MB in {args.cache_dir}")


if __name__ == "__main__":
    main()
