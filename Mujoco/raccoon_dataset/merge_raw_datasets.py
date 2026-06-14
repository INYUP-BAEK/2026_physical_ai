import argparse
import json
import os
import shutil
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def copy_episode(src_dir, dst_dir, copy_mode):
    if copy_mode == "hardlink":
        shutil.copytree(src_dir, dst_dir, copy_function=os.link)
    elif copy_mode == "copy":
        shutil.copytree(src_dir, dst_dir, copy_function=shutil.copy2)
    else:
        raise ValueError(f"unsupported copy_mode: {copy_mode}")


def iter_episode_dirs(root):
    return sorted(path for path in Path(root).glob("episode_*") if path.is_dir())


def max_numeric_scene_id(root):
    max_scene = -1
    for episode_dir in iter_episode_dirs(root):
        meta_path = episode_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = read_json(meta_path)
        except Exception:
            continue
        scene_id = meta.get("scene_id")
        if isinstance(scene_id, int):
            max_scene = max(max_scene, scene_id)
    return max_scene


def merge_raw_datasets(input_roots, out_root, copy_mode="hardlink", overwrite=False):
    out_root = Path(out_root)
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=False)

    merged = []
    next_episode_id = 1
    scene_offsets = []
    next_scene_offset = 0
    for input_root in input_roots:
        input_root = Path(input_root).resolve()
        scene_offsets.append(next_scene_offset)
        max_scene = max_numeric_scene_id(input_root)
        if max_scene >= 0:
            next_scene_offset += max_scene + 1

    for dataset_idx, input_root in enumerate(input_roots):
        input_root = Path(input_root).resolve()
        scene_offset = scene_offsets[dataset_idx]
        episode_dirs = iter_episode_dirs(input_root)
        if not episode_dirs:
            raise FileNotFoundError(f"No episode_* directories under {input_root}")

        for src_dir in episode_dirs:
            meta_path = src_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = read_json(meta_path)
            if not bool(meta.get("success", False)):
                continue

            dst_dir = out_root / f"episode_{next_episode_id:06d}"
            copy_episode(src_dir, dst_dir, copy_mode=copy_mode)

            dst_meta_path = dst_dir / "meta.json"
            dst_meta = read_json(dst_meta_path)
            # copytree(..., copy_function=os.link) hardlinks files, so replacing
            # meta.json must break the link before writing the renumbered metadata.
            dst_meta_path.unlink()
            dst_meta["episode_id_original"] = dst_meta.get("episode_id")
            dst_meta["episode_id"] = int(next_episode_id)
            original_scene_id = dst_meta.get("scene_id")
            dst_meta["scene_id_original"] = original_scene_id
            if isinstance(original_scene_id, int):
                dst_meta["scene_id"] = int(original_scene_id + scene_offset)
            dst_meta["merged_from_root"] = str(input_root)
            dst_meta["merged_from_episode"] = src_dir.name
            dst_meta["merged_input_index"] = int(dataset_idx)
            write_json(dst_meta_path, dst_meta)

            merged.append(
                {
                    "episode_id": int(next_episode_id),
                    "source_root": str(input_root),
                    "source_episode": src_dir.name,
                    "task_type": dst_meta.get("task_type", "grasp"),
                    "instruction": dst_meta.get("instruction", ""),
                }
            )
            next_episode_id += 1

    write_json(
        out_root / "merge_manifest.json",
        {
            "input_roots": [str(Path(root).resolve()) for root in input_roots],
            "copy_mode": copy_mode,
            "episodes": merged,
            "num_episodes": len(merged),
        },
    )
    print(f"merged successful episodes: {len(merged)}")
    print(f"output root: {out_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge raw episode_* datasets with compact renumbering.")
    parser.add_argument("--input_roots", nargs="+", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--copy_mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_raw_datasets(
        input_roots=args.input_roots,
        out_root=args.out_root,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )
