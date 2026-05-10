import argparse
import shutil
from pathlib import Path


TOPIC_CONFIG = [
    {
        "topic": "detection",
        "root": Path("object_detection/object_detection_output"),
        "suffixes": ["_yolo_identification.jpg"],
    },
    {
        "topic": "segmentation_unet",
        "root": Path("segmentation/segmentation_output/custom_segmentation"),
        "suffixes": ["_unet_box_outline.jpg"],
    },
    {
        "topic": "segmentation_postprocess",
        "root": Path("segmentation/segmentation_output/custom_segmentation/postprocessing"),
        "suffixes": ["_connect_masks.jpg"],
    },
    {
        "topic": "soma_segmentation",
        "root": Path("segmentation/segmentation_output/soma_segmentation"),
        "suffixes": ["_soma_mask.jpg"],
    },
    {
        "topic": "skeleton",
        "root": Path("morphology/skeleton_outputs"),
        "suffixes": ["_skeleton.png"],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reorganize pipeline visual outputs into a tile-centric layout. "
            "For each tile, create one folder containing one image per topic."
        )
    )
    parser.add_argument("--output_name", required=True, help="Pipeline output_name used during inference")
    parser.add_argument("--scan_name", required=True, help="Scan folder name used during inference")
    parser.add_argument(
        "--dest_root",
        default="inspection/by_tile",
        help="Destination root for tile-centric folders",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating hard links",
    )
    return parser.parse_args()


def strip_known_suffix(file_name, suffixes):
    for suffix in suffixes:
        if file_name.endswith(suffix):
            return file_name[: -len(suffix)]
    return None


def link_or_copy(src, dst, do_copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if do_copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def gather_topic_files(output_name, scan_name):
    topic_map = {}

    for cfg in TOPIC_CONFIG:
        topic = cfg["topic"]
        topic_dir = cfg["root"] / output_name / scan_name
        if not topic_dir.exists():
            continue

        for f in topic_dir.iterdir():
            if not f.is_file():
                continue

            tile_name = strip_known_suffix(f.name, cfg["suffixes"])
            if tile_name is None:
                continue

            topic_map.setdefault(tile_name, {})[topic] = f

    return topic_map


def main():
    args = parse_args()

    topic_map = gather_topic_files(args.output_name, args.scan_name)
    dest_root = Path(args.dest_root) / args.output_name / args.scan_name
    dest_root.mkdir(parents=True, exist_ok=True)

    topic_counts = {cfg["topic"]: 0 for cfg in TOPIC_CONFIG}

    for tile_name, files_by_topic in sorted(topic_map.items()):
        tile_dir = dest_root / tile_name

        for topic, src in files_by_topic.items():
            dst = tile_dir / f"{topic}{src.suffix.lower()}"
            link_or_copy(src, dst, do_copy=args.copy)
            topic_counts[topic] += 1

    print(f"Created tile-centric view at: {dest_root.resolve()}")
    print(f"Tiles organized: {len(topic_map):,}")
    for topic, count in topic_counts.items():
        print(f"  {topic:<24} {count:,}")


if __name__ == "__main__":
    main()
