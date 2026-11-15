"""Prepare the local single-camera LeRobot v2.1 data for all three policies."""

import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lerobot_dataset import DEFAULT_CACHE, DEFAULT_CONFIG, prepare_lerobot_v21


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", default="/home/kejia/Documents/pnp_red_cup_fov167_1103_openpi")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--image-key", default="observation.images.fisheye_img")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    result = prepare_lerobot_v21(args.data, args.cache_dir, image_key=args.image_key,
                                split_seed=args.split_seed, train_ratio=args.train_ratio,
                                show_progress=True, reader=config["data"]["reader"])
    print(json.dumps({"cache_dir": result["cache_dir"], "split": result["split"],
                      "split_path": result["split_path"], "stats_path": result["stats_path"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
