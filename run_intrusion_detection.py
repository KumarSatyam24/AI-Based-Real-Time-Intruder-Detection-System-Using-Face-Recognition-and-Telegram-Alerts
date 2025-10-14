#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from intrusion_detection import (
    Config,
    setup_directories,
    setup_logging,
    SimpleMotionDetectionSystem,
    list_input_videos,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run Intrusion Detection (modular package)")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--all", action="store_true", help="Process all videos in input_videos/")
    g.add_argument("--select", type=str, help="Comma-separated list of video filenames to process (e.g., intruder_1.mp4,intruder_3.mp4)")
    p.add_argument("--max-frames", type=int, default=None, help="Optional limit on frames per video (default: no limit)")
    p.add_argument("--disable-yolo", action="store_true", help="Disable YOLO person filtering")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config()
    if args.disable_yolo:
        cfg.ENABLE_PERSON_DETECTION = False

    setup_directories(cfg)
    logger = setup_logging(cfg)

    # Resolve selection
    selection = None
    if args.select:
        selection = [s.strip() for s in args.select.split(",") if s.strip()]
        # Validate selection against available videos
        avail = set(list_input_videos(cfg))
        missing = [s for s in selection if s not in avail]
        if missing:
            logger.warning("Some selected videos were not found and will be skipped: %s", ", ".join(missing))
            selection = [s for s in selection if s in avail]

    system = SimpleMotionDetectionSystem(cfg)

    outputs = system.run_batch(selection=selection, max_frames_per_video=args.max_frames)
    if outputs:
        print("Generated:")
        for o in outputs:
            print(" -", o)
        return 0
    else:
        print("No outputs generated. Check input_videos/ for videos and logs for details.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
