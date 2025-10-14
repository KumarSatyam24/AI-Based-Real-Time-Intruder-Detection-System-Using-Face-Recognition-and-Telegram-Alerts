#!/usr/bin/env python3
"""Interactive tool to define door/window ROIs for damage detection."""
from __future__ import annotations

import argparse
import sys
import os

from intrusion_detection import select_rois_for_video, Config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select ROIs (doors/windows) for damage detection")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    
    args = parser.parse_args(argv)
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    success = select_rois_for_video(args.video, frame_width=args.width, frame_height=args.height)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
