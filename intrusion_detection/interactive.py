from __future__ import annotations

import os
from typing import List

from .config import Config


def list_input_videos(cfg: Config) -> list[str]:
    videos = []
    if not os.path.isdir(cfg.INPUT_VIDEO_DIR):
        return videos
    for name in os.listdir(cfg.INPUT_VIDEO_DIR):
        if name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            videos.append(name)
    return sorted(videos)


def prompt_video_selection(cfg: Config) -> List[str]:
    """Simple text input helper for notebooks/console. Returns a list of selected filenames."""
    vids = list_input_videos(cfg)
    if not vids:
        print("No videos found in input directory.")
        return []
    print("Available videos:")
    for i, v in enumerate(vids, 1):
        print(f"  {i}. {v}")
    print("Type 'all' to process all, a single index (e.g. 1), or comma-separated indices (e.g. 1,3)")
    choice = input("Your choice: ").strip().lower()
    if choice == 'all':
        return vids
    if ',' in choice:
        idxs = []
        for p in choice.split(','):
            p = p.strip()
            if p.isdigit():
                idxs.append(int(p))
        return [vids[i - 1] for i in idxs if 1 <= i <= len(vids)]
    if choice.isdigit():
        i = int(choice)
        if 1 <= i <= len(vids):
            return [vids[i - 1]]
    return []
