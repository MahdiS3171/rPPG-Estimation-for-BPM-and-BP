#!/usr/bin/env python
"""Download Google's versioned MediaPipe model bundles and record SHA256 hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MODELS = {
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="models")
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    manifest = {}
    for name,url in MODELS.items():
        target = root/f"{name}_landmarker.task"
        if not target.exists():
            temporary = target.with_suffix(".download")
            with urllib.request.urlopen(url,timeout=45) as response, temporary.open("xb") as f:
                while chunk := response.read(1024*1024):
                    f.write(chunk)
            temporary.rename(target)
        manifest[name] = {"path":str(target),"url":url,"sha256":hashlib.sha256(target.read_bytes()).hexdigest()}
        print(f"{name}: {target} sha256={manifest[name]['sha256']}")
    (root/"landmarkers.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
