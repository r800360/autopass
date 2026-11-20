"""Make a fast-start browser copy and a poster from an archived reel."""
import argparse
from pathlib import Path
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("target", type=Path)
    p.add_argument("--width", type=int, default=1440)
    p.add_argument("--crf", type=int, default=22)
    args = p.parse_args()
    if args.target.resolve() == args.source.resolve():
        p.error("Source and output must differ.")
    args.target.parent.mkdir(parents=True, exist_ok=True)
    ff = get_ffmpeg_exe()
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-n", "-i", str(args.source),
                    "-map_metadata", "-1", "-an", "-vf", f"scale='min({args.width},iw)':-2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
                    "-preset", "fast", "-movflags", "+faststart", str(args.target)], check=True)
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-ss", "4", "-i", str(args.target),
                    "-frames:v", "1", "-c:v", "libwebp", "-quality", "86",
                    str(args.target.with_suffix(".webp"))], check=True)
    print(f"{args.source.stat().st_size:,} to {args.target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
