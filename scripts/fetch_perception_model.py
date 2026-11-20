"""Download the pinned, Apache-2.0 OpenCV Zoo NanoDet weights."""
import argparse
from hashlib import sha256
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopass.carla_backend.rgbd import MODEL_NAME, MODEL_SHA256, MODEL_URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models") / MODEL_NAME)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.is_file():
        with args.out.open("rb") as data:
            if sha256(data.read()).hexdigest() == MODEL_SHA256:
                print(args.out)
                return
        raise ValueError("Existing file does not match. Choose another output path.")
    temp = args.out.with_suffix(".partial")
    with urlopen(MODEL_URL, timeout=60) as response:
        data = response.read(8_000_000)
    temp.write_bytes(data)
    with temp.open("rb") as source:
        if sha256(source.read()).hexdigest() != MODEL_SHA256:
            raise IOError("model digest mismatch")
    temp.replace(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
