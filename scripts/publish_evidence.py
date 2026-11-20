"""Promote committed evidence assets without deploying the static Pages shell."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], input=data, check=True, cwd=ROOT, stdout=subprocess.PIPE).stdout


def tree(lines: list[bytes]) -> str:
    return git("mktree", data=b"".join(line + b"\n" for line in lines)).strip().decode("ascii")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--edition", required=True, type=int,
                     help="strictly newer edition number than the deployed site and prior feed")
    cli.add_argument("--remote", default="origin")
    args = cli.parse_args()
    branch = "refs/heads/evidence"
    manifest = json.loads(git("show", "HEAD:site/data/manifest.json"))
    prior = git("ls-remote", "--heads", args.remote, branch).split()
    old_edition = manifest["edition"]
    parents = []
    if prior:
        git("fetch", args.remote, branch)
        sha = prior[0].decode("ascii")
        parents = ["-p", sha]
        old_edition = max(old_edition, json.loads(git("show", f"{sha}:data/manifest.json"))["edition"])
    if args.edition <= old_edition:
        cli.error(f"edition must be newer than {old_edition}")
    manifest["edition"] = args.edition
    blob = git("hash-object", "-w", "--stdin", data=(json.dumps(manifest, ensure_ascii=False) + "\n").encode()).strip()
    assets = git("ls-tree", "HEAD:site/data").splitlines()
    data = tree([line for line in assets if not line.endswith(b"\tmanifest.json")]
                + [b"100644 blob " + blob + b"\tmanifest.json"])
    media = git("rev-parse", "HEAD:site/media").strip().decode("ascii")
    root = tree([f"040000 tree {data}\tdata".encode(), f"040000 tree {media}\tmedia".encode()])
    commit = git("commit-tree", root, *parents, "-m", f"Publish evidence edition {args.edition}").strip().decode("ascii")
    git("push", args.remote, f"{commit}:{branch}")
    print(f"Evidence edition {args.edition}: {commit}")


if __name__ == "__main__":
    main()
