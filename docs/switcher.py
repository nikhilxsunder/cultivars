# MIT License
#
# Copyright (c) 2026 Nikhil Sunder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Version-switcher JSON for the documentation site, derived from git tags.

The site is deployed as one directory per version under a common root:

- ``<root>/dev/`` -- the ``dev`` branch, rebuilt on every push.
- ``<root>/stable/`` -- the highest release tag, rebuilt when that tag is pushed.
- ``<root>/<version>/`` -- every release tag, frozen at the time it was pushed.

``switcher.json`` lives at the root and lists: ``dev``, ``stable`` (marked
``preferred``, named after the release it currently holds), and every other
release tag, newest first. The release that ``stable`` points at is not listed
a second time under its own number; it becomes a numbered entry once a newer
release supersedes it.

Two commands::

    python docs/switcher.py write --base-url URL --output PATH
    python docs/switcher.py latest

``write`` renders the JSON; ``latest`` prints the highest release tag (without
its ``v`` prefix), which the deploy workflow compares against the tag it is
building to decide whether to refresh ``stable``.

Tags are read with ``git tag``; the checkout must have them (``fetch-depth: 0``
or ``fetch-tags: true`` in CI). Only tags of the form ``v<PEP 440 version>``
count; anything else is ignored with a note on stderr.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version


def release_tags() -> list[Version]:
    """Return every ``v<version>`` tag as a :class:`Version`, highest first."""
    out = subprocess.run(
        ["git", "tag", "--list", "v*"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    versions: list[Version] = []
    for raw in out.split():
        try:
            versions.append(Version(raw[1:]))
        except InvalidVersion:
            print(f"switcher: ignoring tag {raw!r}: not a PEP 440 version", file=sys.stderr)
    return sorted(versions, reverse=True)


def entries(base_url: str, versions: list[Version]) -> list[dict[str, object]]:
    """Build the switcher entries in display order."""
    base = base_url.rstrip("/") + "/"
    rows: list[dict[str, object]] = [{"name": "dev", "version": "dev", "url": base + "dev/"}]
    if versions:
        top = versions[0]
        rows.append(
            {
                "name": f"{top} (stable)",
                "version": "stable",
                "url": base + "stable/",
                "preferred": True,
            }
        )
        for old in versions[1:]:
            rows.append({"name": str(old), "version": str(old), "url": f"{base}{old}/"})
    return rows


def validate(rows: list[dict[str, object]]) -> None:
    """Fail loudly on the shapes the theme's JavaScript silently mishandles."""
    seen: set[str] = set()
    for row in rows:
        for key in ("name", "version", "url"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise SystemExit(f"switcher: entry {row!r} lacks a non-empty {key!r}")
        if row["version"] in seen:
            raise SystemExit(f"switcher: duplicate version {row['version']!r}")
        seen.add(str(row["version"]))
    if sum(bool(row.get("preferred")) for row in rows) != 1:
        raise SystemExit("switcher: exactly one entry must be marked preferred")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write", help="render switcher.json")
    write.add_argument("--base-url", required=True, help="site root, e.g. https://host/cultivars/")
    write.add_argument("--output", required=True, type=Path, help="where to write the JSON")
    sub.add_parser("latest", help="print the highest release tag without its v prefix")
    args = parser.parse_args(argv)

    versions = release_tags()
    if args.command == "latest":
        if not versions:
            raise SystemExit("switcher: no release tags found")
        print(versions[0])
        return 0

    rows = entries(args.base_url, versions)
    validate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"switcher: wrote {len(rows)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
