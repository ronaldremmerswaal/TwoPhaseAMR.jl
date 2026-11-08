#!/usr/bin/env python3
"""Check links between the technical reference and the Documenter API pages.

Run after building the Documenter site, for example:

    python3 docs/techref/check_links.py
    python3 docs/techref/check_links.py --build-dir docs/build

The source paths and the default docs/build directory are resolved relative to
this script. An explicitly supplied build directory is relative to the current
working directory. Both pretty-URL and local Documenter layouts are supported.
"""

import argparse
from bisect import bisect_right
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote


TEX_COMMAND = re.compile(
    r"(?<!\\)(?:\\\\)*\\(?P<kind>apitype|apifun|label)\s*"
    r"\{(?P<value>[^{}]*)\}"
)
PDF_LINK = re.compile(
    r"(?<![\w.-])(?:TwoPhaseAMR-)?techref\.pdf#nameddest="
    r"(?P<label>[^\s\])>\"'`&]*)"
)


@dataclass(frozen=True)
class Reference:
    kind: str
    value: str
    path: Path
    line: int


class ElementIDs(HTMLParser):
    """Collect actual element IDs, including HTML character references."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == "id" and value is not None:
                self.ids.add(value)


def uncomment_tex(text):
    """Remove TeX comments while retaining a map to original line numbers.

    A percent sign preceded by an odd number of backslashes is escaped. A
    comment also removes its newline, as in TeX, so a command or argument can
    continue across a commented line break.
    """
    parts = []
    starts = []
    lines = []
    length = 0
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        end = len(line)
        for index, character in enumerate(line):
            if character != "%":
                continue
            previous = index - 1
            while previous >= 0 and line[previous] == "\\":
                previous -= 1
            if (index - previous - 1) % 2 == 0:
                end = index
                break
        part = line[:end]
        if part:
            starts.append(length)
            lines.append(number)
            parts.append(part)
            length += len(part)
    return "".join(parts), starts, lines


def tex_references(path, text):
    clean, starts, lines = uncomment_tex(text)
    for match in TEX_COMMAND.finditer(clean):
        # The command's backslash is immediately before the named group;
        # preceding escaped backslashes, if any, are not part of the command.
        position = match.start("kind") - 1
        line = lines[bisect_right(starts, position) - 1]
        yield Reference(match.group("kind"), match.group("value"), path, line)


def pdf_references(path, text):
    for match in PDF_LINK.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        # A '+' is literal in a fragment, not a form-encoded space.
        yield Reference("pdf", unquote(match.group("label")), path, line)


def show_path(path, project_root):
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def location(reference, project_root):
    return f"{show_path(reference.path, project_root)}:{reference.line}"


def read_page(build_dir, name, project_root):
    """Read the exact API page, preferring the published pretty-URL layout."""
    candidates = (build_dir / name / "index.html", build_dir / f"{name}.html")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        expected = " or ".join(show_path(p, project_root) for p in candidates)
        print(f"ERROR: missing HTML for /{name}/; expected {expected}")
        return None, None

    parser = ElementIDs()
    try:
        parser.feed(path.read_text(encoding="utf-8"))
        parser.close()
    except (OSError, UnicodeError) as exc:
        print(f"ERROR: cannot read {show_path(path, project_root)}: {exc}")
        return None, path
    return parser.ids, path


def main():
    script_dir = Path(__file__).resolve().parent
    docs_dir = script_dir.parent
    project_root = docs_dir.parent

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "build_dir", nargs="?", type=Path, metavar="BUILD_DIR",
        help="Documenter build directory (default: docs/build relative to this script)",
    )
    parser.add_argument(
        "--build-dir", dest="build_dir_option", type=Path, metavar="BUILD_DIR",
        help="Documenter build directory (alternative to the positional argument)",
    )
    args = parser.parse_args()
    if args.build_dir is not None and args.build_dir_option is not None:
        parser.error("specify the build directory either positionally or with --build-dir")
    build_dir = (args.build_dir_option or args.build_dir or docs_dir / "build").resolve()

    source_errors = 0
    tex_complete = True
    api_references = []
    labels = set()
    for path in sorted(script_dir.rglob("*.tex")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            print(f"ERROR: cannot read {show_path(path, project_root)}: {exc}")
            source_errors += 1
            tex_complete = False
            continue
        for reference in tex_references(path, text):
            if reference.kind == "label":
                labels.add(reference.value)
            else:
                api_references.append(reference)

    pdf_links = []
    markdown_dir = docs_dir / "src"
    if not markdown_dir.is_dir():
        print(f"ERROR: Markdown source directory not found: {markdown_dir}")
        source_errors += 1
    else:
        for path in sorted(markdown_dir.rglob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                print(f"ERROR: cannot read {show_path(path, project_root)}: {exc}")
                source_errors += 1
                continue
            pdf_links.extend(pdf_references(path, text))

    if not build_dir.is_dir():
        print(f"ERROR: Documenter build directory not found: {build_dir}")
    pages = {
        "apitype": read_page(build_dir, "api", project_root),
        "apifun": read_page(build_dir, "api-functions", project_root),
    }
    page_errors = sum(ids is None for ids, _ in pages.values())

    api_passed = api_missing = api_unverified = 0
    for reference in api_references:
        ids, path = pages[reference.kind]
        if reference.kind == "apitype":
            anchor = "TwoPhaseAMR." + reference.value
            url = "/api/#" + anchor
        else:
            anchor = reference.value
            url = "/api-functions/#" + anchor
        where = location(reference, project_root)
        if ids is None:
            api_unverified += 1
            print(f"UNVERIFIED API {url} ({where}): corresponding HTML unavailable")
        elif anchor in ids:
            api_passed += 1
            print(f"OK API {url} ({where})")
        else:
            api_missing += 1
            print(
                f"MISSING API {url} ({where}): id {anchor!r} not found in "
                f"{show_path(path, project_root)}"
            )

    pdf_passed = pdf_missing = pdf_unverified = 0
    for reference in pdf_links:
        where = location(reference, project_root)
        if reference.value in labels:
            pdf_passed += 1
            print(f"OK PDF label {reference.value!r} ({where})")
        elif not tex_complete:
            pdf_unverified += 1
            print(f"UNVERIFIED PDF label {reference.value!r} ({where}): TeX source unavailable")
        else:
            pdf_missing += 1
            print(f"MISSING PDF label {reference.value!r} ({where})")

    print(
        f"API anchors: {len(api_references)} total, {api_passed} passed, "
        f"{api_missing} missing, {api_unverified} unverified"
    )
    print(
        f"PDF labels: {len(pdf_links)} total, {pdf_passed} passed, "
        f"{pdf_missing} missing, {pdf_unverified} unverified"
    )
    print(f"HTML pages: {2 - page_errors}/2 loaded; source errors: {source_errors}")
    return int(bool(page_errors or source_errors or api_missing or pdf_missing))


if __name__ == "__main__":
    raise SystemExit(main())
