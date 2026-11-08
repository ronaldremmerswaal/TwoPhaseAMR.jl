#!/usr/bin/env python3
r"""Check the technical reference against the code, the tests and the
Documenter API pages, for example:

    python3 docs/techref/check_links.py
    python3 docs/techref/check_links.py --skip-api
    python3 docs/techref/check_links.py --build-dir docs/build --strict-api

Blocking checks (they set the exit status):

  * every \srcsym{path}{symbol} names a function, type, constant or macro
    that is defined in the Julia file `path` of the checked-out tree (HEAD in
    CI); a trailing `!` is part of the name;
  * every \testsym{path}{title} names an @testset of `path` with exactly that
    title (`@testset "title" begin` and the `for` form);
  * the coverage manifest coverage.toml: source classification, chapter status
    and audit commits, figure data, and the claims, whose source symbols and
    test titles are checked in the same way;
  * every techref.pdf#nameddest=<label> link in docs/src names a TeX label.

Reports that do not set the exit status:

  * the Documenter anchors of \apitype and \apifun (blocking with
    --strict-api once the API pages have stable @id anchors);
  * per reference chapter, the referenced source files that changed since the
    chapter's audited commit (git diff --name-only <commit>..HEAD): such a
    chapter needs a re-audit. A shallow clone lacks the audited commits; the
    report then says so.

Paths in the TeX sources and the manifest are relative to the repository
root. The default docs/build directory is resolved relative to this script, an
explicitly supplied build directory relative to the current working
directory. Both pretty-URL and local Documenter layouts are supported.
"""

import argparse
from bisect import bisect_right
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import tomllib
from urllib.parse import unquote


TEX_COMMAND = re.compile(
    r"(?<!\\)(?:\\\\)*\\(?P<kind>apitype|apifun|label)\s*"
    r"\{(?P<value>[^{}]*)\}"
)
CODE_COMMAND = re.compile(
    r"(?<!\\)(?:\\\\)*\\(?P<kind>srcsym|testsym)\s*"
    r"\{(?P<file>[^{}]*)\}\s*\{(?P<value>[^{}]*)\}"
)
TEX_INPUT = re.compile(r"(?<!\\)(?:\\\\)*\\input\s*\{(?P<file>[^{}]*)\}")
PDF_LINK = re.compile(
    r"(?<![\w.-])(?:TwoPhaseAMR-)?techref\.pdf#nameddest="
    r"(?P<label>[^\s\])>\"'`&]*)"
)
CHAPTER_STATUS = re.compile(
    r"\\chapterstatus(?:\[(?P<commit>[^\]]*)\])?\{(?P<status>[^{}]*)\}"
)
MANIFEST_SCHEMA = 2


@dataclass(frozen=True)
class Reference:
    kind: str
    value: str
    path: Path
    line: int
    file: str = ""  # \srcsym, \testsym: the referenced repository file


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

    def line_of(match):
        # The command's backslash is immediately before the named group;
        # preceding escaped backslashes, if any, are not part of the command.
        return lines[bisect_right(starts, match.start("kind") - 1) - 1]

    for match in TEX_COMMAND.finditer(clean):
        yield Reference(match.group("kind"), match.group("value"), path, line_of(match))
    for match in CODE_COMMAND.finditer(clean):
        yield Reference(
            match.group("kind"), match.group("value"), path, line_of(match), match.group("file")
        )


def tex_inputs(path, text):
    """Files read by \\input in a TeX file, resolved as TeX does from the
    directory of the main file (that of this script)."""
    clean, _, _ = uncomment_tex(text)
    for match in TEX_INPUT.finditer(clean):
        name = match.group("file")
        yield name if name.endswith(".tex") else name + ".tex"


def pdf_references(path, text):
    for match in PDF_LINK.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        # A '+' is literal in a fragment, not a form-encoded space.
        yield Reference("pdf", unquote(match.group("label")), path, line)


# Julia sources ------------------------------------------------------------

JULIA_SPECIAL = re.compile(r"""#|"|'""")
CHAR_LITERAL = re.compile(r"'(?:\\.[^'\n]*|[^\\'\n])'")
DEFINITION_PREFIX = r"^[ \t]*(?:(?:[A-Za-z_]\w*\.)*@[\w.]+[ \t]+)*"
QUALIFIER = r"(?:[A-Za-z_]\w*\.)*"
SHORT_FORM_TAIL = re.compile(
    r"[ \t]*(?:::[^=\n]+?)?(?:[ \t]+where[ \t]*(?:\{[^{}]*\}|[^=\n]+?))*[ \t]*=(?![=>])"
)
TESTSET = re.compile(
    r"@testset\b(?:[ \t]+[A-Za-z_]\w*[ \t]*=[ \t]*[^\s\"]+)*[ \t]+"
    r"\"(?P<title>(?:[^\"\\\n]|\\.)*)\""
)


def blank_julia(text, strings):
    """Replace the comments of Julia source text, and with `strings` also its
    string and character literals, by spaces. Newlines are kept, so offsets
    and line numbers are unchanged; block comments nest as in Julia."""

    def blank(segment):
        return re.sub(r"[^\n]", " ", segment)

    out = []
    position = 0
    length = len(text)
    while position < length:
        match = JULIA_SPECIAL.search(text, position)
        if match is None:
            out.append(text[position:])
            break
        start = match.start()
        out.append(text[position:start])
        if text.startswith("#=", start):
            depth = 0
            end = start
            while end < length:
                if text.startswith("#=", end):
                    depth += 1
                    end += 2
                elif text.startswith("=#", end):
                    depth -= 1
                    end += 2
                    if depth == 0:
                        break
                else:
                    end += 1
            out.append(blank(text[start:end]))
        elif text[start] == "#":
            end = text.find("\n", start)
            end = length if end < 0 else end
            out.append(blank(text[start:end]))
        elif text[start] == '"':
            delimiter = '"""' if text.startswith('"""', start) else '"'
            end = start + len(delimiter)
            while end < length and not text.startswith(delimiter, end):
                end += 2 if text[end] == "\\" else 1
            end = min(end + len(delimiter), length)
            out.append(blank(text[start:end]) if strings else text[start:end])
        else:
            # A quote after an operand is the adjoint, otherwise a character.
            literal = CHAR_LITERAL.match(text, start)
            operand = start > 0 and (text[start - 1].isalnum() or text[start - 1] in "_)]}'.!")
            if literal is None or operand:
                out.append("'")
                end = start + 1
            else:
                end = literal.end()
                out.append(blank(text[start:end]) if strings else text[start:end])
        position = end
    return "".join(out)


def _bracket_end(code, start):
    """Index after the bracket group that opens at `start`, or None."""
    depth = 0
    for index in range(start, len(code)):
        character = code[index]
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def defines(code, symbol):
    """Whether Julia code, with comments and strings blanked, defines `symbol`
    at the start of a line: `function [M.]symbol`, `[mutable] struct`,
    `abstract type`, `primitive type`, `const`, `macro`, or the short form
    `[M.]symbol(...)[::T][ where ...] = ...`, also after macros such as
    `@inline` or `Base.@kwdef`."""
    name = re.escape(symbol) + r"(?![\w!])"
    keyword_forms = (
        r"function[ \t]+" + QUALIFIER + name,
        r"(?:mutable[ \t]+)?struct[ \t]+" + name,
        r"(?:abstract|primitive)[ \t]+type[ \t]+" + name,
        r"(?:global[ \t]+)?const[ \t]+" + name,
        r"macro[ \t]+" + name,
    )
    for form in keyword_forms:
        if re.search(DEFINITION_PREFIX + form, code, re.M):
            return True
    for match in re.finditer(DEFINITION_PREFIX + QUALIFIER + name + r"[ \t]*(?=[({])", code, re.M):
        end = match.end()
        while end is not None and end < len(code) and code[end] in "({":
            end = _bracket_end(code, end)
        if end is not None and SHORT_FORM_TAIL.match(code, end):
            return True
    return False


def testset_titles(code):
    """Titles of the @testset calls in Julia code with comments blanked."""
    return {
        re.sub(r"\\(.)", r"\1", match.group("title")) for match in TESTSET.finditer(code)
    }


class JuliaSources:
    """Definitions and @testset titles of the repository's Julia files."""

    def __init__(self, project_root):
        self.project_root = project_root
        self.texts = {}
        self.code = {}
        self.titles = {}

    def _text(self, path):
        if path not in self.texts:
            file = self.project_root / path
            if not path.endswith(".jl"):
                self.texts[path] = (None, f"{path} is not a Julia file")
            elif not file.is_file():
                self.texts[path] = (None, f"{path} does not exist")
            else:
                try:
                    self.texts[path] = (file.read_text(encoding="utf-8"), None)
                except (OSError, UnicodeError) as exc:
                    self.texts[path] = (None, f"cannot read {path}: {exc}")
        return self.texts[path]

    def symbol_problem(self, path, symbol):
        """None if `symbol` is defined in `path`, otherwise the problem."""
        text, problem = self._text(path)
        if problem:
            return problem
        if path not in self.code:
            self.code[path] = blank_julia(text, strings=True)
        if not symbol or not defines(self.code[path], symbol):
            return f"{symbol!r} is not defined in {path}"
        return None

    def testset_problem(self, path, title):
        """None if `path` has an @testset titled `title`, otherwise the problem."""
        text, problem = self._text(path)
        if problem:
            return problem
        if path not in self.titles:
            self.titles[path] = testset_titles(blank_julia(text, strings=False))
        if title not in self.titles[path]:
            return f'no @testset "{title}" in {path}'
        return None


def check_code_references(references, julia, project_root):
    """Blocking check of \\srcsym and \\testsym; returns the number of errors."""
    errors = symbols = titles = 0
    for reference in references:
        if reference.kind == "srcsym":
            symbols += 1
            problem = julia.symbol_problem(reference.file, reference.value)
        else:
            titles += 1
            problem = julia.testset_problem(reference.file, reference.value)
        if problem:
            errors += 1
            print(f"MISSING {reference.kind} ({location(reference, project_root)}): {problem}")
    print(
        f"Code references: {symbols} source symbols, {titles} test titles, "
        f"{errors} errors"
    )
    return errors


# Coverage manifest --------------------------------------------------------


def check_claim(claim, script_dir, julia, tex_labels):
    """Validate one claim of the manifest; returns the number of errors."""
    errors = 0
    label = claim.get("label", "")
    if tex_labels is not None and label not in tex_labels:
        print(f"CLAIM {label!r}: label not defined in the TeX sources")
        errors += 1
    if not claim.get("sources"):
        print(f"CLAIM {label!r}: no sources")
        errors += 1
    entries = (
        ("sources", "symbol", julia.symbol_problem),
        ("tests", "testset", julia.testset_problem),
    )
    for key, field, problem_of in entries:
        values = claim.get(key, [])
        if not isinstance(values, list):
            print(f"CLAIM {label!r}: {key} must be a list")
            errors += 1
            continue
        for entry in values:
            if not (
                isinstance(entry, dict)
                and set(entry) == {"file", field}
                and all(isinstance(value, str) for value in entry.values())
            ):
                print(
                    f"CLAIM {label!r}: {key} entry {entry!r} is not "
                    f"{{ file = ..., {field} = ... }}"
                )
                errors += 1
                continue
            problem = problem_of(entry["file"], entry[field])
            if problem:
                print(f"CLAIM {label!r}: {problem}")
                errors += 1
    check = claim.get("check")
    if check:
        defined = any(
            re.search(r"^function\s+" + re.escape(check) + r"\(", p.read_text(encoding="utf-8"), re.M)
            for p in (script_dir / "figures").glob("*.jl")
        )
        if not defined:
            print(f"CLAIM {label!r}: check function {check!r} not found in figures/*.jl")
            errors += 1
    return errors


def read_manifest(script_dir, project_root):
    path = script_dir / "coverage.toml"
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: cannot read {show_path(path, project_root)}: {exc}")
        return None


def check_manifest(manifest, script_dir, project_root, julia, tex_labels=None):
    """Validate coverage.toml: complete source classification, chapter status
    markers and audit commits, generated figure data and the claims. Returns
    the number of errors."""
    if manifest is None:
        return 1
    if manifest.get("schema") != MANIFEST_SCHEMA:
        print(
            f"ERROR: coverage.toml has schema {manifest.get('schema')!r}, "
            f"expected {MANIFEST_SCHEMA}"
        )
        return 1
    errors = 0
    classes = {"equation", "algebraic", "orchestration", "storage", "ui", "infrastructure"}
    labels = {chapter["label"] for chapter in manifest.get("chapters", [])}
    sources = manifest.get("sources", {})
    actual = {
        p.relative_to(project_root).as_posix()
        for directory in ("src", "ext")
        for p in (project_root / directory).rglob("*.jl")
    }
    for name in sorted(actual - sources.keys()):
        print(f"UNCLASSIFIED source {name}: add it to coverage.toml")
        errors += 1
    for name in sorted(sources.keys() - actual):
        print(f"STALE source {name}: listed in coverage.toml but missing")
        errors += 1
    for name, entry in sorted(sources.items()):
        if entry.get("class") not in classes:
            print(f"INVALID class for {name}: {entry.get('class')!r}")
            errors += 1
        unknown = set(entry.get("chapters", [])) - labels
        if unknown or (entry.get("class") == "equation" and not entry.get("chapters")):
            print(f"INVALID chapters for {name}: {sorted(unknown) or 'none listed'}")
            errors += 1
    for chapter in manifest.get("chapters", []):
        tex = script_dir / chapter["file"]
        text = tex.read_text(encoding="utf-8") if tex.is_file() else ""
        marker = CHAPTER_STATUS.search(text)
        if f"\\label{{{chapter['label']}}}" not in text:
            print(f"MISSING chapter label {chapter['label']} in {chapter['file']}")
            errors += 1
        elif marker is None or marker.group("status") != chapter["status"]:
            found = marker.group("status") if marker else None
            print(f"STATUS mismatch {chapter['file']}: manifest {chapter['status']!r}, TeX {found!r}")
            errors += 1
        elif marker.group("commit") != chapter.get("audited_commit"):
            print(
                f"AUDIT COMMIT mismatch {chapter['file']}: manifest "
                f"{chapter.get('audited_commit')!r}, TeX {marker.group('commit')!r}"
            )
            errors += 1
    for figure in manifest.get("figures", []):
        for key in ("source", "data"):
            if not (script_dir / figure[key]).is_file():
                print(f"MISSING figure {key} {figure[key]} ({figure['label']})")
                errors += 1
    claims = manifest.get("claims", [])
    for claim in claims:
        errors += check_claim(claim, script_dir, julia, tex_labels)
    untested = sum(not claim.get("tests") for claim in claims)
    print(
        f"Coverage manifest: {len(sources)} sources classified, "
        f"{len(labels)} chapters, {len(manifest.get('figures', []))} figures, "
        f"{len(claims)} claims ({untested} without a test), "
        f"{errors} errors (audited commit {manifest.get('audited_commit', '?')[:7]})"
    )
    return errors


# Re-audit report ----------------------------------------------------------


def chapter_files(script_dir, chapter_file, texts):
    """The chapter's TeX file and the files it reads with \\input."""
    files = []
    pending = [script_dir / chapter_file]
    while pending:
        path = pending.pop()
        if path in files or path not in texts:
            continue
        files.append(path)
        pending.extend(script_dir / name for name in tex_inputs(path, texts[path]))
    return files


def chapter_sources(manifest, script_dir, texts, references):
    """Per reference chapter: (audited commit, the source files it refers to
    through \\srcsym, its claims and the manifest's source classification)."""
    default = manifest.get("audited_commit", "")
    label_paths = {
        reference.value: reference.path for reference in references if reference.kind == "label"
    }
    result = {}
    for chapter in manifest.get("chapters", []):
        if chapter.get("status") != "reference":
            continue
        paths = set(chapter_files(script_dir, chapter["file"], texts))
        files = {
            reference.file
            for reference in references
            if reference.kind == "srcsym" and reference.path in paths
        }
        for claim in manifest.get("claims", []):
            if label_paths.get(claim.get("label")) in paths:
                files.update(
                    entry["file"]
                    for entry in claim.get("sources", [])
                    if isinstance(entry, dict) and isinstance(entry.get("file"), str)
                )
        files.update(
            name
            for name, entry in manifest.get("sources", {}).items()
            if chapter["label"] in entry.get("chapters", [])
        )
        result[chapter["label"]] = (chapter.get("audited_commit", default), sorted(files))
    return result


def changed_since(project_root, commit, files):
    """Files among `files` changed between `commit` and HEAD, or a problem."""
    git = ["git", "-C", str(project_root)]
    try:
        known = subprocess.run(
            [*git, "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True, text=True
        )
        if known.returncode != 0:
            return None, f"commit {commit[:7]} is not in the local history (shallow clone?)"
        diff = subprocess.run(
            [*git, "diff", "--name-only", f"{commit}..HEAD", "--", *files],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return None, f"git is unavailable ({exc})"
    if diff.returncode != 0:
        return None, diff.stderr.strip() or "git diff failed"
    return diff.stdout.splitlines(), None


def reaudit_report(sources_by_chapter, project_root):
    """Print the chapters whose referenced sources changed since their audit
    (not blocking); returns the number of such chapters."""
    print("Re-audit report (not blocking): referenced sources changed since the audit")
    stale = 0
    for label, (commit, files) in sources_by_chapter.items():
        name = f"{label} (audited {commit[:7]})"
        changed, problem = changed_since(project_root, commit, files)
        if problem:
            print(f"  NOT COMPARED {name}: {problem}")
        elif changed:
            stale += 1
            print(f"  REAUDIT {name}: {len(changed)} of {len(files)} referenced sources changed:")
            for file in changed:
                print(f"    {file}")
        else:
            print(f"  current {name}: none of {len(files)} referenced sources changed")
    return stale


# API pages ----------------------------------------------------------------


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
        print(f"UNAVAILABLE HTML for /{name}/; expected {expected}")
        return None, None

    parser = ElementIDs()
    try:
        parser.feed(path.read_text(encoding="utf-8"))
        parser.close()
    except (OSError, UnicodeError) as exc:
        print(f"UNAVAILABLE HTML {show_path(path, project_root)}: {exc}")
        return None, path
    return parser.ids, path


def check_api_anchors(references, build_dir, project_root):
    """Check \\apitype and \\apifun against the built HTML; returns the number
    of unavailable pages and missing anchors."""
    if not build_dir.is_dir():
        print(f"UNAVAILABLE Documenter build directory: {build_dir}")
    pages = {
        "apitype": read_page(build_dir, "api", project_root),
        "apifun": read_page(build_dir, "api-functions", project_root),
    }
    page_errors = sum(ids is None for ids, _ in pages.values())
    passed = missing = unverified = 0
    for reference in references:
        ids, path = pages[reference.kind]
        if reference.kind == "apitype":
            anchor = "TwoPhaseAMR." + reference.value
            url = "/api/#" + anchor
        else:
            anchor = reference.value
            url = "/api-functions/#" + anchor
        where = location(reference, project_root)
        if ids is None:
            unverified += 1
            print(f"UNVERIFIED API {url} ({where}): corresponding HTML unavailable")
        elif anchor in ids:
            passed += 1
            print(f"OK API {url} ({where})")
        else:
            missing += 1
            print(
                f"MISSING API {url} ({where}): id {anchor!r} not found in "
                f"{show_path(path, project_root)}"
            )
    print(
        f"API anchors: {len(references)} total, {passed} passed, {missing} missing, "
        f"{unverified} unverified; HTML pages: {2 - page_errors}/2 loaded"
    )
    return page_errors + missing


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
    parser.add_argument(
        "--skip-api", action="store_true",
        help="do not check the API anchors (no Documenter build needed)",
    )
    parser.add_argument(
        "--strict-api", action="store_true",
        help="make unavailable API pages and missing anchors blocking",
    )
    args = parser.parse_args()
    if args.build_dir is not None and args.build_dir_option is not None:
        parser.error("specify the build directory either positionally or with --build-dir")
    if args.skip_api and args.strict_api:
        parser.error("--skip-api and --strict-api exclude each other")
    build_dir = (args.build_dir_option or args.build_dir or docs_dir / "build").resolve()

    source_errors = 0
    tex_complete = True
    texts = {}
    references = []
    for path in sorted([*script_dir.rglob("*.tex"), *script_dir.glob("*.sty")]):
        try:
            texts[path] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            print(f"ERROR: cannot read {show_path(path, project_root)}: {exc}")
            source_errors += 1
            tex_complete = False
            continue
        references.extend(tex_references(path, texts[path]))
    labels = {reference.value for reference in references if reference.kind == "label"}

    julia = JuliaSources(project_root)
    code_errors = check_code_references(
        [r for r in references if r.kind in ("srcsym", "testsym")], julia, project_root
    )

    api_errors = 0
    api_references = [r for r in references if r.kind in ("apitype", "apifun")]
    if args.skip_api:
        print(f"API anchors: {len(api_references)} not checked (--skip-api)")
    else:
        api_errors = check_api_anchors(api_references, build_dir, project_root)

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
        f"PDF labels: {len(pdf_links)} total, {pdf_passed} passed, "
        f"{pdf_missing} missing, {pdf_unverified} unverified; source errors: {source_errors}"
    )

    manifest = read_manifest(script_dir, project_root)
    manifest_errors = check_manifest(
        manifest, script_dir, project_root, julia, labels if tex_complete else None
    )
    stale = 0
    if manifest is not None:
        by_chapter = chapter_sources(manifest, script_dir, texts, references)
        stale = reaudit_report(by_chapter, project_root)

    blocking = source_errors + code_errors + pdf_missing + manifest_errors
    if args.strict_api:
        blocking += api_errors
    print(
        f"Blocking errors: {blocking}; API problems "
        f"({'blocking' if args.strict_api else 'not blocking'}): {api_errors}; "
        f"chapters to re-audit (not blocking): {stale}"
    )
    return int(blocking > 0)


if __name__ == "__main__":
    raise SystemExit(main())
