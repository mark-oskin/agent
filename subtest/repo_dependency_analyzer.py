#!/usr/bin/env python3
"""Repo Dependency Analyzer.

Parses Python source files in a directory, extracts imports, and attempts
to map them to known PyPI package names to generate a pip-compatible
``requirements.txt``.

Usage:
    python repo_dependency_analyzer.py [path] [-o output.txt]
"""

from __future__ import annotations

import argparse
import ast
import configparser
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class ImportInfo:
    """Represents a discovered import statement."""
    module_name: str  # e.g. 'numpy', 'os', 'requests.auth'
    alias: str | None  # e.g. 'np' if 'import numpy as np'
    package_guess: str | None = None  # Best guess for pip install name


# Known mappings from import name to PyPI package name.
KNOWN_PACKAGE_MAP: Dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "scp": "paramiko",
    "gi": "PyGObject",
    "attr": "attrs",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "jwt": "PyJWT",
    "magic": "python-magic",
    "macaddress": "macaddress",
    "git": "GitPython",
    "xml": "lxml",
}


def parse_imports(filepath: Path) -> List[ImportInfo]:
    """Parse a Python file and extract top-level imports.

    Handles both ``import X`` and ``from Y import Z`` forms.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    imports: List[ImportInfo] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                imports.append(
                    ImportInfo(
                        module_name=module,
                        alias=alias.asname,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0:
                # Relative import; usually internal.
                continue
            for alias in node.names:
                # We only track the top-level package for simplicity.
                top_level = module.split(".")[0]
                imports.append(
                    ImportInfo(
                        module_name=top_level,
                        alias=alias.asname,
                    )
                )

    return imports


def resolve_stdlib() -> Set[str]:
    """Return a set of standard library module names."""
    if hasattr(sys, "stdlib_module_names"):
        return set(sys.stdlib_module_names)
    return {
        "__future__", "abc", "aifc", "argparse", "array", "ast",
        "asynchat", "asyncio", "asyncore", "atexit", "audioop",
        "base64", "bdb", "binascii", "binhex", "bisect", "builtins",
        "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd",
        "code", "codecs", "codeop", "collections", "colorsys",
        "compileall", "concurrent", "configparser", "contextlib",
        "contextvars", "copy", "copyreg", "cProfile", "crypt",
        "csv", "ctypes", "curses", "dataclasses", "datetime", "dbm",
        "decimal", "difflib", "dis", "distutils", "doctest", "email",
        "encodings", "enum", "errno", "faulthandler", "fcntl",
        "filecmp", "fileinput", "fnmatch", "formatter", "fractions",
        "ftplib", "functools", "gc", "getopt", "getpass", "gettext",
        "glob", "grp", "gzip", "hashlib", "heapq", "hmac", "html",
        "http", "idlelib", "imaplib", "imghdr", "imp", "importlib",
        "inspect", "io", "ipaddress", "itertools", "json", "keyword",
        "lib2to3", "linecache", "locale", "logging", "lzma",
        "mailbox", "mailcap", "marshal", "math", "mimetypes",
        "mmap", "modulefinder", "multiprocessing", "netrc", "nis",
        "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev",
        "parser", "pathlib", "pdb", "pickle", "pickletools", "pipes",
        "pkgutil", "platform", "plistlib", "poplib", "posix",
        "posixpath", "pprint", "profile", "pstats", "pty", "pwd",
        "py_compile", "pyclbr", "pydoc", "queue", "quopri",
        "random", "re", "readline", "reprlib", "resource", "rlcompleter",
        "runpy", "sched", "secrets", "select", "selectors", "shelve",
        "shlex", "shutil", "signal", "site", "smtpd", "smtplib",
        "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "sre",
        "ssl", "stat", "statistics", "string", "stringprep", "struct",
        "subprocess", "sunau", "symtable", "sys", "sysconfig",
        "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile",
        "termios", "test", "textwrap", "threading", "time", "timeit",
        "tkinter", "token", "tokenize", "trace", "traceback", "tracemalloc",
        "tty", "turtle", "turtledemo", "types", "typing", "unicodedata",
        "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave",
        "weakref", "webbrowser", "winreg", "winsound", "wsgiref",
        "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile", "zipimport",
        "zlib", "_thread", "ntpath", "posixpath", "genericpath",
        "_io", "_abc", "_collections_abc", "_functools", "_operator",
        "_signal", "_sre", "_stat", "_string", "_warnings",
        "_weakref", "codeop", "distutils", "encodings",
        "lib2to3", "pipes", "sre_compile", "sre_constants",
        "sre_parse", "compiler", "compileall", "pydoc_data",
        "idlelib", "importlib", "zipapp", "zipfile", "zipimport",
    }


def guess_package_name(module_name: str) -> str | None:
    """Try to guess the PyPI package name for a module name."""
    stdlib = resolve_stdlib()
    if module_name in stdlib:
        return None

    if module_name in KNOWN_PACKAGE_MAP:
        return KNOWN_PACKAGE_MAP[module_name]

    return module_name.lower().replace("_", "-")


def get_version_constraints(src_dir: Path) -> Dict[str, str]:
    """Check for pyproject.toml or setup.cfg in src_dir for version constraints.

    Returns a dict mapping package names to version strings (e.g. 'numpy' -> '>=1.20').
    """
    constraints: Dict[str, str] = {}

    # Check pyproject.toml
    pyproject_path = src_dir / "pyproject.toml"
    if pyproject_path.exists():
        try:
            content = pyproject_path.read_text(encoding="utf-8")
            # Simple regex parsing for [tool.poetry.dependencies] or [project.dependencies]
            in_deps = False
            for line in content.splitlines():
                stripped = line.strip()
                if stripped == "[tool.poetry.dependencies]" or stripped == "[project.dependencies]":
                    in_deps = True
                    continue
                if in_deps and stripped.startswith("["):
                    in_deps = False
                    continue
                if in_deps and "=" in stripped:
                    # Handle 'numpy = "^1.20"' or 'numpy = ">=1.20"'
                    parts = stripped.split("=", 1)
                    if len(parts) == 2:
                        pkg_name = parts[0].strip()
                        version_spec = parts[1].strip().strip('"').strip("'")
                        # Normalize package name for pip
                        constraints[pkg_name.lower().replace("_", "-")] = version_spec
        except Exception:
            pass

    # Check setup.cfg
    setup_cfg_path = src_dir / "setup.cfg"
    if setup_cfg_path.exists():
        try:
            config = configparser.ConfigParser()
            config.read(setup_cfg_path)
            if "options" in config and "install_requires" in config["options"]:
                requires = config["options"]["install_requires"]
                for line in requires.splitlines():
                    line = line.strip()
                    if """==""" in line or ">=" in line or "<=" in line:
                        pkg, version = line.split("=", 1)
                        constraints[pkg.strip().lower().replace("_", "-")] = version.strip()
        except Exception:
            pass

    return constraints


def scan_directory(src_dir: Path, exclude: List[str] | None = None) -> List[Path]:
    """Find all Python files in *src_dir* recursively."""
    if exclude is None:
        exclude = []
    exclude_paths = {Path(e).resolve() for e in exclude}

    python_files: List[Path] = []
    for root, dirs, files in os.walk(src_dir):
        root_path = Path(root)
        dirs[:] = [
            d for d in dirs
            if (root_path / d).resolve() not in exclude_paths
        ]
        for fname in files:
            if fname.endswith(".py"):
                fpath = root_path / fname
                if fpath.resolve() not in exclude_paths:
                    python_files.append(fpath)
    return python_files


def analyze_repo(
    src_dir: Path,
    exclude: List[str] | None = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Analyze a repo and return a mapping of ``module -> pip_package`` and version constraints.

    Returns:
        A tuple of (seen_packages dict, version_constraints dict).
    """
    seen_packages: Dict[str, str] = {}
    stdlib = resolve_stdlib()

    py_files = scan_directory(src_dir, exclude)

    for fpath in py_files:
        imports = parse_imports(fpath)
        for imp in imports:
            pkg = guess_package_name(imp.module_name)
            if pkg is None:
                continue
            if imp.module_name in stdlib:
                continue
            if pkg not in seen_packages:
                seen_packages[pkg] = imp.alias

    version_constraints = get_version_constraints(src_dir)

    return seen_packages, version_constraints


def write_requirements(
    packages: Dict[str, str],
    output_path: Path,
    version_constraints: Dict[str, str] | None = None,
    with_alias: bool = False,
) -> None:
    """Write a pip-compatible ``requirements.txt`` file."""
    if version_constraints is None:
        version_constraints = {}

    lines: List[str] = []
    for pkg in sorted(packages.keys()):
        constraint = version_constraints.get(pkg, "")
        if with_alias and packages[pkg]:
            lines.append(f"# import {pkg} as {packages[pkg]} (if aliased)")
        if constraint:
            lines.append(f"{pkg}{constraint}")
        else:
            lines.append(pkg)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a Python repo and generate requirements.txt",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Root directory of the Python project (default: current dir)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("requirements.txt"),
        help="Output file path (default: requirements.txt)",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        default=[],
        help="Directory or file to exclude (can be repeated)",
    )
    parser.add_argument(
        "--no-alias",
        action="store_true",
        help="Do not include inline comment about import aliases",
    )

    args = parser.parse_args(argv)

    src_path = Path(args.path).resolve()
    if not src_path.is_dir():
        print(f"Error: {src_path} is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output.resolve()

    packages, version_constraints = analyze_repo(src_path, args.exclude)

    if not packages:
        print("No third-party dependencies found.")
        output_path.write_text("", encoding="utf-8")
        return

    write_requirements(packages, output_path, version_constraints, with_alias=not args.no_alias)
    print(f"Found {len(packages)} dependency package(s). Written to {output_path}")


if __name__ == "__main__":
    main()
