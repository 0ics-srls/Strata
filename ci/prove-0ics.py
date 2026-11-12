#!/usr/bin/env python3
"""0ics: Strata's own tests that run without a GPU, as the release image's gate (Dockerfile.ubuntu-ai, stage `prove`).

- Python: serve/ and tools/test_*.py the way upstream runs them (tools/check, steps py-serve and py-tools: pytest with the
  repository's conftest.py), in one pytest run, on a copy of the source WITHOUT the top-level engine/: the setup tests
  expect a checkout with no compiled engine in it. Until 0.1.42 this ran every module through unittest, one process
  each; 0.1.59 brought tests that need pytest's setup (tools/ on the import path, pytest-style modules), and a second way
  of running upstream's suite drifts with every release.
- C++: the conversation cache tests (STRATA_BUILD_CONVERSATION_TESTS, CPU only), built in a scratch directory, ctest.

Excluded, by full test id, each with its reason; anything else that fails stops the build. Writes a summary to the file
given as the only argument (the image keeps it as /opt/strata/PROVE.txt).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parent.parent
LIMIT_S: Final = 2700
EXCLUDED: Final = {
    # get_prebuilt asks github.com/Niko1221/Strata/releases for the ready-made engine (HEAD, real network): upstream
    # publishes it for Windows only, so on Linux the answer is 404 and these expectations never hold.
    "tools/test_setup_engine_hash.py::WhatARefusalDoesToTheCaller::"
    "test_updating_keeps_the_installed_engine_instead_of_stopping": "ready-made engine is Windows-only (404 on Linux)",
    "tools/test_setup_engine_hash.py::WhatARefusalDoesToTheCaller::"
    "test_a_first_install_stops_because_there_is_nothing_to_fall_back_to": "ready-made engine is Windows-only (404 on Linux)",
}


def python_tests(source: Path) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "serve",
           *sorted(str(p.relative_to(source)) for p in (source / "tools").glob("test_*.py"))]
    for test in EXCLUDED:
        cmd += ["--deselect", test]
    # a login shell has USER: tools/test_strata_bench.py falls back to the name "tester" without it, and then asks the
    # redactor (which uses getpass.getuser(), "root" here) to hide a name that is not this machine's user
    env = {**os.environ, "USER": os.environ.get("USER") or __import__("getpass").getuser()}
    try:
        done = subprocess.run(cmd, cwd=source, env=env, capture_output=True, text=True, timeout=LIMIT_S, check=False)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"FAIL pytest serve/ tools/: over {LIMIT_S} s")
    if done.returncode != 0:
        sys.stderr.write((done.stdout + done.stderr)[-8000:])
        raise SystemExit("FAIL pytest serve/ tools/")
    last = next((r.strip() for r in reversed(done.stdout.splitlines()) if " passed" in r), "pytest ?")
    return [f"ok   pytest serve/ tools/test_*.py: {last}"]


def conversation_tests(source: Path) -> list[str]:
    build = Path(tempfile.mkdtemp(prefix="ctest-"))
    try:
        subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-DSTRATA_BUILD_CONVERSATION_TESTS=ON",
                        "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_ENABLE_CUDA=OFF"], check=True, capture_output=True)
        subprocess.run(["cmake", "--build", str(build), "-j8"], check=True, capture_output=True)
        done = subprocess.run(["ctest", "--output-on-failure"], cwd=build, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            sys.stderr.write(done.stdout[-4000:])
            raise SystemExit("FAIL C++ conversation tests")
        # one line per test that passed, by name: the gate's summary says WHICH C++ tests ran, not only how many
        names = [m.group(1) for r in done.stdout.splitlines() if "Passed" in r
                 for m in [re.search(r"Test\s+#\d+:\s+(\S+)", r)] if m]   # ctest pads "Test  #5" under 10
        return ([f"ok   ctest: {next(r for r in done.stdout.splitlines() if 'tests passed' in r).strip()}"] +
                [f"ok   ctest {name}" for name in names])
    finally:
        shutil.rmtree(build, ignore_errors=True)


def main() -> None:
    summary: Final = Path(sys.argv[1])
    scratch: Final = Path(tempfile.mkdtemp(prefix="strata-src-"))
    source: Final = scratch / "strata"
    # only the top-level ones: ignore_patterns matched at every depth and dropped metal/src/engine/ (Metal engine sources
    # since 0.1.5x upstream, read by tools/test_qwen4_intake.py)
    top = {"engine", "build", "build-vision", ".venv"}
    shutil.copytree(ROOT, source, symlinks=True,
                    ignore=lambda d, names: [n for n in names if Path(d) == ROOT and n in top])
    try:
        lines = python_tests(source) + conversation_tests(source)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    lines += [f"skip {test}: {why}" for test, why in sorted(EXCLUDED.items())]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
