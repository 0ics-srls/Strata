#!/usr/bin/env python3
"""0ics: Strata's own tests that run without a GPU, as the release image's gate (Dockerfile.ubuntu-ai, stage `prove`).

- Python: every serve/test_*.py and tools/test_*.py module, each in its own process with a time limit (run all together
  through `unittest discover`, the tools suite does not finish), on a copy of the source WITHOUT engine/: the setup tests
  expect a checkout with no compiled engine in it.
- C++: the conversation cache tests (STRATA_BUILD_CONVERSATION_TESTS, CPU only), built in a scratch directory, ctest.

Excluded, by full test id, each with its reason; anything else that fails stops the build. Writes a summary to the file
given as the only argument (the image keeps it as /opt/strata/PROVE.txt).
"""

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parent.parent
LIMIT_S: Final = 300
EXCLUDED: Final = {
    # get_prebuilt asks github.com/Niko1221/Strata/releases for the ready-made engine (HEAD, real network): upstream
    # publishes it for Windows only, so on Linux the answer is 404 and these expectations never hold.
    "tools.test_setup_engine_hash.WhatARefusalDoesToTheCaller."
    "test_updating_keeps_the_installed_engine_instead_of_stopping": "ready-made engine is Windows-only (404 on Linux)",
    "tools.test_setup_engine_hash.WhatARefusalDoesToTheCaller."
    "test_a_first_install_stops_because_there_is_nothing_to_fall_back_to": "ready-made engine is Windows-only (404 on Linux)",
}

RUN_ONE: Final = """
import sys, unittest
excluded = set(sys.argv[2:])
suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
def keep(s):
    out = unittest.TestSuite()
    for t in s:
        if isinstance(t, unittest.TestSuite):
            out.addTest(keep(t))
        elif t.id() not in excluded:
            out.addTest(t)
    return out
result = unittest.TextTestRunner(verbosity=0).run(keep(suite))
sys.exit(0 if result.wasSuccessful() else 1)
"""


def python_tests(source: Path) -> list[str]:
    lines: list[str] = []
    modules = sorted(p for d in ("serve", "tools") for p in (source / d).glob("test_*.py"))
    for path in modules:
        module = f"{path.parent.name}.{path.stem}"
        # a module written for pytest (parametrize, pytest.raises: upstream's tools/test_nvfp4_local_pack.py since 0.1.5x)
        # runs under pytest, from requirements-dev.txt that the prove stage installs; unittest would find no test in it
        uses_pytest = "import pytest" in path.read_text(encoding="utf-8")
        cmd = ([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path.relative_to(source))]
               if uses_pytest else [sys.executable, "-c", RUN_ONE, module, *EXCLUDED])
        started = time.monotonic()
        try:
            done = subprocess.run(cmd, cwd=source, capture_output=True, text=True, timeout=LIMIT_S, check=False)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"FAIL {module}: over {LIMIT_S} s")
        if done.returncode != 0:
            sys.stderr.write((done.stdout + done.stderr)[-4000:])
            raise SystemExit(f"FAIL {module}")
        ran = (next((r.strip() for r in reversed(done.stdout.splitlines()) if " passed" in r), "pytest ?") if uses_pytest
               else next((r for r in done.stderr.splitlines() if r.startswith("Ran ")), "Ran ?"))
        lines.append(f"ok   {module}: {ran} ({time.monotonic() - started:.0f} s)")
    return lines


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
        return [f"ok   ctest: {next(r for r in done.stdout.splitlines() if 'tests passed' in r).strip()}"]
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
