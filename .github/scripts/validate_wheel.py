import argparse
import importlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path


def check_installation() -> None:
    installation_path = {
        Path(sysconfig.get_path(name)).resolve()
        for name in ("platlib", "purelib")
    }
    module_names = (
        "option_implied_moments",
        "option_implied_moments.ext.omp_utils",
        "option_implied_moments.ext.trapezoid_rnm",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if module.__file__ is None:
            raise RuntimeError(f"Module {module_name} has no installed file")
        module_path = Path(module.__file__).resolve()
        if not any(
            module_path.is_relative_to(path) for path in installation_path
        ):
            raise RuntimeError(
                f"Module {module_name} loaded from {module_path}, "
                f"outside the validation environment"
            )
        if module_name != "option_implied_moments" and not str(
            module_path
        ).endswith(tuple(EXTENSION_SUFFIXES)):
            raise RuntimeError(
                f"Module {module_name} loaded from {module_path}, "
                f"but is not a compiled extension"
            )
        print(f"Installed module: {module_path}", flush=True)


def check_threading(mode: str) -> None:
    from option_implied_moments.ext.omp_utils import (
        get_max_threads,
        set_num_threads,
    )

    initial_threads = get_max_threads()
    if initial_threads <= 0:
        raise RuntimeError("Runtime must provide at least one thread")
    try:
        set_num_threads(2)
        actual_threads = get_max_threads()
        expected_threads = {"serial": (1,), "parallel": (2,), "auto": (1, 2)}
        if actual_threads not in expected_threads[mode]:
            raise RuntimeError(
                f"Expected {mode} thread support, got {actual_threads} threads"
            )
        print(
            f"OpenMP mode: {mode}; requested 2, got {actual_threads} threads",
            flush=True,
        )
    finally:
        set_num_threads(initial_threads)


def run_tests() -> None:
    project = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["OMP_NUM_THREADS"] = "2"

    with tempfile.TemporaryDirectory(prefix="oim-wheel-tests-") as directory:
        workdir = Path(directory)
        shutil.copytree(
            project / "tests",
            workdir / "tests",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (workdir / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "pytest",
                "-c",
                "pytest.ini",
                "tests",
                "-q",
            ],
            cwd=workdir,
            env=environment,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an installed wheel")
    parser.add_argument(
        "--openmp", choices=("serial", "parallel", "auto"), default="auto"
    )
    arguments = parser.parse_args()
    check_installation()
    check_threading(arguments.openmp)
    run_tests()


if __name__ == "__main__":
    main()
