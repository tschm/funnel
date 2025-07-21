"""Tests for Marimo notebooks.

This module contains tests to verify that all Marimo notebooks in the book/marimo
directory can be executed without errors.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_notebooks(root_dir: Path) -> None:
    """Test that all Marimo notebooks can be executed without errors.

    This test finds all Python files in the book/marimo directory (which are
    Marimo notebooks) and executes them as Python scripts. It verifies that
    they all run without errors, ensuring that the notebooks remain functional.

    Args:
        root_dir: Path to the root directory of the project
    """
    # Path to the directory containing Marimo notebooks
    path = root_dir / "book" / "marimo"

    # List all .py files in the directory using glob
    py_files = list(path.glob("*.py"))

    # Ensure we found some files to test
    assert len(py_files) > 0, f"No Python files found in {path}"

    # Track any failures
    failures = []

    # Loop over the files and run them
    for py_file in py_files:
        # Run the Python file as a subprocess
        result = subprocess.run([sys.executable, str(py_file)], capture_output=True, text=True)

        # Check if the file executed successfully
        if result.returncode != 0:
            failures.append((py_file.name, result.stderr))

    # Assert that there were no failures
    if failures:
        failure_messages = "\n".join([f"{name}: {error}" for name, error in failures])
        pytest.fail(f"The following notebooks failed to execute:\n{failure_messages}")
