"""Tests for documentation examples.

This module contains tests to verify that the Python code examples in the
documentation (README.md) can be executed without errors.
"""

import doctest
import os
import re
from pathlib import Path

import pytest


@pytest.fixture()
def docstring(root_dir: Path) -> str:
    """Extract Python code blocks from README.md and format them as a docstring.

    This fixture reads the README.md file, extracts all Python code blocks
    (enclosed in triple backticks with 'python' language identifier), and
    combines them into a single docstring that can be processed by doctest.

    Args:
        root_dir: Path to the root directory of the project

    Returns:
        str: A string containing all Python code blocks formatted as a docstring
    """
    # Read the README.md file
    with open(root_dir / "README.md") as f:
        content = f.read()

    # Extract Python code blocks (assuming they are in triple backticks)
    blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)

    # Join all code blocks into a single string
    code = "\n".join(blocks).strip()

    # Add a docstring wrapper for doctest to process the code
    docstring = f"\n{code}\n"

    return docstring


def test_blocks(root_dir: Path, docstring: str, capfd: pytest.CaptureFixture) -> None:
    """Test that Python code blocks in README.md can be executed without errors.

    This test uses doctest to execute the Python code blocks extracted from
    README.md and verifies that they run without errors. This ensures that
    the documentation examples remain valid and executable.

    Args:
        root_dir: Path to the root directory of the project
        docstring: String containing Python code blocks formatted as a docstring
        capfd: Pytest fixture for capturing stdout/stderr output
    """
    # Change to the root directory to ensure relative imports work
    os.chdir(root_dir)

    try:
        # Run the code blocks as doctests
        doctest.run_docstring_examples(docstring, globals())
    except doctest.DocTestFailure as e:
        # If a DocTestFailure occurs, capture it and manually fail the test
        pytest.fail(f"Doctests failed: {e}")

    # Capture the output after running doctests
    captured = capfd.readouterr()

    # If there is any output (error message), fail the test
    if captured.out:
        pytest.fail(f"Doctests failed with the following output:\n{captured.out} and \n{docstring}")
