"""Module for testing Python code blocks in the README.md file.

This module extracts Python code blocks from the README.md file and runs them
through doctest to ensure they execute correctly. This helps maintain the
accuracy and functionality of code examples in the documentation.
"""

import doctest
import os
import re

import pytest


@pytest.fixture()
def docstring(root_dir):
    """Extract Python code blocks from README.md and format them for doctest.

    This fixture reads the README.md file, extracts all Python code blocks
    (enclosed in triple backticks with 'python' language identifier), and
    formats them as a single docstring that can be processed by doctest.

    Args:
        root_dir: Path to the project root directory.

    Returns:
        str: A string containing all Python code blocks from README.md,
             formatted as a docstring.
    """
    # Read the README.md file
    with open(root_dir / "README.md") as f:
        content = f.read()

    # Extract Python code blocks (assuming they are in triple backticks)
    blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)

    code = "\n".join(blocks).strip()

    # Add a docstring wrapper for doctest to process the code
    docstring = f"\n{code}\n"

    return docstring


def test_blocks(root_dir, docstring, capfd):
    """Test that Python code blocks in README.md execute without errors.

    This function runs the Python code blocks extracted from README.md
    through doctest to verify they execute correctly. It fails the test
    if any errors or output are produced during execution.

    Args:
        root_dir: Path to the project root directory.
        docstring: String containing Python code blocks from README.md.
        capfd: Pytest fixture for capturing stdout/stderr output.
    """
    os.chdir(root_dir)

    try:
        doctest.run_docstring_examples(docstring, globals())
    except doctest.DocTestFailure as e:
        # If a DocTestFailure occurs, capture it and manually fail the test
        pytest.fail(f"Doctests failed: {e}")

    # Capture the output after running doctests
    captured = capfd.readouterr()

    # If there is any output (error message), fail the test
    if captured.out:
        pytest.fail(f"Doctests failed with the following output:\n{captured.out} and \n{docstring}")
