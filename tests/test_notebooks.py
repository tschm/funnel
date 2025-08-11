"""Module for testing Python notebooks in the project.

This module runs all Python files in the book/marimo directory to ensure
they execute without errors. This helps maintain the functionality of
example notebooks and tutorials in the project.
"""

import subprocess
import sys


def test_notebooks(root_dir):
    """Test that all Python notebooks in the book/marimo directory execute without errors.

    This function finds all Python files in the book/marimo directory and runs them
    using subprocess. It prints the output of each file and whether it ran successfully.

    Args:
        root_dir: Path to the project root directory.
    """
    # loop over all notebooks
    path = root_dir / "book" / "marimo"

    # List all .py files in the directory using glob
    py_files = list(path.glob("*.py"))

    # Loop over the files and run them
    for py_file in py_files:
        print(f"Running {py_file.name}...")
        result = subprocess.run([sys.executable, str(py_file)], capture_output=True, text=True)

        # Print the result of running the Python file
        if result.returncode == 0:
            print(f"{py_file.name} ran successfully.")
            print(f"Output: {result.stdout}")
        else:
            print(f"Error running {py_file.name}:")
            print(f"stderr: {result.stderr}")
