"""Tests for package-level attributes and functionality.

This module contains tests for package-level attributes like __version__
and other package-wide functionality.
"""

import re

import ifunnel


def test_version() -> None:
    """Test that the __version__ attribute exists and has the expected format.

    This test verifies that:
    1. The __version__ attribute is defined in the package
    2. The version follows semantic versioning format (X.Y.Z)
    """
    assert hasattr(ifunnel, "__version__")
    assert isinstance(ifunnel.__version__, str)

    # Check that the version follows semantic versioning (X.Y.Z)
    # This regex matches versions like 0.0.0, 1.2.3, 10.20.30
    version_pattern = r"^\d+\.\d+\.\d+$"
    assert re.match(version_pattern, ifunnel.__version__) is not None
