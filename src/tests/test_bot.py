"""Tests for the bot initialization functionality.

This module contains tests to verify that the bot is properly initialized
with the correct data and attributes.
"""

from pathlib import Path

from ifunnel.models.main import initialize_bot


def test_names(resource_dir: Path) -> None:
    """Test that the bot is initialized with the correct names and tickers.

    This test verifies that the bot can be initialized from a parquet file
    and that it correctly loads the ETF names and tickers.

    Args:
        resource_dir: Path to the test resources directory
    """
    # Path to the test data file
    file = resource_dir / "data" / "all_etfs_rets.parquet.gzip"

    # Initialize the bot with the test data
    bot = initialize_bot(file=file)

    # Verify that names and tickers are loaded and not empty
    assert bot.names is not None, "Bot names should not be None"
    assert len(bot.names) > 0, "Bot names should not be empty"
    assert bot.tickers is not None, "Bot tickers should not be None"
    assert len(bot.tickers) > 0, "Bot tickers should not be empty"

    # Verify that names and tickers have the same length
    assert len(bot.names) == len(bot.tickers), "Names and tickers should have the same length"
