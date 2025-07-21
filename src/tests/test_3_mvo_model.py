"""Tests for the MVO (Mean-Variance Optimization) model.

This module contains tests for the MVO model functionality, including target generation
and portfolio optimization with mean-variance constraints.
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifunnel.models.MVOmodel import mvo_model
from ifunnel.models.MVOtargets import get_mvo_targets


@pytest.fixture(scope="module")
def mvo_target_data(
    start_test_date: pd.Timestamp, weekly_returns: pd.DataFrame, request: pytest.FixtureRequest
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate MVO targets and benchmark portfolio values for testing.

    This fixture uses the get_mvo_targets function to generate targets based on
    the benchmark portfolio specified in the test parameter.

    Args:
        start_test_date: Start date for test data
        weekly_returns: DataFrame containing weekly returns
        request: Pytest request object for accessing parameterized fixtures

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (targets, benchmark_port_val)
    """
    # Add one day to start_test_date to ensure we're in the test period
    start_of_test_dataset = str(start_test_date + timedelta(days=1))

    # Get the benchmark from the parameterized fixture
    targets, benchmark_port_val = get_mvo_targets(
        test_date=start_of_test_dataset,
        benchmark=request.getfixturevalue(request.param),  # Benchmark ETF(s)
        budget=100,  # Initial investment amount
        data=weekly_returns,
    )
    return targets, benchmark_port_val


@pytest.mark.parametrize(
    "mvo_target_data, label",
    [("benchmark_isin_1", "1"), ("benchmark_isin_2", "2")],
    indirect=["mvo_target_data"],
)
def test_get_mvo_targets(mvo_target_data: tuple[pd.DataFrame, pd.DataFrame], label: str, resource_dir: Path) -> None:
    """Test the MVO target generation against baseline values.

    This test verifies that the generated MVO targets and benchmark portfolio values
    match the expected baseline values for different benchmark portfolios.

    Args:
        mvo_target_data: Tuple containing (targets, benchmark_port_val) from the fixture
        label: Label identifying which benchmark is being tested ("1" or "2")
        resource_dir: Path to the test resources directory
    """
    # Load expected targets from baseline file
    expected_targets = pd.read_csv(resource_dir / f"mvo/targets_{label}_BASE.csv", index_col=0)

    # Load expected benchmark portfolio values from baseline file
    expected_benchmark_port_val = pd.read_csv(
        resource_dir / f"mvo/benchmark_port_val_{label}_BASE.csv",
        index_col=0,
        parse_dates=True,
    )
    # Ensure datetime format is consistent
    expected_benchmark_port_val.index = expected_benchmark_port_val.index.astype("datetime64[us]")

    # Unpack the fixture data
    targets, benchmark_port_val = mvo_target_data

    # Commented code below shows how the baseline files were originally created
    # targets.to_csv(f"tests/mvo/targets_{label}_ACTUAL.csv")
    # benchmark_port_val.to_csv(f"tests/mvo/benchmark_port_val_{label}_ACTUAL.csv")

    # Verify that generated data matches expected data
    pd.testing.assert_frame_equal(targets, expected_targets)
    pd.testing.assert_frame_equal(benchmark_port_val, expected_benchmark_port_val)


@pytest.mark.parametrize("mvo_target_data", ["benchmark_isin_2"], indirect=True)
def test_mvo_model(
    test_narrow_dataset: pd.DataFrame,
    moments: tuple[list[np.ndarray], list[np.ndarray]],
    mvo_target_data: tuple[pd.DataFrame, pd.DataFrame],
    resource_dir: Path,
) -> None:
    """Test the MVO portfolio optimization model against baseline values.

    This test verifies that the MVO model produces the expected portfolio allocations,
    portfolio values, and risk values when optimizing with mean-variance constraints.

    Args:
        test_narrow_dataset: DataFrame containing returns for the test period for selected assets
        moments: Tuple containing (sigma_list, mu_list) where sigma_list is a list of
                covariance matrices and mu_list is a list of mean vectors
        mvo_target_data: Tuple containing (targets, benchmark_port_val) from the fixture
        resource_dir: Path to the test resources directory
    """
    # Load expected portfolio allocation from baseline file
    expected_port_allocation = pd.read_csv(resource_dir / "mvo/port_allocation_BASE.csv", index_col=0)
    # Load expected portfolio value from baseline file
    expected_port_value = pd.read_csv(resource_dir / "mvo/port_value_BASE.csv", index_col=0, parse_dates=True)
    # Load expected portfolio risk from baseline file
    expected_port_risk = pd.read_csv(resource_dir / "mvo/port_risk_BASE.csv", index_col=0)

    # Extract targets from the fixture data (ignore benchmark portfolio values)
    targets, _ = mvo_target_data
    # Unpack the moments
    sigma_lst, mu_lst = moments

    # Run the MVO optimization model
    port_allocation, port_value, port_risk = mvo_model(
        mu_lst=mu_lst,  # Mean vectors
        sigma_lst=sigma_lst,  # Covariance matrices
        test_ret=test_narrow_dataset,  # Test dataset
        targets=targets,  # Risk targets
        budget=100,  # Initial investment amount
        trans_cost=0.001,  # Transaction cost
        max_weight=1,  # Maximum weight constraint
        solver="ECOS",  # Optimization solver
        lower_bound=0,  # Lower bound for weights
    )

    # Commented code below shows how the baseline files were originally created
    # port_allocation.to_csv("mvo/port_allocation_BASE.csv")
    # port_value.to_csv("mvo/port_value_BASE.csv")
    # port_risk.to_csv("mvo/port_risk_BASE.csv")

    # Identify active constraints (where risk is at or very close to the target)
    active_constraints = (targets.to_numpy() - port_risk.to_numpy()) < 1e-5

    # Verify that generated data matches expected data
    # Use a small tolerance for floating point differences
    pd.testing.assert_frame_equal(port_allocation, expected_port_allocation, atol=1e-5)
    pd.testing.assert_frame_equal(port_value, expected_port_value)
    # Only check active constraints since inactive ones may have different values
    pd.testing.assert_frame_equal(port_risk[active_constraints], expected_port_risk[active_constraints])
