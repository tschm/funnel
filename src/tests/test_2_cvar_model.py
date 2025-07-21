"""Tests for the CVaR (Conditional Value at Risk) model.

This module contains tests for the CVaR model functionality, including target generation
and portfolio optimization with CVaR constraints.
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifunnel.models.CVaRmodel import cvar_model
from ifunnel.models.CVaRtargets import get_cvar_targets
from ifunnel.models.ScenarioGeneration import ScenarioGenerator


@pytest.fixture(scope="module")
def n_simulations_target() -> int:
    """Provide the number of simulations for target generation.

    Returns:
        int: Number of simulations for target generation
    """
    return 250


@pytest.fixture()
def cvar_target_data(
    start_test_date: pd.Timestamp,
    weekly_returns: pd.DataFrame,
    scgen: ScenarioGenerator,
    n_simulations_target: int,
    request: pytest.FixtureRequest,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate CVaR targets and benchmark portfolio values for testing.

    This fixture uses the get_cvar_targets function to generate targets based on
    the benchmark portfolio specified in the test parameter.

    Args:
        start_test_date: Start date for test data
        weekly_returns: DataFrame containing weekly returns
        scgen: Scenario generator instance
        n_simulations_target: Number of simulations for target generation
        request: Pytest request object for accessing parameterized fixtures

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (targets, benchmark_port_val)
    """
    # Add one day to start_test_date to ensure we're in the test period
    start_of_test_dataset = str(start_test_date + timedelta(days=1))

    # Get the benchmark from the parameterized fixture
    targets, benchmark_port_val = get_cvar_targets(
        test_date=start_of_test_dataset,
        benchmark=request.getfixturevalue(request.param),  # Benchmark ETF(s)
        budget=100,  # Initial investment amount
        cvar_alpha=0.05,  # Risk level for CVaR calculation
        data=weekly_returns,
        scgen=scgen,
        n_simulations=n_simulations_target,
    )
    return targets, benchmark_port_val


@pytest.fixture(scope="module")
def cvar_dir(resource_dir: Path) -> Path:
    """Provide the directory containing CVaR test resources.

    Args:
        resource_dir: Path to the test resources directory

    Returns:
        Path: Path to the CVaR test resources directory
    """
    return resource_dir / "cvar"


@pytest.mark.parametrize(
    "cvar_target_data, label",
    [("benchmark_isin_1", "1"), ("benchmark_isin_2", "2")],
    indirect=["cvar_target_data"],
)
def test_get_cvar_targets(cvar_target_data: tuple[pd.DataFrame, pd.DataFrame], label: str, cvar_dir: Path) -> None:
    """Test the CVaR target generation against baseline values.

    This test verifies that the generated CVaR targets and benchmark portfolio values
    match the expected baseline values for different benchmark portfolios.

    Args:
        cvar_target_data: Tuple containing (targets, benchmark_port_val) from the fixture
        label: Label identifying which benchmark is being tested ("1" or "2")
        cvar_dir: Path to the CVaR test resources directory
    """
    # Load expected targets from baseline file
    expected_targets = pd.read_csv(cvar_dir / f"targets_{label}_BASE.csv", index_col=0)

    # Load expected benchmark portfolio values from baseline file
    expected_benchmark_port_val = pd.read_csv(
        cvar_dir / f"benchmark_port_val_{label}_BASE.csv",
        index_col=0,
        parse_dates=True,
    )
    # Ensure datetime format is consistent
    expected_benchmark_port_val.index = expected_benchmark_port_val.index.astype("datetime64[us]")

    # Unpack the fixture data
    targets, benchmark_port_val = cvar_target_data

    # Commented code below shows how the baseline files were originally created
    # targets.to_csv(cvar_dir / f"targets_{label}_BASE.csv")
    # benchmark_port_val.to_csv(cvar_dir / f"benchmark_port_val_{label}_BASE.csv")

    # Verify that generated data matches expected data
    pd.testing.assert_frame_equal(targets, expected_targets)
    pd.testing.assert_frame_equal(benchmark_port_val, expected_benchmark_port_val)


@pytest.mark.parametrize("cvar_target_data", ["benchmark_isin_2"], indirect=True)
def test_cvar_model(
    test_narrow_dataset: pd.DataFrame,
    mc_scenarios: np.ndarray,
    cvar_target_data: tuple[pd.DataFrame, pd.DataFrame],
    resource_dir: Path,
    cvar_dir: Path,
) -> None:
    """Test the CVaR portfolio optimization model against baseline values.

    This test verifies that the CVaR model produces the expected portfolio allocations,
    portfolio values, and CVaR values when optimizing with CVaR constraints.

    Args:
        test_narrow_dataset: DataFrame containing returns for the test period for selected assets
        mc_scenarios: Array of generated Monte Carlo scenarios
        cvar_target_data: Tuple containing (targets, benchmark_port_val) from the fixture
        resource_dir: Path to the test resources directory
        cvar_dir: Path to the CVaR test resources directory
    """
    # Load expected portfolio allocation from baseline file
    expected_port_allocation = pd.read_csv(cvar_dir / "port_allocation_BASE.csv", index_col=0)
    # Load expected portfolio value from baseline file
    expected_port_value = pd.read_csv(cvar_dir / "port_value_BASE.csv", index_col=0, parse_dates=True)
    # Load expected portfolio CVaR from baseline file
    expected_port_cvar = pd.read_csv(cvar_dir / "port_cvar_BASE.csv", index_col=0)

    # Load the scenarios from the baseline file
    # Commented code shows how the baseline scenarios were originally saved
    # np.savez_compressed("scgen/scenarios_BASE.npz", scenarios=mc_scenarios)
    generated_scenarios = np.load(resource_dir / "scgen/scenarios_BASE.npz")["scenarios"]

    # Extract targets from the fixture data (ignore benchmark portfolio values)
    targets, _ = cvar_target_data

    # Run the CVaR optimization model
    port_allocation, port_value, port_cvar = cvar_model(
        test_ret=test_narrow_dataset,
        scenarios=generated_scenarios,  # Monte Carlo scenarios
        targets=targets,  # CVaR targets
        budget=100,  # Initial investment amount
        cvar_alpha=0.05,  # Risk level for CVaR calculation
        trans_cost=0.001,  # Transaction cost
        max_weight=1,  # Maximum weight constraint
        solver="ECOS",  # Optimization solver
        lower_bound=0,  # Lower bound for weights
    )

    # Commented code below shows how the baseline files were originally created
    # port_allocation.to_csv(cvar_dir / "port_allocation_BASE.csv")
    # port_value.to_csv(cvar_dir / "port_value_BASE.csv")
    # port_cvar.to_csv(cvar_dir / "port_cvar_BASE.csv")

    # Identify active constraints (where CVaR is at or very close to the target)
    active_constraints = (targets.to_numpy() - port_cvar.to_numpy()) < 1e-5

    # Verify that generated data matches expected data
    # Use a small tolerance for floating point differences
    pd.testing.assert_frame_equal(port_allocation, expected_port_allocation, atol=1e-5)
    pd.testing.assert_frame_equal(port_value, expected_port_value)
    # Only check active constraints since inactive ones may have different values
    pd.testing.assert_frame_equal(port_cvar[active_constraints], expected_port_cvar[active_constraints])
