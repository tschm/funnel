"""Tests for scenario generation functionality.

This module contains tests for the scenario generation components of the ifunnel package,
including moment generation and Monte Carlo scenario generation.
"""

from pathlib import Path

import numpy as np

# def test_monte_carlo_scenarios(mc_scenarios: np.ndarray) -> None:
#     """
#     Test Monte Carlo scenario generation against baseline values.
#
#     Args:
#         mc_scenarios: Generated Monte Carlo scenarios from the fixture
#     """
#     expected_scenarios = np.load("tests/scgen/scenarios_BASE.npz")["scenarios"]
#
#     # np.savez_compressed("scgen/scenarios_BASE.npz", scenarios=mc_scenarios)
#     np.testing.assert_array_equal(mc_scenarios, expected_scenarios)


def test_moments(moments: tuple[list[np.ndarray], list[np.ndarray]], resource_dir: Path) -> None:
    """Test moment generation (means and covariances) against baseline values.

    This test verifies that the generated statistical moments (means and covariance matrices)
    match the expected baseline values within a small tolerance.

    Args:
        moments: Tuple containing (sigma_list, mu_list) where sigma_list is a list of
                covariance matrices and mu_list is a list of mean vectors
        resource_dir: Path to the test resources directory
    """
    # Load expected sigma (covariance) matrices from baseline file
    expected_sigmas = np.load(resource_dir / "scgen/sigma_list_BASE.npz")
    expected_sigma_list = list(expected_sigmas[k] for k in expected_sigmas)

    # Load expected mu (mean) vectors from baseline file
    expected_mus = np.load(resource_dir / "scgen/mu_list_BASE.npz")
    expected_mu_list = list(expected_mus[k] for k in expected_mus)

    # Unpack the moments fixture
    sigma_list, mu_list = moments

    # Commented code below shows how the baseline files were originally created
    # np.savez_compressed(
    #     "scgen/mu_list_BASE.npz",
    #     **dict(zip([f"mu_{i}" for i in range(len(mu_list))], mu_list))
    # )
    # np.savez_compressed(
    #     "scgen/sigma_list_BASE.npz",
    #     **dict(zip([f"sigma_{i}" for i in range(len(sigma_list))], sigma_list))
    # )

    # Use almost_equal instead of exact equality because floating point calculations
    # on different architectures may produce slightly different results
    np.testing.assert_array_almost_equal(mu_list, expected_mu_list)
    np.testing.assert_array_almost_equal(sigma_list, expected_sigma_list)
