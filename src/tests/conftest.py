"""Test fixtures for the ifunnel package.

This module contains pytest fixtures that are used across the test suite to provide
common test data, configurations, and utilities.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifunnel.models.MST import minimum_spanning_tree
from ifunnel.models.ScenarioGeneration import MomentGenerator, ScenarioGenerator


@pytest.fixture(scope="module", name="root_dir")
def root_fixture() -> Path:
    """Provide the root directory of the project.

    Returns:
        Path: The root directory path of the project
    """
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture() -> Path:
    """Provide the resources directory for tests.

    Returns:
        Path: The path to the test resources directory
    """
    return Path(__file__).parent / "resources"


@pytest.fixture()
def rng() -> np.random.Generator:
    """Provide a random number generator with a fixed seed for reproducibility.

    Returns:
        np.random.Generator: A seeded random number generator
    """
    test_rng = np.random.default_rng(seed=42)
    return test_rng


@pytest.fixture()
def scgen(rng: np.random.Generator) -> ScenarioGenerator:
    """Provide a scenario generator with a fixed random seed.

    Args:
        rng: A seeded random number generator

    Returns:
        ScenarioGenerator: A scenario generator instance
    """
    sg = ScenarioGenerator(rng)
    return sg


@pytest.fixture(scope="session")
def weekly_returns(resource_dir: Path) -> pd.DataFrame:
    """Load weekly returns data from parquet file.

    Args:
        resource_dir: Path to the test resources directory

    Returns:
        pd.DataFrame: DataFrame containing weekly returns for ETFs
    """
    weekly_returns = pd.read_parquet(resource_dir / "data/all_etfs_rets.parquet.gzip")
    return weekly_returns


@pytest.fixture(scope="session")
def tickers(weekly_returns: pd.DataFrame) -> np.ndarray:
    """Extract ticker symbols from weekly returns data.

    Args:
        weekly_returns: DataFrame containing weekly returns

    Returns:
        np.ndarray: Array of ticker symbols
    """
    tickers = weekly_returns.columns.values
    return tickers


@pytest.fixture(scope="session")
def names(resource_dir: Path) -> np.ndarray:
    """Load ETF names from parquet file.

    Args:
        resource_dir: Path to the test resources directory

    Returns:
        np.ndarray: Array of ETF names
    """
    df_names = pd.read_parquet(resource_dir / "data/all_etfs_rets_name.parquet.gzip")
    names = df_names.columns.values
    return names


@pytest.fixture(scope="session")
def start_train_date() -> pd.Timestamp:
    """Provide the start date for the training period.

    Returns:
        pd.Timestamp: Start date for training data
    """
    return pd.to_datetime("2014-06-11")


@pytest.fixture(scope="session")
def end_train_date() -> pd.Timestamp:
    """Provide the end date for the training period.

    Returns:
        pd.Timestamp: End date for training data
    """
    return pd.to_datetime("2017-07-01")


@pytest.fixture(scope="session")
def start_test_date() -> pd.Timestamp:
    """Provide the start date for the test period.

    Returns:
        pd.Timestamp: Start date for test data
    """
    return pd.to_datetime("2017-07-01")


@pytest.fixture(scope="session")
def end_test_date() -> pd.Timestamp:
    """Provide the end date for the test period.

    Returns:
        pd.Timestamp: End date for test data
    """
    return pd.to_datetime("2022-07-20")


@pytest.fixture(scope="session")
def benchmark_isin_1(tickers: np.ndarray, names: np.ndarray) -> list[str]:
    """Provide the first benchmark ETF ticker.

    Args:
        tickers: Array of ticker symbols
        names: Array of ETF names

    Returns:
        List[str]: List containing the ticker for the benchmark ETF
    """
    benchmarks = ["iShares MSCI ACWI ETF"]
    # Find the ticker corresponding to the benchmark ETF name
    benchmark_isin = [tickers[list(names).index(name)] for name in benchmarks]
    return benchmark_isin


@pytest.fixture(scope="session")
def benchmark_isin_2(tickers: np.ndarray, names: np.ndarray) -> list[str]:
    """Provide the second set of benchmark ETF tickers.

    Args:
        tickers: Array of ticker symbols
        names: Array of ETF names

    Returns:
        List[str]: List containing the tickers for the benchmark ETFs
    """
    benchmarks = [
        "iShares MSCI All Country Asia ex Japan Index Fund ETF",
        "iShares MSCI ACWI ETF",
    ]
    # Find the tickers corresponding to the benchmark ETF names
    benchmark_isin = [tickers[list(names).index(name)] for name in benchmarks]
    return benchmark_isin


@pytest.fixture(scope="session")
def whole_wide_dataset(
    weekly_returns: pd.DataFrame, start_train_date: pd.Timestamp, end_test_date: pd.Timestamp
) -> pd.DataFrame:
    """Provide the complete dataset for the entire period (training + test).

    Args:
        weekly_returns: DataFrame containing weekly returns
        start_train_date: Start date for training data
        end_test_date: End date for test data

    Returns:
        pd.DataFrame: DataFrame containing returns for the entire period
    """
    whole_dataset = weekly_returns[
        (weekly_returns.index >= start_train_date) & (weekly_returns.index <= end_test_date)
    ].copy()
    return whole_dataset


@pytest.fixture(scope="session")
def train_wide_dataset(
    weekly_returns: pd.DataFrame, start_train_date: pd.Timestamp, end_train_date: pd.Timestamp
) -> pd.DataFrame:
    """Provide the training dataset.

    Args:
        weekly_returns: DataFrame containing weekly returns
        start_train_date: Start date for training data
        end_train_date: End date for training data

    Returns:
        pd.DataFrame: DataFrame containing returns for the training period
    """
    train_dataset = weekly_returns[
        (weekly_returns.index >= start_train_date) & (weekly_returns.index <= end_train_date)
    ].copy()
    return train_dataset


@pytest.fixture(scope="session")
def subset_of_assets(train_wide_dataset: pd.DataFrame) -> list[str]:
    """Generate a subset of assets using minimum spanning tree algorithm.

    This fixture runs the minimum spanning tree algorithm multiple times to
    reduce the asset universe to a more manageable size.

    Args:
        train_wide_dataset: DataFrame containing returns for the training period

    Returns:
        List[str]: List of selected asset tickers
    """
    n_mst_runs = 2
    subset_mst_df = train_wide_dataset
    # Run MST algorithm multiple times to reduce the asset universe
    for i in range(n_mst_runs):
        subset_mst, subset_mst_df, _, _ = minimum_spanning_tree(subset_mst_df)
    return subset_mst


@pytest.fixture(scope="session")
def whole_narrow_dataset(whole_wide_dataset: pd.DataFrame, subset_of_assets: list[str]) -> pd.DataFrame:
    """Provide the complete dataset for the entire period, but only for the subset of assets.

    Args:
        whole_wide_dataset: DataFrame containing returns for the entire period
        subset_of_assets: List of selected asset tickers

    Returns:
        pd.DataFrame: DataFrame containing returns for the entire period for selected assets
    """
    whole_dataset = whole_wide_dataset[subset_of_assets]
    return whole_dataset


@pytest.fixture(scope="session")
def test_narrow_dataset(
    weekly_returns: pd.DataFrame,
    start_test_date: pd.Timestamp,
    end_test_date: pd.Timestamp,
    subset_of_assets: list[str],
) -> pd.DataFrame:
    """Provide the test dataset for the subset of assets.

    Args:
        weekly_returns: DataFrame containing weekly returns
        start_test_date: Start date for test data
        end_test_date: End date for test data
        subset_of_assets: List of selected asset tickers

    Returns:
        pd.DataFrame: DataFrame containing returns for the test period for selected assets
    """
    # Filter by date range
    test_dataset = weekly_returns[
        (weekly_returns.index > start_test_date) & (weekly_returns.index <= end_test_date)
    ].copy()
    # Filter by selected assets
    test_dataset = test_dataset[subset_of_assets]
    return test_dataset


@pytest.fixture(scope="session")
def length_test_dataset(test_narrow_dataset: pd.DataFrame) -> int:
    """Provide the number of time periods in the test dataset.

    Args:
        test_narrow_dataset: DataFrame containing returns for the test period

    Returns:
        int: Number of time periods in the test dataset
    """
    return test_narrow_dataset.shape[0]


@pytest.fixture(scope="session")
def n_simulations() -> int:
    """Provide the number of Monte Carlo simulations to run.

    Returns:
        int: Number of simulations
    """
    return 250


@pytest.fixture()
def mc_scenarios(
    moments: tuple[list[np.ndarray], list[np.ndarray]],
    whole_narrow_dataset: pd.DataFrame,
    length_test_dataset: int,
    n_simulations: int,
    scgen: ScenarioGenerator,
) -> np.ndarray:
    """Generate Monte Carlo scenarios for testing.

    This fixture uses the ScenarioGenerator to create Monte Carlo simulations
    based on the provided moments and dataset parameters.

    Args:
        moments: Tuple containing (sigma_list, mu_list) where sigma_list is a list of
                covariance matrices and mu_list is a list of mean vectors
        whole_narrow_dataset: DataFrame containing returns for the entire period for selected assets
        length_test_dataset: Number of time periods in the test dataset
        n_simulations: Number of Monte Carlo simulations to run
        scgen: Scenario generator instance

    Returns:
        np.ndarray: Array of generated scenarios
    """
    sigma_lst, mu_lst = moments

    # Generate Monte Carlo scenarios using the scenario generator
    scenarios = scgen.monte_carlo(
        data=whole_narrow_dataset,
        n_simulations=n_simulations,
        n_test=length_test_dataset,
        sigma_lst=sigma_lst,
        mu_lst=mu_lst,
    )

    return scenarios


@pytest.fixture(scope="session")
def moments(whole_narrow_dataset: pd.DataFrame, length_test_dataset: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate statistical moments (means and covariances) for test periods.

    This fixture uses the MomentGenerator to create sigma (covariance) and mu (mean)
    lists for each test period.

    Args:
        whole_narrow_dataset: DataFrame containing returns for the entire period for selected assets
        length_test_dataset: Number of time periods in the test dataset

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Tuple containing (sigma_list, mu_list) where
                                                  sigma_list is a list of covariance matrices and
                                                  mu_list is a list of mean vectors
    """
    # Generate sigma and mu lists for each test period
    sigma_lst, mu_lst = MomentGenerator.generate_sigma_mu_for_test_periods(
        data=whole_narrow_dataset, n_test=length_test_dataset
    )

    return sigma_lst, mu_lst
