"""Testing the clustering module."""

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage

from ifunnel.models.clustering import fancy_dendrogram


def test_fancy_dendrogram_basic():
    """Test basic functionality of the fancy_dendrogram function."""
    data = [[1, 2], [2, 3], [6, 7], [8, 9]]
    z = linkage(data, method="complete")
    result = fancy_dendrogram(z, no_plot=True)
    assert isinstance(result, dict)
    assert "leaves" in result
    assert "dcoord" in result


def test_fancy_dendrogram_with_max_d():
    """Test fancy_dendrogram with max_d parameter."""
    data = [[1, 2], [2, 3], [6, 7], [8, 9]]
    z = linkage(data, method="complete")
    max_d = 5
    result = fancy_dendrogram(z, max_d=max_d, no_plot=True)
    assert "color_list" in result
    assert len(result["color_list"]) > 0


def test_fancy_dendrogram_plot_creation():
    """Ensure fancy_dendrogram creates a plot when no_plot=False."""
    data = [[1, 2], [2, 3], [6, 7], [8, 9]]
    z = linkage(data, method="complete")
    plt.figure()
    fancy_dendrogram(z, no_plot=False)
    assert plt.gcf().number == 1
