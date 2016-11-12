from prince import util


def test_cosine_similarity():
    """Check `cosine_similarity` works as intented."""

    # Orthogonal vectors
    assert util.cosine_similarity((0, 1), (1, 0)) == 0
    # Identical vectors
    assert util.cosine_similarity((0, 1), (0, 1)) == 1
    # Different magnitude vectors
    assert util.cosine_similarity((1, 1), (42, 42)) == 1


def test_correlation_ratio():
    """Check `correlation_ratio` works as intended."""

    # Each group has a variance of 0
    assert util.correlation_ratio(['a', 'a', 'b', 'b'], [1, 1, 2, 2]) == 0.25 / (0.25 + 0)
    # The first group has a variance of 0.25 and the second one 0
    assert util.correlation_ratio(['a', 'a', 'b', 'b'], [1, 2, 1, 1]) == 0.0625 / (0.0625 + 0.125)
    # Each group has a variance of 0.25
    assert util.correlation_ratio(['a', 'a', 'b', 'b'], [1, 2, 1, 2]) == 0 / (0 + 0.25)
    # Each group has a variance of 0.25 and the groups are unbalanced
    assert util.correlation_ratio(
        ['a', 'a', 'a', 'a', 'b', 'b'],
        [1, 1, 2, 2, 1, 1]
     ) == 0.0625 / (0.0625 + 2/3 * 0.25)
