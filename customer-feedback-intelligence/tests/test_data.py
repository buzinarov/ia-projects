"""Dataset cleaning invariants. Needs the CSV (cached locally, or downloaded in
CI); skips cleanly if it can't be fetched."""
import pytest

from src.data import TEXT_COLUMN, load_reviews


@pytest.fixture(scope="module")
def reviews():
    try:
        return load_reviews()
    except Exception as exc:  # offline and uncached
        pytest.skip(f"reviews dataset unavailable: {exc}")


def test_no_empty_reviews(reviews):
    assert reviews[TEXT_COLUMN].notna().all()
    assert (reviews[TEXT_COLUMN].str.len() > 0).all()


def test_deduped(reviews):
    assert reviews[TEXT_COLUMN].is_unique


def test_expected_columns_and_scale(reviews):
    for col in ["source_index", TEXT_COLUMN, "Rating", "Recommended IND", "Department Name"]:
        assert col in reviews.columns
    # The published set is 23,486 rows; after dropping empties/dupes we keep ~22.6k.
    assert 22000 < len(reviews) < 23000


def test_first_review_is_the_brief_example(reviews):
    assert reviews.iloc[0][TEXT_COLUMN].startswith("Absolutely wonderful")
