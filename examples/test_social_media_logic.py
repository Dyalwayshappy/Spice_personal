import pytest
from social_media_logic import check_tweet


def test_valid_tweet():
    result = check_tweet("Hello world")
    assert result["is_valid"] is True
    assert result["length"] == 11
    assert result["remaining"] == 269


def test_empty_tweet():
    result = check_tweet("")
    assert result["is_valid"] is True
    assert result["length"] == 0
    assert result["remaining"] == 280


def test_exact_limit():
    text = "x" * 280
    result = check_tweet(text)
    assert result["is_valid"] is True
    assert result["length"] == 280
    assert result["remaining"] == 0


def test_over_limit():
    text = "x" * 281
    result = check_tweet(text)
    assert result["is_valid"] is False
    assert result["length"] == 281
    assert result["remaining"] == -1
