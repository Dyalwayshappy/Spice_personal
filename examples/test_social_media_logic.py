"""Tests for social_media_logic.py"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from social_media_logic import check_post_length, TWITTER_CHAR_LIMIT


def test_valid_short_post():
    result = check_post_length("Hello, world!")
    assert result["valid"] is True
    assert result["length"] == 13
    assert result["limit"] == TWITTER_CHAR_LIMIT
    assert result["remaining"] == TWITTER_CHAR_LIMIT - 13


def test_exactly_at_limit():
    content = "x" * TWITTER_CHAR_LIMIT
    result = check_post_length(content)
    assert result["valid"] is True
    assert result["length"] == TWITTER_CHAR_LIMIT
    assert result["remaining"] == 0


def test_over_limit():
    content = "x" * (TWITTER_CHAR_LIMIT + 1)
    result = check_post_length(content)
    assert result["valid"] is False
    assert result["length"] == TWITTER_CHAR_LIMIT + 1
    assert result["remaining"] == -1


def test_empty_post():
    result = check_post_length("")
    assert result["valid"] is True
    assert result["length"] == 0
    assert result["remaining"] == TWITTER_CHAR_LIMIT


def test_return_keys():
    result = check_post_length("test")
    assert set(result.keys()) == {"valid", "length", "limit", "remaining"}
