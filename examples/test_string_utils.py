# examples/test_string_utils.py

from examples.string_utils import normalize_text

def test_normalize_text():
    assert normalize_text(" Hello ") == "hello"
