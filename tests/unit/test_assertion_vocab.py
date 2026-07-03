import pytest

from phentrieve.assertion_vocab import (
    AFFIRMED,
    NEGATED,
    NORMAL,
    UNCERTAIN,
    canonicalize_assertion,
    is_excluded,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("present", AFFIRMED),
        ("affirmed", AFFIRMED),
        ("abnormal", AFFIRMED),
        ("absent", NEGATED),
        ("negated", NEGATED),
        ("excluded", NEGATED),
        ("no", NEGATED),
        ("normal", NORMAL),
        ("uncertain", UNCERTAIN),
        ("suspected", UNCERTAIN),
        ("  ABSENT  ", NEGATED),
        (None, AFFIRMED),
        ("nonsense", AFFIRMED),
    ],
)
def test_canonicalize(raw, expected):
    assert canonicalize_assertion(raw) == expected


@pytest.mark.parametrize("raw", ["absent", "negated", "excluded", "NO"])
def test_is_excluded_true(raw):
    assert is_excluded(raw) is True


@pytest.mark.parametrize("raw", ["present", "affirmed", "normal", "uncertain", None])
def test_is_excluded_false(raw):
    assert is_excluded(raw) is False
