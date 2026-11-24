import pytest

from voice.dial.parser import DialParser


@pytest.fixture
def parser():
    return DialParser()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("call nine eight seven six five four three two one zero", ["9876543210"]),
        ("Dial 9876543210.", ["9876543210"]),
        ("Please call 98765 43210", ["9876543210"]),
        ("call 98-765-43210 now", ["9876543210"]),
        ("dial number double nine triple eight seven six five four", ["99888765" + "4"]),
        ("call plus nine one nine eight seven six five four three two one zero", ["+919876543210"]),
        ("dial +91 98765 43210", ["+919876543210"]),
        ("call oh one one two three four five six", ["01123456"]),
        ("call to the number 112", ["112"]),
        ("dial ९८७६५४३२१०", ["9876543210"]),
        ("call 100 and then dial 200", ["100", "200"]),
    ],
)
def test_finds_numbers(parser, text, expected):
    assert parser.find_numbers(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "9876543210",                      # no dial command
        "call one of my friends",          # too few digits
        "call me tomorrow",
        "dial 1234567890123456",           # more than max digits
        "",
    ],
)
def test_rejects(parser, text):
    assert parser.find_numbers(text) == []


def test_without_trigger_requirement():
    parser = DialParser(require_trigger=False)
    assert parser.find_numbers("my number is 9876543210") == ["9876543210"]


def test_pattern_and_plus_disabled():
    parser = DialParser(number_pattern=r"^[0-9]{10}$", allow_plus=False)
    assert parser.find_numbers("call 9876543210") == ["9876543210"]
    assert parser.find_numbers("call plus 91 9876543210") == []  # 12 digits fails the pattern


def test_custom_triggers():
    parser = DialParser(triggers=["ring"])
    assert parser.find_numbers("ring 555 0100") == ["5550100"]
    assert parser.find_numbers("call 555 0100") == []
