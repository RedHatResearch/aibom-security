import struct

from aibom_verifier.hf.safetensors_io import data_byte_range, parse_header_length
from aibom_verifier.slots.comparer import ExactBytesComparer


def test_parse_header_length_known_value():
    encoded = struct.pack("<Q", 12345)
    assert parse_header_length(encoded) == 12345


def test_parse_header_length_ignores_trailing_bytes():
    encoded = struct.pack("<Q", 42) + b"trailing-json-bytes-not-part-of-length"
    assert parse_header_length(encoded) == 42


def test_data_byte_range_known_offsets():
    start, end = data_byte_range(header_length=100, begin=50, end=150)
    assert (start, end) == (8 + 100 + 50, 8 + 100 + 150 - 1)


def test_data_byte_range_zero_offset_start():
    start, end = data_byte_range(header_length=0, begin=0, end=8)
    assert (start, end) == (8, 15)


def test_exact_bytes_comparer():
    comparer = ExactBytesComparer()
    assert comparer.equal(b"abc", b"abc")
    assert not comparer.equal(b"abc", b"abd")
