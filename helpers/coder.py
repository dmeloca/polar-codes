from .bit import Bit, xor


def encode(u1: Bit, u2: Bit) -> tuple[Bit, Bit]:
    return xor(u1, u2), u2
