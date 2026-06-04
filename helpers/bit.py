from typing import Literal

Bit = Literal[0, 1]


def xor(a: Bit, b: Bit) -> Bit:
    return 1 if a != b else 0
