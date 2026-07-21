"""
https://www.etsi.org/deliver/etsi_ts/138200_138299/138212/15.02.00_60/ts_138212v150200p.pdf
where the six 5G NR CRC generator polynomials are stated and standardized.
Each entry: crc_type -> (R, taps) where R = degree = number of parity bits,
and taps lists every nonzero exponent of g(D), including the leading R and
the constant term 0.
"""

GEN_POLYS = {
    "CRC6":   (6,  [6, 5, 0]),
    "CRC11":  (11, [11, 10, 9, 5, 0]),
    "CRC16":  (16, [16, 12, 5, 0]),
    "CRC24A": (24, [24, 23, 18, 17, 14, 11, 10, 7, 6, 5, 4, 3, 1, 0]),
    "CRC24B": (24, [24, 23, 6, 5, 1, 0]),
    "CRC24C": (24, [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]),
}

def full_form_int(crc_type):
    """
    g(D) as one Python int (as a decimal number and in its full form), where
    the most-significant bit (MSB), i.e., the D^r term, is at position r.
    """
    #!Position r is from right to left?
    R, taps = GEN_POLYS[crc_type]
    return sum(1 << t for t in taps)