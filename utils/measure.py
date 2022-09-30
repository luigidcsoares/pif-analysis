#!/usr/bin/env python3

from math import log

def KL(prior, posterior):
    assert len(prior) == len(posterior)

    # Compute p(x) * log2 (p(x) / q(x)) for element x,
    # assuming that the result is 0 when p(x) = 0.
    kl_x = lambda qx, px: 0 if px == 0 else px * log(px / qx, 2)

    return sum(map(kl_x , prior, posterior))
