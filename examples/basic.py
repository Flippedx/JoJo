"""Minimal non-plotting example for all three supported transit models."""

import numpy as np
from JoJo import oblate_lc, egg_lc, ring_lc


def main():
    time = np.linspace(-0.15, 0.15, 301)
    oblate = [0, 0.3, 10, 0.1, 0.2, 0.4, 0, np.pi / 2, 0.3, 0.2, 0]
    egg = [0, 0.3, 10, 0.12, 0.08, 0.07, 0.4, 0, np.pi / 2, 0.3, 0.2, 0]
    ring = [0, 0.3, 10, 0.1, 0.12, 0.2, 0.4, 0.4, 0, np.pi / 2, 0.3, 0.2, 0, 0.5]
    for name, function, parameters in [
        ("oblate", oblate_lc, oblate),
        ("egg", egg_lc, egg),
        ("ring", ring_lc, ring),
    ]:
        result = function(parameters, time, exp_time=0)
        flux = result[0] if isinstance(result, tuple) else result
        print(f"{name}: min flux = {flux.min():.9f}, finite = {np.isfinite(flux).all()}")


if __name__ == "__main__":
    main()
