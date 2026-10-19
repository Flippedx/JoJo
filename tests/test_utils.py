import numpy as np

from JoJo.utils import limb_darkening_to_q, q_to_limb_darkening


def test_limb_darkening_roundtrip():
    u1, u2 = np.array([0.1, 0.3, 0.6]), np.array([0.2, 0.2, 0.1])
    q1, q2 = limb_darkening_to_q(u1, u2)
    a, b = q_to_limb_darkening(q1, q2)
    np.testing.assert_allclose(a, u1)
    np.testing.assert_allclose(b, u2)
