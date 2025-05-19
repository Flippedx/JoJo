# JoJo
- JoJo is a package for modeling the transit light curves of oblate planets and planets with rings.
- ```jojo_oblate``` module is used to compute the transiting light curves of oblate planets by using Green's theorem.
- ```jojo_ring``` module can compute the transiting light curves of spherical planets with rings, also using Green's theorem. Comparing to the existing code <a href='https://github.com/EdanRein/pyPplusS'>pyPplusS</a>, JoJo_with_ring can be about 5-10 times faster.

### Installation
You can install ``JoJo`` through:
```
git clone https://github.com/Flippedx/JoJo
cd JoJo
pip install -e .
```

### Theory

By applying Green’s theorem, the two-dimensional areal integral over the occultation area can be transferred into a one-dimensional line integration over the boundary. The detailed derivation can be found in <a href='https://iopscience.iop.org/article/10.3847/1538-3881/ad9b8c'>Liu et al. 2024</a>.

### Usage

- The key function is ```oblate_lc``` in ```jojo_oblate.py```, which returns the light curves with the given transiting parameters and time series. Now it works only for the quadratic limb-darkening law. 

- For comparison, the code to generate quadratic limb-darkening light curves of spherical planets developed by <a href='https://ui.adsabs.harvard.edu/abs/2013PASP..125...83E/abstract'>Eastman, Gaudi & Agol 2013</a> is included in ```occultquad.py```, you can call ```spherical_lc``` in ```jojo_oblate.py``` to generate the light curve of a spherical planet with the same cross-section of the oblate planet.

- Based on ```spherical_lc``` and ```oblate_lc```, ```ring_lc``` in ```jojo_ring.py``` returns the light curves of a spherical planets with ring given transiting parameters and time series.

See ```example.ipynb``` for specific usages.

### Future work
- [x] Calculate light curves for planets with rings.
- [ ] Applying other LD laws.
- [ ] May develop JAX version.
- [ ] Improve the documentation.

### Reference
<a href='https://iopscience.iop.org/article/10.3847/1538-3881/ad9b8c'>Liu et al. 2024</a>