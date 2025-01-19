## JoJo

JoJo is a Python code for computing the transiting light curves of oblate planets by using Green's theorem.

### Theory

By applying Green’s theorem, the two-dimensional areal integral over the occultation area can be transferred into a one-dimensional line integration over the boundary. The detailed derivation can be found in <a href='https://iopscience.iop.org/article/10.3847/1538-3881/ad9b8c'>Liu et al. 2024</a>.

### Usage

The key function is ```compute_oblate_transit_lightcurve``` in ```JoJo.py```, which returns the light curves with the given transiting parameters and time series. Now it works only for the quadratic limb-darkening law. See ```example.ipynb``` for specific usages.

For comparison, the code to generate quadratic limb-darkening light curves of spherical planets developed by <a href='https://ui.adsabs.harvard.edu/abs/2013PASP..125...83E/abstract'>Eastman, Gaudi & Agol 2013</a> is included in ```occultquad.py```, you can call ```compute_spherical_transit_lightcurve``` in ```JoJo.py``` to generate the light curve of a spherical planet with the same cross-section of the oblate planet.

### Future work
- [ ] Applying other LD laws.
- [ ] Calculate light curves for planets with rings.
- [ ] May develop JAX version.

### Reference
<a href='https://iopscience.iop.org/article/10.3847/1538-3881/ad9b8c'>Liu et al. 2024</a>