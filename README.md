# JoJo

JoJo is a Python code for computing the transiting light curves of oblate planets by using Green's theorem.

### Theory

By applying Green’s theorem, the two-dimensional areal integral over the occultation area can be transferred into a one-dimensional line integration over the boundary. The detailed derivation can be found in <a href='https://www.overleaf.com/project/6540c5890aa1271d632eba87'>Liu et al. 2024</a>.

### Usage

The key function is ```compute_oblate_transit_lightcurve``` in ```JoJo.py```, which returns the light curves with the given transiting parameters and time series. Now it works only for the quadratic limb-darkening law. See ```example.py``` for specific usages.

For comparison, the code to generate quadratic limb-darkening light curves for spherical planets is included in ```spherical_transit.py``` (<a href='[Analytic Light Curves for Planetary Transit Searches - IOPscience](https://iopscience.iop.org/article/10.1086/345520)'>Mandel & Agol 2002</a>), which, you can call ```compute_spherical_transit_lightcurve``` to generate the light curve of a spherical planet with the same cross-section of the oblate planet.

### Reference
Liu et al. 2024