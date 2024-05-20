import numpy as np

def occultquad(z, p0, gamma, retall=False, verbose=False):
    """Quadratic limb-darkening light curve; cf. Section 4 of Mandel & Agol (2002).

    :INPUTS:
        z -- sequence of positional offset values

        p0 -- planet/star radius ratio

        gamma -- two-sequence.
           quadratic limb darkening coefficients.  (c1=c3=0; c2 =
           gamma[0] + 2*gamma[1], c4 = -gamma[1]).  If only a single
           gamma is used, then you're assuming linear limb-darkening.

    :OPTIONS:
        retall -- bool.
           If True, in addition to the light curve return the
           uniform-disk light curve, lambda^d, and eta^d parameters.
           Using these quantities allows for quicker model generation
           with new limb-darkening coefficients -- the speed boost is
           roughly a factor of 50.  See the second example below.

    :EXAMPLE:
       ::

         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 100)
         gammavals = [[0., 0.], [1., 0.], [2., -1.]]
         figure()
         for gammas in gammavals:
             f = transit.occultquad(z, 0.1, gammas)
             plot(z, f)

       ::

         # Calculate the same geometric transit with two different
         #    sets of limb darkening coefficients:
         from pylab import *
         import transit
         p, b = 0.1, 0.5
         x = (arange(300.)/299. - 0.5)*2.
         z = sqrt(x**2 + b**2)
         gammas = [.25, .75]
         F1, Funi, lambdad, etad = transit.occultquad(z, p, gammas, retall=True)

         gammas = [.35, .55]
         F2 = 1. - ((1. - gammas[0] - 2.*gammas[1])*(1. - F1) +
            (gammas[0] + 2.*gammas[1])*(lambdad + 2./3.*(p > z)) + gammas[1]*etad) /
            (1. - gammas[0]/3. - gammas[1]/6.)
         figure()
         plot(x, F1, x, F2)
         legend(['F1', 'F2'])


    :SEE ALSO:
       :func:`t2z`, :func:`occultnonlin_small`, :func:`occultuniform`

    :NOTES:
       In writing this I relied heavily on the occultquad IDL routine
       by E. Agol and J. Eastman, especially for efficient computation
       of elliptical integrals and for identification of several
       apparent typographic errors in the 2002 paper (see comments in
       the source code).

       From some cursory testing, this routine appears about 9 times
       slower than the IDL version.  The difference drops only
       slightly when using precomputed quantities (i.e., retall=True).
       A large portion of time is taken up in :func:`ellpic_bulirsch`
       and :func:`ellke`, but at least as much is taken up by this
       function itself.  More optimization (or a C wrapper) is desired!
    """
    # 2011-04-15 15:58 IJC: Created; forking from smallplanet_nonlinear
    # 2011-05-14 22:03 IJMC: Now linear-limb-darkening is allowed with
    #                        a single parameter passed in.
    # 2013-04-13 21:06 IJMC: Various code tweaks; speed increased by
    #                        ~20% in some cases.
    # import pdb

    # Initialize:
    z = np.array(z, copy=False)
    lambdad = np.zeros(z.shape, float)
    etad = np.zeros(z.shape, float)
    F = np.ones(z.shape, float)

    p = np.abs(p0)  # Save the original input

    # Define limb-darkening coefficients:
    if len(gamma) < 2 or not hasattr(gamma, '__iter__'):  # Linear limb-darkening
        gamma = np.concatenate([gamma.ravel(), [0.]])
        c2 = gamma[0]
    else:
        c2 = gamma[0] + 2 * gamma[1]

    c4 = -gamma[1]

    # Test the simplest case (a zero-sized planet):
    if p == 0:
        if retall:
            ret = np.ones(z.shape, float), np.ones(z.shape, float), \
                  np.zeros(z.shape, float), np.zeros(z.shape, float)
        else:
            ret = np.ones(z.shape, float)
        return ret

    # Define useful constants:
    fourOmega = 1. - gamma[0] / 3. - gamma[1] / 6.  # Actually 4*Omega
    a = (z - p) * (z - p)
    b = (z + p) * (z + p)
    # k = 0.5 * np.sqrt((1. - a) / (z * p))  # 8%

    p2 = p * p
    z2 = z * z
    ninePi = 9 * np.pi

    # Define the many necessary indices for the different cases:
    pgt0 = p > 0

    i01 = pgt0 * (z >= (1. + p))
    i02 = pgt0 * (z > (.5 + np.abs(p - 0.5))) * (z < (1. + p))
    i03 = pgt0 * (p < 0.5) * (z > p) * (z < (1. - p))
    i04 = pgt0 * (p < 0.5) * (z == (1. - p))
    i05 = pgt0 * (p < 0.5) * (z == p)
    i06 = (p == 0.5) * (z == 0.5)
    i07 = (p > 0.5) * (z == p)
    i08 = (p > 0.5) * (z >= np.abs(1. - p)) * (z < p)
    i09 = pgt0 * (p < 1) * (z > 0) * (z < (0.5 - np.abs(p - 0.5)))
    i10 = pgt0 * (p < 1) * (z == 0)
    i11 = (p > 1) * (z >= 0.) * (z < (p - 1.))
    # any01 = i01.any()
    # any02 = i02.any()
    # any03 = i03.any()
    any04 = i04.any()
    any05 = i05.any()
    any06 = i06.any()
    any07 = i07.any()
    # any08 = i08.any()
    # any09 = i09.any()
    any10 = i10.any()
    any11 = i11.any()
    # print n01, n02, n03, n04, n05, n06, n07, n08, n09, n10, n11
    if verbose:
        allind = i01 + i02 + i03 + i04 + i05 + i06 + i07 + i08 + i09 + i10 + i11
        nused = (i01.sum() + i02.sum() + i03.sum() + i04.sum() + \
                 i05.sum() + i06.sum() + i07.sum() + i08.sum() + \
                 i09.sum() + i10.sum() + i11.sum())

        print("%i/%i indices used" % (nused, i01.size))
        if not allind.all():
            print("Some indices not used!")

    # pdb.set_trace()

    # Lambda^e and eta^d are more tricky:
    # Simple cases:
    lambdad[i01] = 0.
    etad[i01] = 0.

    if any06:
        lambdad[i06] = 1. / 3. - 4. / ninePi
        etad[i06] = 0.09375  # = 3./32.

    if any11:
        lambdad[i11] = 1.
        # etad[i11] = 1.  # This is what the paper says
        etad[i11] = 0.5  # Typo in paper (according to J. Eastman)

    # Lambda_1:
    ilam1 = i02 + i08
    q1 = p2 - z2[ilam1]
    ## This is what the paper says:
    # ellippi = ellpic_bulirsch(1. - 1./a[ilam1], k[ilam1])
    # ellipe, ellipk = ellke(k[ilam1])

    # This is what J. Eastman's code has:

    # 2011-04-24 20:32 IJMC: The following codes act funny when
    #                        sqrt((1-a)/(b-a)) approaches unity.
    qq = np.sqrt((1. - a[ilam1]) / (b[ilam1] - a[ilam1]))
    ellippi = ellpic_bulirsch(1. / a[ilam1] - 1., qq)
    ellipe, ellipk = ellke(qq)
    lambdad[ilam1] = (1. / (ninePi * np.sqrt(p * z[ilam1]))) * \
                     (((1. - b[ilam1]) * (2 * b[ilam1] + a[ilam1] - 3) - \
                       3 * q1 * (b[ilam1] - 2.)) * ellipk + \
                      4 * p * z[ilam1] * (z2[ilam1] + 7 * p2 - 4.) * ellipe - \
                      3 * (q1 / a[ilam1]) * ellippi)

    # Lambda_2:
    ilam2 = i03 + i09
    q2 = p2 - z2[ilam2]

    ## This is what the paper says:
    # ellippi = ellpic_bulirsch(1. - b[ilam2]/a[ilam2], 1./k[ilam2])
    # ellipe, ellipk = ellke(1./k[ilam2])

    # This is what J. Eastman's code has:
    ailam2 = a[ilam2]  # Pre-cached for speed
    bilam2 = b[ilam2]  # Pre-cached for speed
    omailam2 = 1. - ailam2  # Pre-cached for speed
    ellippi = ellpic_bulirsch(bilam2 / ailam2 - 1, np.sqrt((bilam2 - ailam2) / (omailam2)))
    ellipe, ellipk = ellke(np.sqrt((bilam2 - ailam2) / (omailam2)))

    lambdad[ilam2] = (2. / (ninePi * np.sqrt(omailam2))) * \
                     ((1. - 5 * z2[ilam2] + p2 + q2 * q2) * ellipk + \
                      (omailam2) * (z2[ilam2] + 7 * p2 - 4.) * ellipe - \
                      3 * (q2 / ailam2) * ellippi)

    # Lambda_3:
    # ellipe, ellipk = ellke(0.5/ k)  # This is what the paper says
    if any07:
        ellipe, ellipk = ellke(0.5 / p)  # Corrected typo (1/2k -> 1/2p), according to J. Eastman
        lambdad[i07] = 1. / 3. + (16. * p * (2 * p2 - 1.) * ellipe -
                                  (1. - 4 * p2) * (3. - 8 * p2) * ellipk / p) / ninePi

    # Lambda_4
    # ellipe, ellipk = ellke(2. * k)  # This is what the paper says
    if any05:
        ellipe, ellipk = ellke(2. * p)  # Corrected typo (2k -> 2p), according to J. Eastman
        lambdad[i05] = 1. / 3. + (2. / ninePi) * (4 * (2 * p2 - 1.) * ellipe + (1. - 4 * p2) * ellipk)

    # Lambda_5
    ## The following line is what the 2002 paper says:
    # lambdad[i04] = (2./(3*np.pi)) * (np.arccos(1 - 2*p) - (2./3.) * (3. + 2*p - 8*p2))
    # The following line is what J. Eastman's code says:
    if any04:
        lambdad[i04] = (2. / 3.) * (np.arccos(1. - 2 * p) / np.pi - \
                                    (6. / ninePi) * np.sqrt(p * (1. - p)) * \
                                    (3. + 2 * p - 8 * p2) - \
                                    float(p > 0.5))

    # Lambda_6
    if any10:
        lambdad[i10] = -(2. / 3.) * (1. - p2) ** 1.5

    # Eta_1:
    ilam3 = ilam1 + i07  # = i02 + i07 + i08
    z2ilam3 = z2[ilam3]  # pre-cache for better speed
    twoZilam3 = 2 * z[ilam3]  # pre-cache for better speed
    # kappa0 = np.arccos((p2+z2ilam3-1)/(p*twoZilam3))
    # kappa1 = np.arccos((1-p2+z2ilam3)/(twoZilam3))
    # etad[ilam3] = \
    #    (0.5/np.pi) * (kappa1 + kappa0*p2*(p2 + 2*z2ilam3) - \
    #                    0.25*(1. + 5*p2 + z2ilam3) * \
    #                    np.sqrt((1. - a[ilam3]) * (b[ilam3] - 1.)))
    etad[ilam3] = \
        (0.5 / np.pi) * ((np.arccos((1 - p2 + z2ilam3) / (twoZilam3))) + (
            np.arccos((p2 + z2ilam3 - 1) / (p * twoZilam3))) * p2 * (p2 + 2 * z2ilam3) - \
                         0.25 * (1. + 5 * p2 + z2ilam3) * \
                         np.sqrt((1. - a[ilam3]) * (b[ilam3] - 1.)))

    # Eta_2:
    etad[ilam2 + i04 + i05 + i10] = 0.5 * p2 * (p2 + 2. * z2[ilam2 + i04 + i05 + i10])

    # We're done!

    ## The following are handy for debugging:
    # term1 = (1. - c2) * lambdae
    # term2 = c2*lambdad
    # term3 = c2*(2./3.) * (p>z).astype(float)
    # term4 = c4 * etad
    # Lambda^e is easy:
    lambdae = 1. - occultuniform(z, p)  # 14%
    F = 1. - ((1. - c2) * lambdae + \
              c2 * (lambdad + (2. / 3.) * (p > z)) - \
              c4 * etad) / fourOmega  # 13%

    # pdb.set_trace()
    if retall:
        ret = F, lambdae, lambdad, etad
    else:
        ret = F

    # pdb.set_trace()
    return ret


def occultuniform(z, p, complement=False, verbose=False):
    """Uniform-disk transit light curve (i.e., no limb darkening).

    :INPUTS:
       z -- scalar or sequence; positional offset values of planet in
            units of the stellar radius.

       p -- scalar;  planet/star radius ratio.

       complement : bool
         If True, return (1 - occultuniform(z, p))

    :SEE ALSO:  :func:`t2z`, :func:`occultquad`, :func:`occultnonlin_small`
    """
    # 2011-04-15 16:56 IJC: Added a tad of documentation
    # 2011-04-19 15:21 IJMC: Cleaned up documentation.
    # 2011-04-25 11:07 IJMC: Can now handle scalar z input.
    # 2011-05-15 10:20 IJMC: Fixed indexing check (size, not len)
    # 2012-03-09 08:30 IJMC: Added "complement" argument for backwards
    #                        compatibility, and fixed arccos error at
    #                        1st/4th contact point (credit to
    #                        S. Aigrain @ Oxford)
    # 2013-04-13 21:28 IJMC: Some code optimization; ~20% gain.

    z = np.abs(np.array(z, copy=True))
    fsecondary = np.zeros(z.shape, float)
    if p < 0:
        pneg = True
        p = np.abs(p)
    else:
        pneg = False

    p2 = p * p

    if len(z.shape) > 0:  # array entered
        i1 = (1 + p) < z
        i2 = (np.abs(1 - p) < z) * (z <= (1 + p))
        i3 = z <= (1 - p)
        i4 = z <= (p - 1)

        any2 = i2.any()
        any3 = i3.any()
        any4 = i4.any()
        # print i1.sum(),i2.sum(),i3.sum(),i4.sum()

        if any2:
            zi2 = z[i2]
            zi2sq = zi2 * zi2
            arg1 = 1 - p2 + zi2sq
            acosarg1 = (p2 + zi2sq - 1) / (2. * p * zi2)
            acosarg2 = arg1 / (2 * zi2)
            acosarg1[acosarg1 > 1] = 1.  # quick fix for numerical precision errors
            acosarg2[acosarg2 > 1] = 1.  # quick fix for numerical precision errors
            k0 = np.arccos(acosarg1)
            k1 = np.arccos(acosarg2)
            k2 = 0.5 * np.sqrt(4 * zi2sq - arg1 * arg1)
            fsecondary[i2] = (1. / np.pi) * (p2 * k0 + k1 - k2)

        fsecondary[i1] = 0.
        if any3: fsecondary[i3] = p2
        if any4: fsecondary[i4] = 1.

        if verbose:
            if not (i1 + i2 + i3 + i4).all():
                print("warning -- some input values not indexed!")
            if i1.sum() + i2.sum() + i3.sum() + i4.sum() != z.size:
                print("warning -- indexing didn't get the right number of values")



    else:  # scalar entered
        if (1 + p) <= z:
            fsecondary = 0.
        elif (np.abs(1 - p) < z) * (z <= (1 + p)):
            z2 = z * z
            k0 = np.arccos((p2 + z2 - 1) / (2. * p * z))
            k1 = np.arccos((1 - p2 + z2) / (2 * z))
            k2 = 0.5 * np.sqrt(4 * z2 - (1 + z2 - p2) ** 2)
            fsecondary = (1. / np.pi) * (p2 * k0 + k1 - k2)
        elif z <= (1 - p):
            fsecondary = p2
        elif z <= (p - 1):
            fsecondary = 1.

    if pneg:
        fsecondary *= -1

    if complement:
        return fsecondary
    else:
        return 1. - fsecondary


def ellke(k):
    """Compute Hasting's polynomial approximation for the complete
    elliptic integral of the first (ek) and second (kk) kind.

    :INPUTS:
       k -- scalar or Numpy array

    :OUTPUTS:
       ek, kk

    :NOTES:
       Adapted from the IDL function of the same name by J. Eastman (OSU).
       """
    # 2011-04-19 09:15 IJC: Adapted from J. Eastman's IDL code.

    m1 = 1. - k ** 2
    logm1 = np.log(m1)

    # First kind:
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639

    ee1 = 1. + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ee2 = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * (-logm1)

    # Second kind:
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    ek1 = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ek2 = (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * logm1

    return ee1 + ee2, ek1 - ek2


eps = np.finfo(float).eps


def ellpic_bulirsch(n, k, tol=1000 * eps, maxiter=1e4):
    """Compute the complete elliptical integral of the third kind
    using the algorithm of Bulirsch (1965).

    :INPUTS:
       n -- scalar or Numpy array

       k-- scalar or Numpy array

    :NOTES:
       Adapted from the IDL function of the same name by J. Eastman (OSU).
       """
    # 2011-04-19 09:15 IJMC: Adapted from J. Eastman's IDL code.
    # 2011-04-25 11:40 IJMC: Set a more stringent tolerance (from 1e-8
    #                  to 1e-14), and fixed tolerance flag to the
    #                  maximum of all residuals.
    # 2013-04-13 21:31 IJMC: Changed 'max' call to 'any'; minor speed boost.

    # Make p, k into vectors:
    # if not hasattr(n, '__iter__'):
    #    n = array([n])
    # if not hasattr(k, '__iter__'):
    #    k = array([k])

    if not hasattr(n, '__iter__'):
        n = np.array([n])
    if not hasattr(k, '__iter__'):
        k = np.array([k])

    if len(n) == 0 or len(k) == 0:
        return np.array([])

    kc = np.sqrt(1. - k ** 2)
    p = n + 1.

    if min(p) < 0:
        print("Negative p")

    # Initialize:
    m0 = np.array(1.)
    c = np.array(1.)
    p = np.sqrt(p)
    d = 1. / p
    e = kc.copy()

    outsideTolerance = True
    iter = 0
    while outsideTolerance and iter < maxiter:
        f = c.copy()
        c = d / p + c
        g = e / p
        d = 2. * (f * g + d)
        p = g + p;
        g = m0.copy()
        m0 = kc + m0
        if ((np.abs(1. - kc / g)) > tol).any():
            kc = 2. * np.sqrt(e)
            e = kc * m0
            iter += 1
        else:
            outsideTolerance = False
        # if (iter/10.) == (iter/10):
        #    print iter, (np.abs(1. - kc/g))
        # pdb.set_trace()
        ## For debugging:
        # print min(np.abs(1. - kc/g)) > tol
        # print 'tolerance>>', tol
        # print 'minimum>>  ', min(np.abs(1. - kc/g))
        # print 'maximum>>  ', max(np.abs(1. - kc/g)) #, (np.abs(1. - kc/g))

    return .5 * np.pi * (c * m0 + d) / (m0 * (m0 + p))