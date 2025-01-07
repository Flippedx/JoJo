import numpy as np
import matplotlib.pyplot as plt
from occultquad import occultquad
import time as time_module

def eig_quartic_roots(p):
    '''Finds quartic roots via numerical eigenvalue solver
    `npumpy.linalg.eigvals` from a 4x4 companion matrix'''
    a, b, c, d = (p[1]/p[0], p[2]/p[0],
                  p[3]/p[0], p[4]/p[0])
    A = np.zeros((4, 4))
    A[1:, :3] = np.eye(3)
    A[:, 3] = - np.array([d, c, b, a]).T
    roots = np.linalg.eigvals(A)
    return roots

def solve_star_ellipse_intersections(x0, y0, rp_eq, f, epsilon):
    ''' z_cen & alpha are 1D arraies, rp_eq, f, and epsilon are float numbers.
    flag=(-1, 0, 1): outside, intersect, inside. '''
    n_points = len(x0)
    d0 = np.sqrt(x0**2+y0**2)
    flags = np.zeros(n_points, dtype=int)
    
    if f == 0.: # circular case
        alphas = np.full((n_points, 2), np.nan)
        x_intersect = np.full((n_points, 2), np.nan)
        y_intersect = np.full((n_points, 2), np.nan)
        flags[d0>=(1+rp_eq)] =-1 # fully outside
        flags[d0<=(1-rp_eq)] = 1 # fully inside
        ## dealing with intersecting case ##
        INTERSECT = flags==0
        d0_subset = d0[INTERSECT]
        cos_da = (1+d0_subset**2-rp_eq**2)/(2*d0_subset)
        delta_alpha = np.arccos(cos_da)
        alpha0 = np.arccos(x0[INTERSECT]/d0_subset)
        alphas[INTERSECT, 0] = alpha0 - delta_alpha
        alphas[INTERSECT, 1] = alpha0 + delta_alpha
        x_intersect, y_intersect = np.cos(alphas), np.sin(alphas)
        return (flags, x_intersect, y_intersect, alphas)
    ##
    flags[d0>=(1+rp_eq)] =-1
    flags[d0<=(1-rp_eq)] = 1
    alphas = np.full((n_points, 4), np.nan)
    x_intersect = np.full((n_points, 4), np.nan)
    y_intersect = np.full((n_points, 4), np.nan)
    ## use complex coefficients ##
    b2 = (1-f)**2
    r2 = rp_eq**2
    coeffs = np.zeros((n_points, 5))*0j
    coeffs[:, 0] = 1-b2
    coeffs[:, 1] = 4*b2*x0 - 4j*y0
    coeffs[:, 2] =-2*(1+b2-2*b2*r2+2*b2*x0**2+2*y0**2)
    coeffs[:, 3] = 4*b2*x0 + 4j*y0
    coeffs[:, 4] = 1-b2
    for i in range(n_points):
        if flags[i] != 0:
            continue
        ## if complex coefficients are used ##
        z = np.roots(coeffs[i]) # this can be improved
        xp, yp = np.real(z), np.imag(z)
        good = np.abs((xp-x0[i])**2+(yp-y0[i])**2/b2-r2)<epsilon
        z = np.unique(z[good])
        n_intersection = len(z)
        # if n_intersection > 2:
        #     raise IOError('More than 2 intersections found!')
        # elif n_intersection==2 and np.abs(z[0]-z[1])<(rp_eq/1000): # two intersection points too close to each other
        #     n_intersection = 1
        if n_intersection < 2:
            if d0[i] < 1: # fully inside
                flags[i] = 1
            else: # fully outside
                flags[i] =-1
            continue
        alphas_ini = np.angle(z)
        order = np.argsort(alphas_ini)
        alphas[i, :len(z)] = alphas_ini[order]
        z = z[order]
        x_intersect[i, :len(z)] = np.real(z)
        y_intersect[i, :len(z)] = np.imag(z)
    return (flags, x_intersect, y_intersect, alphas)

def full_occultation(flags, x0, y0, rp_eq, f, u1, u2):
    if len(flags) == 0:
        return np.zeros_like(x0, dtype=float)
    if len(flags) == 1:
        raise IOError('Only one point in transit!')
    if np.any(flags) != 1:
        raise IOError('at least one point not fully inside!')
    r2 = rp_eq**2
    d2 = x0**2 + y0**2
    delta_flux_analytic = np.pi*(1-f)*r2*(1-u1-2*u2+u2/4.*((2-2*f+f**2)*r2+4*d2))
    delta_flux_numeric = np.zeros_like(x0, dtype=float)
    ##
    n_step = 30
    angles_edge = np.linspace(0, 2*np.pi, n_step+1)
    delta_angle = 2*np.pi/n_step
    angles = (angles_edge[:-1]+angles_edge[1:])/2.
    delta_x, delta_y = rp_eq*np.cos(angles), (1-f)*rp_eq*np.sin(angles)
    ##
    prefac = (u1/2.+u2)*(1-f)
    xp = np.array(list(map(lambda x: x+delta_x, x0)))
    yp = np.array(list(map(lambda y: y+delta_y, y0)))
    mu_p = np.sqrt(1-xp**2-yp**2)
    integrand = (mu_p*xp + (1-yp**2)*np.arctan(xp/mu_p))*delta_x[None, :]
    delta_flux_numeric = np.sum(integrand, axis=1)*prefac*delta_angle
    ##
    delta_flux = delta_flux_analytic + delta_flux_numeric
    return delta_flux

def partial_occultation(flags, alphas, x_intersect, y_intersect, x0, y0, rp_eq, f, u1, u2, n_step=30):
    ## THERE IS STILL ISSUE REGARDING WHICH PART OF THE ELLIPSE TO USE ##
    if len(flags) == 0:
        return np.zeros_like(x0, dtype=float)
    if len(flags) == 1:
        raise IOError('Only one point during ingree+egress!')
    if np.any(flags) != 0:
        raise IOError('at least one point without intersections!')
    ## first, compute contribution from stellar limb ##
    delta_alpha = alphas[:, 1] - alphas[:, 0]
    delta_flux_limb = (0.5-u1/2.-0.75*u2)*delta_alpha + \
            (0.5-u1/2.-11*u2/12.)*(x_intersect[:, 1]*y_intersect[:, 1]-x_intersect[:, 0]*y_intersect[:, 0]) + \
            u2/6.*(x_intersect[:, 1]*y_intersect[:, 1]**3-x_intersect[:, 0]*y_intersect[:, 0]**3)
    flag_nx1 = (x_intersect[:, 1]<0)
    flag_px1 = (x_intersect[:, 1]>=0)
    delta_flux_limb[flag_nx1] += (u1+2*u2)*np.pi/4*(4./3.-y_intersect[flag_nx1, 1]+y_intersect[flag_nx1, 1]**3/3.-\
            y_intersect[flag_nx1, 0]+y_intersect[flag_nx1, 0]**3/3.)
    delta_flux_limb[flag_px1] += (u1+2*u2)*np.pi/4*(y_intersect[flag_px1, 1]-y_intersect[flag_px1, 1]**3/3.-\
            y_intersect[flag_px1, 0]+y_intersect[flag_px1, 0]**3/3.)
    ## second, compute contribution from partial ellipse ##
    cosE, sinE = (x_intersect-x0[:, None])/rp_eq, (y_intersect-y0[:, None])/rp_eq/(1-f)
    cosE[cosE<-1] = -1
    cosE[cosE>1] = 1
    E = np.arccos(cosE)
    E[sinE<0] *= -1. # the formal definition [-pi, pi)
    E[E[:, 1]>E[:, 0], 0] += 2*np.pi # make sure E_low>E_up
    first_integral = 0.5*(1-f)*rp_eq**2*(E[:, 0]-E[:, 1]) \
            + 0.5*(x_intersect[:, 0]*y_intersect[:, 0]-x_intersect[:, 1]*y_intersect[:, 1]) \
            + 0.5*x0*(y_intersect[:, 0]-y_intersect[:, 1]) \
            - 0.5*y0*(x_intersect[:, 0]-x_intersect[:, 1])
    ##
    b = (1-f)*rp_eq
    a2, b2 = rp_eq**2, b**2
    ab = (1-f)*a2
    x2, y2 = x0**2, y0**2
    second_integral = 3./8.*ab*(a2+b2+4*x2+4*y2)*(E[:, 0]-E[:, 1]) \
            - 1.5*rp_eq*b2*y0*(np.cos(E[:,0])-np.cos(E[:,1])) \
            - 1.5*b2*x0*y0*(np.cos(2*E[:,0])-np.cos(2*E[:,1])) \
            - 0.5*rp_eq*b2*y0*(np.cos(3*E[:,0])-np.cos(3*E[:,1])) \
            + 0.25*b*x0*(9*a2+3*b2+4*x2+12*y2)*(np.sin(E[:,0])-np.sin(E[:,1])) \
            + 0.25*ab*(a2+3*x2+3*y2)*(np.sin(2*E[:,0])-np.sin(2*E[:,1])) \
            + 0.25*b*x0*(a2-b2)*(np.sin(3*E[:,0])-np.sin(3*E[:,1])) \
            + 0.03125*ab*(a2-3*b2)*(np.sin(4*E[:,0])-np.sin(4*E[:,1]))
    ##
    delta_flux_analytic = (1-u1-2*u2)*first_integral + u2/3.*second_integral
    ##
    delta_angles = (E[:,0]-E[:,1])/n_step
    prefac = (u1/2.+u2)*(1-f)
    ##
    angle_edge = np.linspace(E[:, 1], E[:, 0], n_step+1).T
    angle = (angle_edge[:, 1:] + angle_edge[:, :-1])/2.
    xp = x0[:, None] + rp_eq*np.cos(angle)
    yp = y0[:, None] + rp_eq*np.sin(angle)*(1-f)
    mu_p = np.sqrt(1-xp**2-yp**2)
    integrand = (mu_p*xp + (1-yp**2)*(np.arctan(xp/mu_p)-np.pi/2.)) * (xp-x0[:, None]) # add the pi/2 term to increase the integral precision
    delta_flux_numeric = np.sum(integrand, axis=1)*delta_angles*prefac
    delta_flux_numeric += np.pi/2*(u1/2.+u2)*(y_intersect[:, 0]-y_intersect[:, 0]**3/3.-y_intersect[:, 1]+y_intersect[:, 1]**3/3.) # counteract the pi/2 term
    ##
    delta_flux = delta_flux_limb + delta_flux_analytic + delta_flux_numeric
    return delta_flux

def compute_oblate_transit_lightcurve(transit_parameters, time_array, exp_time=None, supersample_factor=5, n_step=100):
    """
    Compute the lightcurve at given time array (time_array) due to an oblate planet.

    Parameters
    ----------
    transit_parameters : array-like
        Array containing the transit parameters:
        - t_0 : float
            Time of the transit center.
        - b_0 : float
            Impact parameter.
        - period : float
            Orbital period of the planet.
        - rp_eq : float
            Equatorial radius of the planet.
        - f : float
            Oblateness of the planet.
        - obliquity : float
            Obliquity of the planet.
        - u_1 : float
            Quadratic limb-darkening coefficient.
        - u_2 : float
            Quadratic limb-darkening coefficient.
        - log10_rho_star : float
            Logarithm (base 10) of the stellar density.
    time_array : array-like
        Array of time points at which to compute the light curve.
    exp_time : float, optional
        Exposure time for each observation. If None, it is set to the average difference between consecutive time points in time_array. Default is None.
    supersample_factor : int, optional
        Factor by which to supersample the time array for long exposures. Default is 5.

    Returns
    -------
    flux_array : array-like
        Array of flux values corresponding to the input time array.
    """

    epsilon = 1e-10 # precision used in the calculation
    t_0, b_0, period, rp_eq, f, obliquity, u_1, u_2, log10_rho_star = transit_parameters
    a_over_rstar = 3.753*(period**2*10**log10_rho_star)**(1./3.)

    if exp_time == None:
        exp_time = np.average(np.diff(time_array))
    if f*rp_eq**2 < 1e-6: # if expected oblate signal too small, use spherical transit instead
        flux_array, contacts = compute_spherical_transit_lightcurve(transit_parameters, time_array, exp_time=exp_time, supersample_factor=supersample_factor)
        return (flux_array, contacts)
    
    ## find contact points ##
    dt_out = np.sqrt((1+np.sqrt(1-f)*rp_eq)**2-b_0**2)/a_over_rstar/(2*np.pi)*period
    dt_in =  np.sqrt((1-np.sqrt(1-f)*rp_eq)**2-b_0**2)/a_over_rstar/(2*np.pi)*period
    contacts = np.array([t_0-dt_out, t_0-dt_in, t_0+dt_in, t_0+dt_out])
    ####
    ## handle long exposures ##
    if exp_time>0.007: #if exposure>10min, use supersample_factor
        LONG_EXPOSURE = True
        exp_time_begin = time_array - 0.5*exp_time
        exp_time_end = time_array + 0.5*exp_time
        time_array_supersample = np.linspace(exp_time_begin, exp_time_end, supersample_factor).T.flatten()
    else:
        LONG_EXPOSURE = False
        time_array_supersample = time_array
    x0_ini = (time_array_supersample-t_0)/period*2*np.pi*a_over_rstar # assuming long-period; should be slight different if short P
    y0_ini = np.ones_like(x0_ini, dtype=float)*b_0
    x0 = x0_ini*np.cos(obliquity) + y0_ini*np.sin(obliquity)
    y0 =-x0_ini*np.sin(obliquity) + y0_ini*np.cos(obliquity)
    x0[x0<0] *= -1
    y0[y0<0] *= -1
    flags, x_intersect, y_intersect, alphas = solve_star_ellipse_intersections(x0, y0, rp_eq, f, epsilon)
    flux_total = np.pi*(6-2*u_1-u_2)/6.
    delta_flux = np.zeros_like(time_array_supersample, dtype=float)
    ##
    INTERSECT = flags==0
    delta_flux[INTERSECT] = partial_occultation(flags[INTERSECT], alphas[INTERSECT], x_intersect[INTERSECT], y_intersect[INTERSECT], x0[INTERSECT], y0[INTERSECT], rp_eq, f, u_1, u_2, n_step)
    #delta_flux[INTERSECT] = ellipse_integral(x_intersect[INTERSECT], y_intersect[INTERSECT], x0[INTERSECT], y0[INTERSECT], rp_eq, (1-f)*rp_eq, u_1, u_2, n_step) \
    #                    + circle_integral(alphas[INTERSECT], x_intersect[INTERSECT], y_intersect[INTERSECT], u_1, u_2)
    FULL = flags==1
    delta_flux[FULL] = full_occultation(flags[FULL], x0[FULL], y0[FULL], rp_eq, f, u_1, u_2)
    ##
    flux_array = 1-delta_flux/flux_total
    ##
    if LONG_EXPOSURE:
        flux_array = np.mean(flux_array.reshape((-1, supersample_factor)), axis=1)
    return (flux_array, contacts)

def compute_spherical_transit_lightcurve(transit_parameters, time_array, exp_time=None, supersample_factor=5):
    """
    Compute the lightcurve at given time array (time_array) due to a spherical planet which has the same projection with the given oblate planet.

    Parameters:
    -----------
    transit_parameters : array-like
        Array containing the transit parameters:
        - t_0 : float
            Time of the transit center.
        - b_0 : float
            Impact parameter.
        - period : float
            Orbital period of the planet.
        - rp_eq : float
            Equatorial radius of the planet.
        - f : float
            Oblateness of the planet.
        - obliquity : float
            Obliquity of the planet.
        - u_1 : float
            Quadratic limb-darkening coefficient.
        - u_2 : float
            Quadratic limb-darkening coefficient.
        - log10_rho_star : float
            Logarithm (base 10) of the stellar density.
    time_array : array-like
        Array of time points at which to compute the light curve.
    exp_time : float, optional
        Exposure time for each observation. If None, it is set to the average difference between consecutive time points in time_array. Default is None.
    supersample_factor : int, optional
        Factor by which to supersample the time array for long exposures. Default is 5.
        
    Returns:
    --------
    flux_array : array-like
        Array of flux values corresponding to the input time array.
    """

    t_0, b_0, period, rp_eq, f, obliquity, u_1, u_2, log10_rho_star = transit_parameters
    a_over_rstar = 3.753*(period**2*10**log10_rho_star)**(1./3.)
    ## find contact points ##
    dt_out = np.sqrt((1+rp_eq)**2-b_0**2)/a_over_rstar/(2*np.pi)*period
    dt_in =  np.sqrt((1-rp_eq)**2-b_0**2)/a_over_rstar/(2*np.pi)*period
    contacts = np.array([t_0-dt_out, t_0-dt_in, t_0+dt_in, t_0+dt_out])
    ####
    if exp_time == None:
        exp_time = np.average(np.diff(time_array))
    ## handle long exposures ##
    if exp_time>0.007: #if exposure>10min, use supersample_factor
        LONG_EXPOSURE = True
        exp_time_begin = time_array - 0.5*exp_time
        exp_time_end = time_array + 0.5*exp_time
        time_array_supersample = np.linspace(exp_time_begin, exp_time_end, supersample_factor).T.flatten()
    else:
        LONG_EXPOSURE = False
        time_array_supersample = time_array
    z_array = np.sqrt(b_0**2 + ((time_array_supersample-t_0)/period*2*np.pi*a_over_rstar)**2)
    flux_array = occultquad(z_array, u_1, u_2, np.sqrt(1-f)*rp_eq)
    if LONG_EXPOSURE:
        flux_array = np.mean(flux_array.reshape((-1, supersample_factor)), axis=1)
    return (flux_array, contacts)

def test_zhu2014():
    ''' test the program with the Zhu et al. (2014) Figure 1 blue curve. '''
    t_0, b_0, period = 0., 0., 100. # period in days
    rp_eq, f, obliquity = 0.1, 0.1, 0.
    u_1, u_2 = 0.4, 0.26
    log10_rho_star = 0.5 # rho-star in g/cc
    exp_time = 0.001 # short-cadence
    transit_parameters = np.array([t_0, b_0, period, rp_eq, f, obliquity, u_1, u_2, log10_rho_star])
    time_array = np.linspace(t_0-5/24., t_0+5/24., 1000)
    flux_oblate, time_contacts = compute_oblate_transit_lightcurve(transit_parameters, time_array, exp_time=exp_time)
    flux_spherical = compute_spherical_transit_lightcurve(transit_parameters, time_array, exp_time=exp_time)
#    plot_lightcurves(time_array, flux_oblate, flux_spherical, time_contacts)
    ax1 = plt.subplot(211)
    plt.plot(24*(time_array-t_0), flux_oblate)
    plt.plot(24*(time_array-t_0), flux_spherical)
    plt.ylabel('Flux')
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(24*(time_array-t_0), 1e6*(flux_oblate-flux_spherical))
    for ax in [ax1, ax2]:
        for time in time_contacts:
            ax.axvline(24*(time-t_0), ls='--', color='k')
    plt.xlabel('Time since mid transit (hr)')
    plt.ylabel('Oblate transit - spherical transit (ppm)')
    plt.show()
    return

def test_kepler167():
    ''' test the program with Kepler-167e system. '''
    ## below are the parameter values for Kepler-167e ##
#    t_0, b_0, period = 253.28699, 0.233, 1071.23228 # period in days
    t_0, b_0, period = 0., 1.7e-3, 30.36
    rp_eq, f, obliquity = 0.105688442, 0.1, 0.
    u_1, u_2 = 0.63, 0.14
#    u_1, u_2 = 1., 0.8488
#    log10_rho_star = 0.460 # rho_star in unit g/cc
    log10_rho_star = -5.4929e-3 # rho_star in unit g/cc
    exp_time = 0.02 # 30-min cadence
#    exp_time = 0.0001 # 30-min cadence
    transit_parameters = np.array([t_0, b_0, period, rp_eq, f, obliquity, u_1, u_2, log10_rho_star])
    time_array = np.linspace(t_0-12/24., t_0+12/24., 10000)

#    ## test speed ##
#    time_start = time_module.time()
#    for i in range(10):
#        transit_parameters[4] = np.random.random()*0.4
#        transit_parameters[5] = np.random.random()*np.pi - np.pi/2.
#        flux_oblate, time_contacts = compute_oblate_transit_lightcurve(transit_parameters, time_array, exp_time=exp_time)
#    time_end = time_module.time()
#    print(time_end-time_start)
#    return

    time_start = time_module.time()
    flux_oblate, time_contacts = compute_oblate_transit_lightcurve(transit_parameters, time_array, exp_time=exp_time)
    time_mid = time_module.time()
    flux_spherical = compute_spherical_transit_lightcurve(transit_parameters, time_array, exp_time=exp_time)
    time_end = time_module.time()
    print(time_end-time_mid, time_mid-time_start, (time_mid-time_start)/(time_end-time_mid))

    ## plot a set of residual curves with different obliquity angles ##
    obliquities = np.linspace(-90, 90, 7)
    for value in obliquities:
        transit_parameters[5] = value/180.*np.pi
        flux_oblate, time_contacts = compute_oblate_transit_lightcurve(transit_parameters, time_array, exp_time=exp_time)
        if value < 0:
            plt.plot(24*(time_array-t_0), 1e6*(flux_oblate-flux_spherical), ls='--', label='obliquity=%d deg'%value)
        else:
            plt.plot(24*(time_array-t_0), 1e6*(flux_oblate-flux_spherical), label='obliquity=%d deg'%value)
    for time in time_contacts:
        plt.axvline(24*(time-t_0), ls='--', color='k')
    plt.legend(loc=0)
    plt.xlabel('Time since mid transit (hr)')
    plt.ylabel('Oblate transit - spherical transit (ppm)')
    plt.show()
    return

if __name__=='__main__':
    test_kepler167()
#    test_zhu2014()
