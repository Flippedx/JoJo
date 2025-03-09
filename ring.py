import numpy as np
import time
from JoJo import oblate_lc, spherical_lc, full_occultation, solve_star_ellipse_intersections

def solve_planet_ring_intersections(x_0, y_0, r_p, r_rin, r_rout, f_r):
    '''Solve the intersections between a planet and a ring.'''
    factor_in = r_rin**2 * (2*f_r - f_r**2)
    factor_out = r_rout**2 * (2*f_r - f_r**2)

    x_in = np.sqrt(r_rin**2 * (r_p**2 - r_rin**2 * (1-f_r)**2) / factor_in)
    y_in = np.sqrt(r_rin**2 * (1-f_r)**2 * (r_rin**2 - r_p**2) / factor_in)
    xs_in = np.column_stack((x_0 - x_in, x_0 - x_in, x_0 + x_in, x_0 + x_in))
    ys_in = np.column_stack((y_0 - y_in, y_0 + y_in, y_0 - y_in, y_0 + y_in))

    if (r_p > r_rin * (1-f_r)) and (r_p < r_rout * (1-f_r)):
        xs_out = np.full_like(xs_in, np.nan)
        ys_out = np.full_like(ys_in, np.nan)
    else:
        x_out = np.sqrt(r_rout**2 * (r_p**2 - r_rout**2 * (1-f_r)**2) / factor_out)
        y_out = np.sqrt(r_rout**2 * (1-f_r)**2 * (r_rout**2 - r_p**2) / factor_out)
        xs_out = np.column_stack((x_0 - x_out, x_0 - x_out, x_0 + x_out, x_0 + x_out))
        ys_out = np.column_stack((y_0 - y_out, y_0 + y_out, y_0 - y_out, y_0 + y_out))
    return xs_in, ys_in, xs_out, ys_out

def ellipse_integral(x_bound, y_bound, x0, y0, a, b, u1, u2, n_step):
    ''' Calculate the integral on the ellipse boundary where the stellar and ellipse intersect. 
    From bound[:, :, 1] to bound[:, :, 0]'''
    delta_flux_ellipse = np.zeros_like(x0)
    if x_bound.ndim == 2:
        x_bound = x_bound[:, None]
        y_bound = y_bound[:, None]

    a2, b2 = a**2, b**2
    ab = a * b
    x2, y2 = x0**2, y0**2

    for i in range(x_bound.shape[1]):
        x_intersect = x_bound[:, i]
        y_intersect = y_bound[:, i]
        cosE, sinE = (x_intersect - x0[:, None]) / a, (y_intersect - y0[:, None]) / b
        cosE = np.clip(cosE, -1, 1)
        E = np.arccos(cosE)
        E[sinE < 0] *= -1.  # the formal definition [-pi, pi)
        E[E[:, 1] > E[:, 0], 0] += 2 * np.pi  # make sure E_low > E_up

        delta_E = E[:, 0] - E[:, 1]
        cosE0, cosE1 = np.cos(E[:, 0]), np.cos(E[:, 1])
        sinE0, sinE1 = np.sin(E[:, 0]), np.sin(E[:, 1])
        cos2E0, cos2E1 = np.cos(2 * E[:, 0]), np.cos(2 * E[:, 1])
        sin2E0, sin2E1 = np.sin(2 * E[:, 0]), np.sin(2 * E[:, 1])
        cos3E0, cos3E1 = np.cos(3 * E[:, 0]), np.cos(3 * E[:, 1])
        sin3E0, sin3E1 = np.sin(3 * E[:, 0]), np.sin(3 * E[:, 1])
        sin4E0, sin4E1 = np.sin(4 * E[:, 0]), np.sin(4 * E[:, 1])

        first_integral = 0.5 * ab * delta_E \
                         + 0.5 * (x_intersect[:, 0] * y_intersect[:, 0] - x_intersect[:, 1] * y_intersect[:, 1]) \
                         + 0.5 * x0 * (y_intersect[:, 0] - y_intersect[:, 1]) \
                         - 0.5 * y0 * (x_intersect[:, 0] - x_intersect[:, 1])

        second_integral = 3. / 8. * ab * (a2 + b2 + 4 * x2 + 4 * y2) * delta_E \
                          - 1.5 * a * b2 * y0 * (cosE0 - cosE1) \
                          - 1.5 * b2 * x0 * y0 * (cos2E0 - cos2E1) \
                          - 0.5 * a * b2 * y0 * (cos3E0 - cos3E1) \
                          + 0.25 * b * x0 * (9 * a2 + 3 * b2 + 4 * x2 + 12 * y2) * (sinE0 - sinE1) \
                          + 0.25 * ab * (a2 + 3 * x2 + 3 * y2) * (sin2E0 - sin2E1) \
                          + 0.25 * b * x0 * (a2 - b2) * (sin3E0 - sin3E1) \
                          + 0.03125 * ab * (a2 - 3 * b2) * (sin4E0 - sin4E1)

        delta_flux_analytic = (1 - u1 - 2 * u2) * first_integral + u2 / 3. * second_integral

        delta_angles = delta_E / n_step
        prefac = (u1 / 2. + u2) * b / a

        angle_edge = np.linspace(E[:, 1], E[:, 0], n_step + 1).T
        angle = (angle_edge[:, 1:] + angle_edge[:, :-1]) / 2.
        xp = x0[:, None] + a * np.cos(angle)
        yp = y0[:, None] + b * np.sin(angle)
        r2 = xp**2 + yp**2
        invalid_points = r2 > 1
        r2[invalid_points] = 1 - 1e-10  # Avoid insufficiently precise intersections in the case of tangency 
        mu_p = np.sqrt(1 - r2)
        integrand = (mu_p * xp + (1 - yp**2) * (np.arctan(xp / mu_p) - np.pi / 2.)) * (xp - x0[:, None])
        delta_flux_numeric = np.sum(integrand, axis=1) * delta_angles * prefac
        delta_flux_numeric += np.pi / 2 * (u1 / 2. + u2) * (y_intersect[:, 0] - y_intersect[:, 0]**3 / 3. - y_intersect[:, 1] + y_intersect[:, 1]**3 / 3.)

        delta_flux_part = np.nan_to_num(delta_flux_analytic + delta_flux_numeric)
        delta_flux_ellipse += delta_flux_part

    return delta_flux_ellipse

def star_integral(alphas, x_bound, y_bound, u1, u2):
    ''' Calculate the integral on the stellar boundary where the stellar and ellipse intersect. From bound[:, :, 0] to bound[:, :, 1]'''
    delta_flux_star = np.zeros(x_bound.shape[0])
    if x_bound.ndim == 2:
        x_bound = x_bound[:, None]
        y_bound = y_bound[:, None]
        alphas = alphas[:, None]

    delta_alpha = alphas[:, :, 1] - alphas[:, :, 0]
    x_intersect_0, x_intersect_1 = x_bound[:, :, 0], x_bound[:, :, 1]
    y_intersect_0, y_intersect_1 = y_bound[:, :, 0], y_bound[:, :, 1]

    delta_flux_limb = (0.5 - u1 / 2. - 0.75 * u2) * delta_alpha + \
                      (0.5 - u1 / 2. - 11 * u2 / 12.) * (x_intersect_1 * y_intersect_1 - x_intersect_0 * y_intersect_0) + \
                      u2 / 6. * (x_intersect_1 * y_intersect_1**3 - x_intersect_0 * y_intersect_0**3)

    flag_pn = (x_intersect_1 < 0) & (x_intersect_0 > 0)
    flag_pp = (x_intersect_1 >= 0) & (x_intersect_0 > 0)
    flag_nn = (x_intersect_1 < 0) & (x_intersect_0 <= 0)

    delta_flux_limb[flag_pn] += (u1 + 2 * u2) * np.pi / 4 * (4. / 3. - y_intersect_1[flag_pn] + y_intersect_1[flag_pn]**3 / 3. - \
                                                             y_intersect_0[flag_pn] + y_intersect_0[flag_pn]**3 / 3.)
    delta_flux_limb[flag_pp] += (u1 + 2 * u2) * np.pi / 4 * (y_intersect_1[flag_pp] - y_intersect_1[flag_pp]**3 / 3. - \
                                                             y_intersect_0[flag_pp] + y_intersect_0[flag_pp]**3 / 3.)
    delta_flux_limb[flag_nn] += (u1 + 2 * u2) * np.pi / 4 * (y_intersect_0[flag_nn] - y_intersect_0[flag_nn]**3 / 3. - \
                                                             y_intersect_1[flag_nn] + y_intersect_1[flag_nn]**3 / 3.)

    delta_flux_limb = np.nan_to_num(delta_flux_limb)
    delta_flux_star = np.sum(delta_flux_limb, axis=1)
    return delta_flux_star

def ring_integral_ring_part(x_sr, y_sr, x_pr, y_pr, parity, x0, y0, r_r, f_r, u1, u2, n_step):
    '''Calculate the ring occulted integration in the edge of rings. (outside the planet)'''
    ## parity = 1: counterclockwise; parity = -1: clockwise
    # [n, 4, 0] is the end point, [n, 4, 1] is the start point
    x_bound = np.full((x_sr.shape[0], 4, 2), np.nan)
    y_bound = np.full((x_sr.shape[0], 4, 2), np.nan)

    if np.all(np.isnan(x_pr)): # planet does not intersect with the ellipse
        flag_2 = np.sum(~np.isnan(x_sr), axis=1) == 2
        x_bound[flag_2, 0] = np.column_stack((x_sr[flag_2, 0], x_sr[flag_2, 1]))
        y_bound[flag_2, 0] = np.column_stack((y_sr[flag_2, 0], y_sr[flag_2, 1]))
        flag_4 = np.sum(~np.isnan(x_sr), axis=1) == 4
        x_bound[flag_4, 0] = np.column_stack((x_sr[flag_4, 0], x_sr[flag_4, 3]))
        y_bound[flag_4, 0] = np.column_stack((y_sr[flag_4, 0], y_sr[flag_4, 3]))
        x_bound[flag_4, 1] = np.column_stack((x_sr[flag_4, 1], x_sr[flag_4, 2]))
        y_bound[flag_4, 1] = np.column_stack((y_sr[flag_4, 1], y_sr[flag_4, 2]))
        ring_part_integral = parity * ellipse_integral(x_bound, y_bound, x0, y0, r_r, r_r*(1-f_r), u1, u2, n_step)
        
        flag_0 = np.all(np.isnan(x_sr), axis=1) & (x0**2 + y0**2 <= 1)
        ring_part_integral[flag_0] = full_occultation(np.ones_like(x0[flag_0]), x0[flag_0], y0[flag_0], r_r, f_r, u1, u2)
    else:
        flag_outp = (x_sr > x_pr[:, -1].reshape(-1, 1)) | (x_sr < x_pr[:, 0].reshape(-1, 1))
        x_sr_outp = np.where(flag_outp, x_sr, np.nan)
        y_sr_outp = np.where(flag_outp, y_sr, np.nan)

        # pr2 -> pr1
        flag_01 = (np.all(np.isnan(x_sr_outp), axis=1)) & (x_pr[:, 2]**2 + y_pr[:, 2]**2 >= 1) & (x_pr[:, 0]**2 + y_pr[:, 0]**2 < 1)
        x_bound[flag_01, 0] = np.column_stack((x_pr[flag_01, 0], x_pr[flag_01, 1]))
        y_bound[flag_01, 0] = np.column_stack((y_pr[flag_01, 0], y_pr[flag_01, 1]))

        # pr2 -> pr1 + pr3 -> pr4
        flag_02 = (np.all(np.isnan(x_sr_outp), axis=1)) & (x_pr[:, 3]**2 + y_pr[:, 3]**2 <= 1)
        x_bound[flag_02, 0] = np.column_stack((x_pr[flag_02, 0], x_pr[flag_02, 1]))
        y_bound[flag_02, 0] = np.column_stack((y_pr[flag_02, 0], y_pr[flag_02, 1]))
        x_bound[flag_02, 1] = np.column_stack((x_pr[flag_02, 3], x_pr[flag_02, 2]))
        y_bound[flag_02, 1] = np.column_stack((y_pr[flag_02, 3], y_pr[flag_02, 2]))

        # sr2 -> pr1
        flag_10 = (np.sum(~np.isnan(x_sr_outp), axis=1) == 1) & (x_sr_outp[:, 1] < x_pr[:, 0])
        x_bound[flag_10, 0] = np.column_stack((x_pr[flag_10, 0], x_sr_outp[flag_10, 1]))
        y_bound[flag_10, 0] = np.column_stack((y_pr[flag_10, 0], y_sr_outp[flag_10, 1]))

        # pr2 -> pr1 + pr3 -> sr1
        flag_11 = (np.sum(~np.isnan(x_sr_outp), axis=1) == 1) & (x_sr_outp[:, 0] > x_pr[:, -1])
        x_bound[flag_11, 0] = np.column_stack((x_pr[flag_11, 0], x_pr[flag_11, 1]))
        y_bound[flag_11, 0] = np.column_stack((y_pr[flag_11, 0], y_pr[flag_11, 1]))
        x_bound[flag_11, 1] = np.column_stack((x_sr_outp[flag_11, 0], x_pr[flag_11, 2]))
        y_bound[flag_11, 1] = np.column_stack((y_sr_outp[flag_11, 0], y_pr[flag_11, 2]))

        # sr2 -> sr1
        flag_20 = (np.sum(~np.isnan(x_sr_outp), axis=1) == 2) & (x_sr_outp[:, 0] <= x_pr[:, 0]) & (x_sr_outp[:, 1] <= x_pr[:, 0])
        x_bound[flag_20, 0] = np.column_stack((x_sr_outp[flag_20, 0], x_sr_outp[flag_20, 1]))
        y_bound[flag_20, 0] = np.column_stack((y_sr_outp[flag_20, 0], y_sr_outp[flag_20, 1]))

        # pr3 -> sr1 + sr2 -> pr1
        flag_21 = (np.sum(~np.isnan(x_sr_outp), axis=1) == 2) & (x_sr_outp[:, 0] > x_pr[:, -1]) & (x_sr_outp[:, 1] < x_pr[:, 0])
        x_bound[flag_21, 0] = np.column_stack((x_sr_outp[flag_21, 0], x_pr[flag_21, 2]))
        y_bound[flag_21, 0] = np.column_stack((y_sr_outp[flag_21, 0], y_pr[flag_21, 2]))
        x_bound[flag_21, 1] = np.column_stack((x_pr[flag_21, 0], x_sr_outp[flag_21, 1]))
        y_bound[flag_21, 1] = np.column_stack((y_pr[flag_21, 0], y_sr_outp[flag_21, 1]))

        # pr2 -> pr1 + sr2 -> pr4 + pr3 -> sr1
        flag_22 = (np.sum(~np.isnan(x_sr_outp), axis=1) == 2) & (x_sr_outp[:, 0] > x_pr[:, -1]) & (x_sr_outp[:, 1] > x_pr[:, -1])
        x_bound[flag_22, 0] = np.column_stack((x_pr[flag_22, 0], x_pr[flag_22, 1]))
        y_bound[flag_22, 0] = np.column_stack((y_pr[flag_22, 0], y_pr[flag_22, 1]))
        x_bound[flag_22, 1] = np.column_stack((x_pr[flag_22, 3], x_sr_outp[flag_22, 1]))
        y_bound[flag_22, 1] = np.column_stack((y_pr[flag_22, 3], y_sr_outp[flag_22, 1]))
        x_bound[flag_22, 2] = np.column_stack((x_sr_outp[flag_22, 0], x_pr[flag_22, 2]))
        y_bound[flag_22, 2] = np.column_stack((y_sr_outp[flag_22, 0], y_pr[flag_22, 2]))

        # pr3 -> sr1 + sr2 -> pr4 + pr2 -> sr3 + sr4 -> pr1
        flag_4 = np.sum(~np.isnan(x_sr_outp), axis=1) == 4
        x_bound[flag_4, 0] = np.column_stack((x_sr_outp[flag_4, 0], x_pr[flag_4, 2]))
        y_bound[flag_4, 0] = np.column_stack((y_sr_outp[flag_4, 0], y_pr[flag_4, 2]))
        x_bound[flag_4, 1] = np.column_stack((x_pr[flag_4, 3], x_sr_outp[flag_4, 1]))
        y_bound[flag_4, 1] = np.column_stack((y_pr[flag_4, 3], y_sr_outp[flag_4, 1]))
        x_bound[flag_4, 2] = np.column_stack((x_sr_outp[flag_4, 2], x_pr[flag_4, 1]))
        y_bound[flag_4, 2] = np.column_stack((y_sr_outp[flag_4, 2], y_pr[flag_4, 1]))
        x_bound[flag_4, 3] = np.column_stack((x_pr[flag_4, 0], x_sr_outp[flag_4, 3]))
        y_bound[flag_4, 3] = np.column_stack((y_pr[flag_4, 0], y_sr_outp[flag_4, 3]))

        # ellipse_integral integrates from x_bound[:, :, 1] to x_bound[:, :, 0]
        ring_part_integral = parity * ellipse_integral(x_bound, y_bound, x0, y0, r_r, r_r*(1-f_r), u1, u2, n_step)
    return ring_part_integral

def ring_integral_star_part(x_sp, y_sp, alpha_sp, x_sri, y_sri, alpha_sri, x_sro, y_sro, alpha_sro, x0, y0, r_in, r_out, r_p, f_r, u1, u2):
    '''Calculate the ring occulted integration in the edge of stars. (inside the ring and outside the planet)'''
    # The intersections of the star and the ring that outside the planet
    flag_sri_outp = (x_sri-x0[:, None])**2 + (y_sri-y0[:, None])**2 > r_p**2
    x_sri1 = np.where(flag_sri_outp, x_sri, np.nan)
    y_sri1 = np.where(flag_sri_outp, y_sri, np.nan)
    alpha_sri1 = np.where(flag_sri_outp, alpha_sri, np.nan)
    flag_sro_outp = (x_sro-x0[:, None])**2 + (y_sro-y0[:, None])**2 > r_p**2
    x_sro1 = np.where(flag_sro_outp, x_sro, np.nan)
    y_sro1 = np.where(flag_sro_outp, y_sro, np.nan)
    alpha_sro1 = np.where(flag_sro_outp, alpha_sro, np.nan)

    # The intersections of the star and the planet that inside the ring
    flag_sp_inr = ((x_sp-x0[:, None])**2 + (y_sp-y0[:, None])**2/(1-f_r)**2 > r_in**2) & ((x_sp-x0[:, None])**2 + (y_sp-y0[:, None])**2/(1-f_r)**2 < r_out**2)
    x_sp1 = np.where(flag_sp_inr, x_sp, np.nan)
    y_sp1 = np.where(flag_sp_inr, y_sp, np.nan)
    alpha_sp1 = np.where(flag_sp_inr, alpha_sp, np.nan)

    # Arrange the intersections in alpha order from smallest to largest
    alpha_index = np.argsort(np.hstack((alpha_sp1, alpha_sri1, alpha_sro1)), axis=1)
    x_st = np.hstack((x_sp1, x_sri1, x_sro1))
    y_st = np.hstack((y_sp1, y_sri1, y_sro1))
    alpha_st = np.hstack((alpha_sp1, alpha_sri1, alpha_sro1))
    x_bound = np.take_along_axis(x_st, alpha_index, axis=1).reshape(x_st.shape[0], -1, 2)
    y_bound = np.take_along_axis(y_st, alpha_index, axis=1).reshape(x_st.shape[0], -1, 2)
    alpha_bound = np.take_along_axis(alpha_st, alpha_index, axis=1).reshape(x_st.shape[0], -1, 2)

    # star_integral integrates from x_bound[:, :, 0] to x_bound[:, :, 1]
    star_part_integral = star_integral(alpha_bound, x_bound, y_bound, u1, u2)
    return star_part_integral

def ring_integral_planet_part(x_sp, y_sp, x_pri, y_pri, x_pro, y_pro, x0, y0, r_in, r_out, r_p, f_r, u1, u2, n_step):
    '''Calculate the ring occulted integration in the edge of planets. (inside the rings)'''
    # The intersections of the plent and the ring that inside the star
    flag_pri_ins = x_pri**2 + y_pri**2 < 1
    x_pri1 = np.where(flag_pri_ins, x_pri, np.nan)
    y_pri1 = np.where(flag_pri_ins, y_pri, np.nan)
    if np.all(np.isnan(x_pro)):
        x_pro1 = np.full((x0.shape[0], 2), np.nan)
        y_pro1 = np.full((x0.shape[0], 2), np.nan)
    else:
        flag_pro_ins = x_pro**2 + y_pro**2 < 1
        x_pro1 = np.where(flag_pro_ins, x_pro, np.nan)
        y_pro1 = np.where(flag_pro_ins, y_pro, np.nan)
    
    # The intersections of the star and the planet that inside the ring
    flag_sp_inr = ((x_sp-x0[:, None])**2 + (y_sp-y0[:, None])**2/(1-f_r)**2 > r_in**2) & ((x_sp-x0[:, None])**2 + (y_sp-y0[:, None])**2/(1-f_r)**2 < r_out**2)
    x_sp1 = np.where(flag_sp_inr, x_sp, np.nan)
    y_sp1 = np.where(flag_sp_inr, y_sp, np.nan)
    
    # Arrange the intersections in alpha order from smallest to largest
    x_pt = np.hstack((x_sp1, x_pri1, x_pro1))
    y_pt = np.hstack((y_sp1, y_pri1, y_pro1))
    cosE, sinE = (x_pt-x0[:, None])/r_p, (y_pt-y0[:, None])/r_p
    cosE[cosE<-1] = -1
    cosE[cosE>1] = 1
    E = np.arccos(cosE)
    E[sinE<0] *= -1. # the formal definition [-pi, pi)
    E_index = np.argsort(E, axis=1)[:, ::-1] # from largest to smallest
    x_bound = np.take_along_axis(x_pt, E_index, axis=1).reshape(x_pt.shape[0], -1, 2)
    y_bound = np.take_along_axis(y_pt, E_index, axis=1).reshape(x_pt.shape[0], -1, 2)

    # ellipse_integral integrates from x_bound[:, :, 1] to x_bound[:, :, 0]
    planet_part_integral = ellipse_integral(x_bound, y_bound, x0, y0, r_p, r_p, u1, u2, n_step)
    return -planet_part_integral

def ring_lc(transit_parameters, time_array, n_step=300):
    """
    Compute the trensit light curve of a spherical planet with a ring.

    Parameters:
    -----------
    transit_parameters : list
        A list of parameters for the model:
        - t_0 : float
            Time of the transit center.
        - b_0 : float
            Impact parameter.
        - period : float
            Orbital period of the planet.
        - r_p : float
            Radius of the planet.
        - r_in : float
            Inner radius of the ring.
        - r_out : float
            Outer radius of the ring.
        - f_r : float
            Flattening factor of the ring.
        - obliquity : float
            Obliquity of the ring.
        - u1 : float
            Linear limb-darkening coefficient.
        - u2 : float
            Quadratic limb-darkening coefficient.
        - log10_rho_star : float
            Logarithm (base 10) of the stellar density.
        - opacity : float
            Opacity of the ring.
    time_array : array_like
        Array of time points at which to compute the light curve.
    n_step : int, optional
        Number of steps for numerical integration (default is 300).
    
    Returns:
    --------
    flux_array : array_like
        Array of flux values corresponding to the input time points.
    """

    t_0, b_0, period, r_p, r_in, r_out, f_r, obliquity, u1, u2, log10_rho_star, opacity = transit_parameters
    if r_in > r_out:
        raise IOError('The radius of the inner ring should be smaller than the outer ring')
    if r_in < r_p:
        raise IOError('The radius of the planet should not be larger than the inner ring')
    
    a_over_rstar = 3.753*(period**2*10**log10_rho_star)**(1./3.)
    x0_ini = (time_array-t_0)/period*2*np.pi*a_over_rstar
    y0_ini = np.ones_like(x0_ini, dtype=float)*b_0
    x0 = x0_ini*np.cos(obliquity) + y0_ini*np.sin(obliquity)
    y0 =-x0_ini*np.sin(obliquity) + y0_ini*np.cos(obliquity)
    x0[x0<0] *= -1
    y0[y0<0] *= -1

    pars_in = [t_0, b_0, period, r_in, f_r, obliquity, u1, u2, log10_rho_star]
    pars_out = [t_0, b_0, period, r_out, f_r, obliquity, u1, u2, log10_rho_star]
    pars_p = [t_0, b_0, period, r_p, 0, 0, u1, u2, log10_rho_star]

    planet_part, contacts_p = spherical_lc(pars_p, time_array)
    planet_part = 1 - planet_part

    flags_sp, x_sp, y_sp, alphas_sp = solve_star_ellipse_intersections(x0, y0, r_p, 0., 1e-8)
    flags_sri, x_sri, y_sri, alphas_sri = solve_star_ellipse_intersections(x0, y0, r_in, f_r, 1e-8)
    flags_sro, x_sro, y_sro, alphas_sro = solve_star_ellipse_intersections(x0, y0, r_out, f_r, 1e-8)
    
    flux_total = np.pi*(6-2*u1-u2)/6.

    # planet has no intersection with the ring
    if r_p<=r_in*(1-f_r):
        ring_out_part, contacts_rout = oblate_lc(pars_out, time_array, n_step=n_step)
        ring_out_part = 1 - ring_out_part
        ring_in_part, contacts_rin = oblate_lc(pars_in, time_array, n_step=n_step)
        ring_in_part = 1 - ring_in_part
        flux_array = 1 - (planet_part + opacity * (ring_out_part - ring_in_part))
    # planet has intersections with the ring
    else:
        x_pri, y_pri, x_pro, y_pro = solve_planet_ring_intersections(x0, y0, r_p, r_in, r_out, f_r)
        ring_part = ring_integral_ring_part(x_sri, y_sri, x_pri, y_pri, -1, x0, y0, r_in, f_r, u1, u2, n_step) # inner ring
        ring_part += ring_integral_ring_part(x_sro, y_sro, x_pro, y_pro, 1, x0, y0, r_out, f_r, u1, u2, n_step) # outer ring
        ring_part += ring_integral_planet_part(x_sp, y_sp, x_pri, y_pri, x_pro, y_pro, x0, y0, r_in, r_out, r_p, f_r, u1, u2, n_step)
        ring_part += ring_integral_star_part(x_sp, y_sp, alphas_sp, x_sri, y_sri, alphas_sri, x_sro, y_sro, alphas_sro, x0, y0, r_in, r_out, r_p, f_r, u1, u2)
        flux_array =  1 - (planet_part + opacity * ring_part / flux_total)
    return flux_array


if __name__=='__main__':
    # perform a test with Kepler-51d's parameters (Liu et al. 2024)
    t_0, b_0, period = 0, 0.003, 130.1845
    r_p, r_ri, r_ro, obliquity = 0.0977, 0.11, 0.13, np.pi/4
    u_1, u_2 = 0.208, 0.138
    log10_rho_star = 0.32 # rho_star in unit g/cc
    opacity = 0.5

    for f in [0., 0.2, 0.4]:
        transit_parameters = [t_0, b_0, period, r_p, r_ri, r_ro, f, obliquity, u_1, u_2, log10_rho_star, opacity]
        time_array = np.linspace(-0.1, 0.1, 1000)
        time0 = time.time()
        flux_array = ring_lc(transit_parameters, time_array)
        time1 = time.time()
        print('Time cost when f_r = %.1f: %.5f s' % (f, time1-time0))

