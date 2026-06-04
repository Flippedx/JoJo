__version__ = '0.1.0'

__all__ = [
    'oblate_lc',
    'spherical_lc',
    'full_occultation',
    'solve_star_ellipse_intersections',
    'ring_lc',
    'solve_planet_ring_intersections',
    'jax_oblate_lc',
    'flux_jacobian',
    'fisher_matrix',
    'model_to_sampling_parameters',
    'sampling_to_model_parameters',
    'limb_darkening_to_q',
    'q_to_limb_darkening',
    'plot_residual_diagnostics',
    'plot_oblateness_obliquity_polar_corner',
]

_EXPORTS = {
    'oblate_lc': ('jojo_oblate', 'oblate_lc'),
    'spherical_lc': ('jojo_oblate', 'spherical_lc'),
    'full_occultation': ('jojo_oblate', 'full_occultation'),
    'solve_star_ellipse_intersections': ('jojo_oblate', 'solve_star_ellipse_intersections'),
    'ring_lc': ('jojo_ring', 'ring_lc'),
    'solve_planet_ring_intersections': ('jojo_ring', 'solve_planet_ring_intersections'),
    'jax_oblate_lc': ('jojo_oblate_jax', 'jax_oblate_lc'),
    'flux_jacobian': ('jojo_oblate_jax', 'flux_jacobian'),
    'fisher_matrix': ('jojo_oblate_jax', 'fisher_matrix'),
    'model_to_sampling_parameters': ('utils', 'model_to_sampling_parameters'),
    'sampling_to_model_parameters': ('utils', 'sampling_to_model_parameters'),
    'limb_darkening_to_q': ('utils', 'limb_darkening_to_q'),
    'q_to_limb_darkening': ('utils', 'q_to_limb_darkening'),
    'plot_residual_diagnostics': ('utils', 'plot_residual_diagnostics'),
    'plot_oblateness_obliquity_polar_corner': ('utils', 'plot_oblateness_obliquity_polar_corner'),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, object_name = _EXPORTS[name]
    module = __import__(f'{__name__}.{module_name}', fromlist=[object_name])
    value = getattr(module, object_name)
    globals()[name] = value
    return value
