__version__ = '0.1.0'

__all__ = [
    'oblate_lc',
    'spherical_lc',
    'full_occultation',
    'solve_star_ellipse_intersections',
    'ring_lc',
    'solve_planet_ring_intersections',
]

_EXPORTS = {
    'oblate_lc': ('jojo_oblate', 'oblate_lc'),
    'spherical_lc': ('jojo_oblate', 'spherical_lc'),
    'full_occultation': ('jojo_oblate', 'full_occultation'),
    'solve_star_ellipse_intersections': ('jojo_oblate', 'solve_star_ellipse_intersections'),
    'ring_lc': ('jojo_ring', 'ring_lc'),
    'solve_planet_ring_intersections': ('jojo_ring', 'solve_planet_ring_intersections'),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, object_name = _EXPORTS[name]
    module = __import__(f'{__name__}.{module_name}', fromlist=[object_name])
    value = getattr(module, object_name)
    globals()[name] = value
    return value
