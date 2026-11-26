__version__ = "0.2.0"

__all__ = [
    "oblate_lc",
    "spherical_lc",
    "ring_lc",
    "egg_lc",
    "plot_shape",
]

_EXPORTS = {
    "oblate_lc": ("models.oblate", "oblate_lc"),
    "spherical_lc": ("models.oblate", "spherical_lc"),
    "ring_lc": ("models.ring", "ring_lc"),
    "egg_lc": ("models.egg", "egg_lc"),
    "plot_shape": ("plotting", "plot_shape"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, object_name = _EXPORTS[name]
    module = __import__(f"{__name__}.{module_name}", fromlist=[object_name])
    value = getattr(module, object_name)
    globals()[name] = value
    return value
