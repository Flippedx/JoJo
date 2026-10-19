"""Utilities for parameterization and posterior diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np


MODEL_PARAMETER_NAMES = (
    "tc",
    "b",
    "period",
    "rp_me",
    "oblateness",
    "obliquity",
    "ecc",
    "omega",
    "u1",
    "u2",
    "log10_rho_star",
)

SAMPLING_PARAMETER_NAMES = (
    "tc",
    "b",
    "rp_me",
    "fcos2ob",
    "fsin2ob",
    "ecosw",
    "esinw",
    "q1",
    "q2",
    "log10_rho_star",
)


def _as_parameter_dict(parameters, names):
    if isinstance(parameters, dict):
        return parameters
    array = np.asarray(parameters)
    if array.shape[-1] != len(names):
        raise ValueError(f"Expected last dimension {len(names)}, got shape {array.shape}.")
    return {name: array[..., index] for index, name in enumerate(names)}


def _stack_parameter_dict(parameters, names):
    return np.stack([np.asarray(parameters[name]) for name in names], axis=-1)


def _logit(x):
    x = np.asarray(x, dtype=float)
    return np.log(x) - np.log1p(-x)


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def limb_darkening_to_q(u1, u2, clip=True):
    """Convert quadratic limb-darkening coefficients to Kipping q1/q2."""
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    sum_u = u1 + u2
    safe_sum = np.where(np.abs(sum_u) > 1e-15, sum_u, np.nan)
    q1 = sum_u**2
    q2 = u1 / (2.0 * safe_sum)
    if clip:
        q1 = np.clip(q1, 0.0, 1.0)
        q2 = np.clip(q2, 0.0, 1.0)
    return q1, q2


def q_to_limb_darkening(q1, q2, clip=True):
    """Convert Kipping q1/q2 variables to quadratic limb darkening u1/u2."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    if clip:
        q1 = np.clip(q1, 0.0, 1.0)
        q2 = np.clip(q2, 0.0, 1.0)
    sqrt_q1 = np.sqrt(q1)
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return u1, u2


def native_shape_to_components(oblateness, obliquity, ecc, omega):
    """Map native angular shape/orbit parameters to nonsingular components."""
    oblateness = np.asarray(oblateness, dtype=float)
    obliquity = np.asarray(obliquity, dtype=float)
    ecc = np.asarray(ecc, dtype=float)
    omega = np.asarray(omega, dtype=float)
    return {
        "fcos2ob": oblateness * np.cos(2.0 * obliquity),
        "fsin2ob": oblateness * np.sin(2.0 * obliquity),
        "ecosw": ecc * np.cos(omega),
        "esinw": ecc * np.sin(omega),
    }


def components_to_native_shape(fcos2ob, fsin2ob, ecosw, esinw, radius_epsilon=0.0, clip=False):
    """Map nonsingular components back to native f/obliquity/e/omega."""
    fcos2ob = np.asarray(fcos2ob, dtype=float)
    fsin2ob = np.asarray(fsin2ob, dtype=float)
    ecosw = np.asarray(ecosw, dtype=float)
    esinw = np.asarray(esinw, dtype=float)
    oblateness = np.sqrt(fcos2ob**2 + fsin2ob**2 + radius_epsilon**2) - radius_epsilon
    ecc = np.sqrt(ecosw**2 + esinw**2 + radius_epsilon**2) - radius_epsilon
    obliquity = 0.5 * np.arctan2(fsin2ob, fcos2ob)
    omega = np.arctan2(esinw, ecosw)
    if clip:
        oblateness = np.clip(oblateness, 0.0, 0.95)
        ecc = np.clip(ecc, 0.0, 0.95)
    return oblateness, obliquity, ecc, omega


def model_to_sampling_parameters(model_parameters, b_max=0.95, clip=True, as_dict=True):
    """Convert JoJo's native 11-parameter vector to HMC-friendly variables.

    Native order is ``MODEL_PARAMETER_NAMES``. The sampling variables are
    ``tc, raw_b, log_rp_me, fcos2ob, fsin2ob, ecosw, esinw, q1, q2,
    log10_rho_star``.
    """
    params = _as_parameter_dict(model_parameters, MODEL_PARAMETER_NAMES)
    b_scaled = np.asarray(params["b"], dtype=float) / b_max
    rp_me = np.asarray(params["rp_me"], dtype=float)
    if clip:
        b_scaled = np.clip(b_scaled, 1e-12, 1.0 - 1e-12)
        rp_me = np.clip(rp_me, 1e-300, None)
    components = native_shape_to_components(
        params["oblateness"],
        params["obliquity"],
        params["ecc"],
        params["omega"],
    )
    q1, q2 = limb_darkening_to_q(params["u1"], params["u2"], clip=clip)
    sampling = {
        "tc": params["tc"],
        "raw_b": _logit(b_scaled),
        "log_rp_me": np.log(rp_me),
        "fcos2ob": components["fcos2ob"],
        "fsin2ob": components["fsin2ob"],
        "ecosw": components["ecosw"],
        "esinw": components["esinw"],
        "q1": q1,
        "q2": q2,
        "log10_rho_star": params["log10_rho_star"],
    }
    if as_dict:
        return sampling
    return _stack_parameter_dict(sampling, SAMPLING_PARAMETER_NAMES)


def sampling_to_model_parameters(
    sampling_parameters,
    period=None,
    base_model_parameters=None,
    b_max=0.95,
    radius_epsilon=0.0,
    clip=True,
    as_dict=False,
):
    """Convert HMC-friendly variables back to JoJo's native model parameters."""
    sampling = _as_parameter_dict(sampling_parameters, SAMPLING_PARAMETER_NAMES)
    if period is None:
        if base_model_parameters is None:
            raise ValueError("Pass either period or base_model_parameters.")
        period = _as_parameter_dict(base_model_parameters, MODEL_PARAMETER_NAMES)["period"]

    oblateness, obliquity, ecc, omega = components_to_native_shape(
        sampling["fcos2ob"],
        sampling["fsin2ob"],
        sampling["ecosw"],
        sampling["esinw"],
        radius_epsilon=radius_epsilon,
        clip=clip,
    )
    u1, u2 = q_to_limb_darkening(sampling["q1"], sampling["q2"], clip=clip)
    model = {
        "tc": sampling["tc"],
        "b": b_max * _sigmoid(sampling["raw_b"]),
        "period": period,
        "rp_me": np.exp(sampling["log_rp_me"]),
        "oblateness": oblateness,
        "obliquity": obliquity,
        "ecc": ecc,
        "omega": omega,
        "u1": u1,
        "u2": u2,
        "log10_rho_star": sampling["log10_rho_star"],
    }
    if as_dict:
        return model
    return _stack_parameter_dict(model, MODEL_PARAMETER_NAMES)


def samples_to_oblateness_obliquity(samples):
    """Return oblateness and obliquity arrays from native or component samples."""
    if isinstance(samples, dict):
        params = samples
    else:
        samples = np.asarray(samples)
        if samples.shape[-1] == len(MODEL_PARAMETER_NAMES):
            params = _as_parameter_dict(samples, MODEL_PARAMETER_NAMES)
        elif samples.shape[-1] == len(SAMPLING_PARAMETER_NAMES):
            params = _as_parameter_dict(samples, SAMPLING_PARAMETER_NAMES)
        else:
            raise ValueError(
                "Array samples must use either native model order "
                f"({len(MODEL_PARAMETER_NAMES)} columns) or sampling order "
                f"({len(SAMPLING_PARAMETER_NAMES)} columns)."
            )
    if "oblateness" in params and "obliquity" in params:
        oblateness = np.asarray(params["oblateness"], dtype=float)
        obliquity = np.asarray(params["obliquity"], dtype=float)
    elif "fcos2ob" in params and "fsin2ob" in params:
        oblateness, obliquity, _, _ = components_to_native_shape(
            params["fcos2ob"],
            params["fsin2ob"],
            np.zeros_like(params["fcos2ob"], dtype=float),
            np.zeros_like(params["fsin2ob"], dtype=float),
        )
    else:
        raise ValueError("Samples must contain oblateness/obliquity or fcos2ob/fsin2ob.")
    return np.ravel(oblateness), np.mod(np.ravel(obliquity), np.pi)


def plot_residual_diagnostics(
    time,
    flux,
    flux_err=None,
    model_flux=None,
    model_parameters=None,
    model_fn=None,
    clean_flux=None,
    output_path=None,
    show=False,
    title=None,
):
    """Plot data/model and residual diagnostics.

    If ``model_flux`` is not provided, pass ``model_parameters`` and a
    ``model_fn``. The model function may return either ``flux`` or
    ``(flux, extra)``.
    """
    import matplotlib.pyplot as plt

    time = np.asarray(time)
    flux = np.asarray(flux)
    if model_flux is None:
        if model_parameters is None or model_fn is None:
            raise ValueError("Pass model_flux, or both model_parameters and model_fn.")
        model_result = model_fn(model_parameters, time)
        model_flux = model_result[0] if isinstance(model_result, tuple) else model_result
    model_flux = np.asarray(model_flux)
    residual_ppm = 1e6 * (flux - model_flux)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=False, gridspec_kw={"height_ratios": [2, 1, 1]})
    if flux_err is None:
        axes[0].plot(time, flux, ".", ms=3, alpha=0.45, color="0.35", label="data")
        axes[1].plot(time, residual_ppm, ".", ms=3, alpha=0.45, color="0.35")
    else:
        flux_err = np.asarray(flux_err)
        error_ppm = 1e6 * flux_err
        axes[0].errorbar(time, flux, yerr=flux_err, fmt=".", ms=3, alpha=0.45, color="0.35", label="data")
        axes[1].errorbar(time, residual_ppm, yerr=error_ppm, fmt=".", ms=3, alpha=0.45, color="0.35")
    if clean_flux is not None:
        axes[0].plot(time, clean_flux, color="C2", lw=1.3, label="reference")
    axes[0].plot(time, model_flux, color="C3", lw=1.4, label="model")
    axes[0].set_ylabel("relative flux")
    axes[0].legend(loc="best")
    if title:
        axes[0].set_title(title)

    axes[1].axhline(0.0, color="0.2", lw=1)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("residual [ppm]")

    axes[2].hist(residual_ppm[np.isfinite(residual_ppm)], bins=40, histtype="stepfilled", alpha=0.7, color="0.35")
    axes[2].axvline(0.0, color="0.2", lw=1)
    axes[2].set_xlabel("residual [ppm]")
    axes[2].set_ylabel("count")
    rms_ppm = np.sqrt(np.nanmean(residual_ppm**2))
    axes[2].text(0.98, 0.90, f"RMS = {rms_ppm:.2f} ppm", ha="right", va="top", transform=axes[2].transAxes)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    return fig, axes


def plot_oblateness_obliquity_polar_corner(
    samples,
    truth=None,
    output_path=None,
    show=False,
    max_points=5000,
    random_seed=0,
):
    """Plot a polar corner-style diagnostic for oblateness and obliquity."""
    import matplotlib.pyplot as plt

    oblateness, obliquity = samples_to_oblateness_obliquity(samples)
    finite = np.isfinite(oblateness) & np.isfinite(obliquity)
    oblateness = oblateness[finite]
    obliquity = obliquity[finite]
    if len(oblateness) == 0:
        raise ValueError("No finite oblateness/obliquity samples to plot.")

    if len(oblateness) > max_points:
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(len(oblateness), size=max_points, replace=False)
        plot_f = oblateness[keep]
        plot_ob = obliquity[keep]
    else:
        plot_f = oblateness
        plot_ob = obliquity

    fig = plt.figure(figsize=(7.5, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.2), height_ratios=(1.2, 4), hspace=0.05, wspace=0.05)
    ax_ob = fig.add_subplot(gs[0, 0])
    ax_blank = fig.add_subplot(gs[0, 1])
    ax_polar = fig.add_subplot(gs[1, 0], projection="polar")
    ax_f = fig.add_subplot(gs[1, 1], sharey=ax_polar)

    ax_ob.hist(obliquity, bins=40, color="0.35", alpha=0.75)
    ax_ob.set_xlim(0.0, np.pi)
    ax_ob.set_ylabel("count")
    ax_ob.tick_params(labelbottom=False)

    ax_polar.scatter(plot_ob, plot_f, s=5, alpha=0.18, linewidths=0, color="C0")
    ax_polar.set_thetamin(0)
    ax_polar.set_thetamax(180)
    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(1)
    ax_polar.set_xlabel("obliquity")
    ax_polar.set_ylabel("oblateness", labelpad=18)

    ax_f.hist(oblateness, bins=40, orientation="horizontal", color="0.35", alpha=0.75)
    ax_f.tick_params(labelleft=False)
    ax_f.set_xlabel("count")

    if truth is not None:
        truth_f, truth_ob = samples_to_oblateness_obliquity(truth)
        truth_f = float(np.ravel(truth_f)[0])
        truth_ob = float(np.ravel(truth_ob)[0])
        ax_ob.axvline(truth_ob, color="C3", lw=1.5)
        ax_f.axhline(truth_f, color="C3", lw=1.5)
        ax_polar.scatter([truth_ob], [truth_f], color="C3", s=35, zorder=5)

    ax_blank.axis("off")
    fig.suptitle("Oblateness-obliquity posterior")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    return fig, (ax_ob, ax_polar, ax_f)
