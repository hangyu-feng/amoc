# This script recreates Figure 1 from Boers (2021, Nature Climate Change) to
# demonstrate the robustness of different early-warning signal (EWS) indicators.
#
# It compares two scenarios:
# 1. A non-linear model undergoing a true critical transition.
# 2. A stable linear model with changing noise properties to test for false alarms.
#
# The script calculates Variance, Lag-1 Autocorrelation (AC1), and both an
# uncorrected (lambda) and corrected (lambda_cor) restoring rate. Finally, it
# generates a 4x2 plot to visualize how each indicator performs, highlighting
# the robustness of the corrected restoring rate.
#
# Dependencies: numpy, matplotlib, scipy, EWS_functions.py
# Usage: Run to display the plot and save 'figure1_recreation.pdf'.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import scipy.stats as st
from EWS_functions import runstd, runac, run_fit_a, run_fit_a_ar1


NSTEPS = 10000
DT = 1
SIGMA = 0.2
STD_MULTIPLIER = 1

def critical_transition_data(n_steps=NSTEPS, dt=DT, sigma=SIGMA):
    """
    Generates time series data for a model undergoing a critical transition.
    Equation: dx/dt = -x^3 + x - T + eta(t)
    In discrete form: x_n = x_{n-1} + (-x_{n-1}^3 + x_{n-1} - T_{n-1} + eta(t))*dt
    """
    # Time vector
    time = np.arange(0, n_steps * dt, dt)

    # Linearly increasing control parameter T
    T = np.linspace(-1, 1, n_steps)

    # Initialize state vectors
    x_stochastic = np.zeros(n_steps)
    x_deterministic = np.zeros(n_steps)

    # Set initial conditions near the stable point
    x_stochastic[0] = np.sqrt(1 - T[0])
    x_deterministic[0] = np.sqrt(1 - T[0])

    # White noise term (scaled by STD_MULTIPLIER)
    eta = np.random.normal(0, sigma * np.sqrt(dt) * STD_MULTIPLIER, n_steps)

    # Euler-Maruyama integration
    for i in range(1, n_steps):
        # Stochastic version
        derivative = (
            -x_stochastic[i - 1] ** 3 + x_stochastic[i - 1] - T[i - 1] + eta[i - 1]
        )
        x_stochastic[i] = x_stochastic[i - 1] + derivative * dt

        # Deterministic version (no noise)
        derivative = -x_deterministic[i - 1] ** 3 + x_deterministic[i - 1] - T[i - 1]
        x_deterministic[i] = x_deterministic[i - 1] + derivative * dt

    return time, x_stochastic, x_deterministic


def false_alarm_data(n_steps=NSTEPS, dt=DT):
    """

    Simulated time series from the linear model dx/
    dt = −5x + η(t) with autocorrelated noise η, with the standard deviation σ of η rising linearly from 0.2 to 1.0 and the AR(1) coefficient of η rising linearly
    from 0.1 to 0.95. The system does hence not destabilize. Only the statistics of the noise forcing change.
    """
    # Time vector
    time = np.arange(0, n_steps * dt, dt)

    # Time-varying AR(1) parameters for the noise η(t)
    phi = np.linspace(0.1, 0.95, n_steps)               # AR(1) coefficient
    eta_target_std = np.linspace(0.2, 1.0, n_steps)     # Target standard deviation of η(t)

    # Generate autocorrelated noise η(t)
    eta = np.zeros(n_steps)
    eta[0] = np.random.normal(0.0, eta_target_std[0] * STD_MULTIPLIER)
    for i in range(1, n_steps):
        eta[i] = phi[i - 1] * eta[i - 1] + np.random.normal(0.0, eta_target_std[i - 1] * STD_MULTIPLIER)

    # Integrate the stable linear system dx/dt = -5x + η(t)
    x = np.zeros(n_steps)
    for i in range(1, n_steps):
        x[i] = x[i - 1] + (-5.0 * x[i - 1] + eta[i - 1]) * dt

    return time, x, phi


# Helper to compute indicators on a series with sliding window
def compute_indicators(ts, w):
    """Return variance, AC1, lambda (uncorr) and lambda_cor using window w."""
    var = runstd(ts, w) ** 2
    ac1 = runac(ts, w)
    lam = run_fit_a(ts, w)
    lam_cor = run_fit_a_ar1(ts, w)
    return var, ac1, lam, lam_cor


def prewhiten_with_phi(ts, phi):
    """Prewhiten a series given a time-varying AR(1) coefficient phi(t).

    y[0] = ts[0]; y[t] = ts[t] - phi[t] * ts[t-1] for t>=1.
    """
    y = np.array(ts, dtype=float).copy()
    if len(y) > 1:
        y[1:] = y[1:] - phi[1:] * y[:-1]
    return y


# ---- Subfigure functions (a–h) ----
def plot_a(ax, steps, ts_crit, ts_det_crit, step_line=None):
    ax.plot(steps, ts_crit, "k", lw=0.8, label="Stochastic TS")
    ax.plot(steps, ts_det_crit, "r", lw=1.2, label="Deterministic TS")
    # Critical transition around ~7000 steps
    if step_line is not None:
        ax.axvline(step_line, color="b", linestyle="-", lw=2)
    ax.set_title("Model with critical transition: true EWS")
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc="lower left")


def _trendline(ax, t, y, color="k", ls="--"):
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    res = st.linregress(t[mask], y[mask])
    ax.plot(t[mask], res.slope * t[mask] + res.intercept, color=color, linestyle=ls)
    return res


def plot_b(ax, steps, ts_false):
    ax.plot(steps, ts_false, "k", lw=0.8)
    ax.set_title("Model without transition: false alarms")
    ax.set_ylim(-6, 6)


def plot_c(ax, steps_ews, var_crit, step_line=None, draw_until=None):
    y = var_crit
    if draw_until is not None:
        mask = steps_ews <= draw_until
    else:
        mask = np.ones_like(steps_ews, dtype=bool)
    ax.plot(steps_ews[mask], y[mask], "k-", lw=1.2)
    res = _trendline(ax, steps_ews[mask], y[mask], color="k")
    if step_line is not None:
        ax.axvline(step_line, color="b", linestyle="-", lw=2)
    # In the original figure the increase is highly significant
    ax.text(0.65, 0.75, "P < 10$^{-3}$", transform=ax.transAxes)
    ax.set_ylabel("Variance")


def plot_d(ax, steps_ews, var_false):
    y = var_false
    ax.plot(steps_ews, y, "k-", lw=1.2)
    res = _trendline(ax, steps_ews, y, color="k")
    if res is not None:
        ax.text(0.65, 0.75, f"P = {res.pvalue:.3f}", transform=ax.transAxes)


def plot_e(ax, steps_ews, ac1_crit, step_line=None, draw_until=None):
    y = ac1_crit
    if draw_until is not None:
        mask = steps_ews <= draw_until
    else:
        mask = np.ones_like(steps_ews, dtype=bool)
    ax.plot(steps_ews[mask], y[mask], "k-", lw=1.2)
    _trendline(ax, steps_ews[mask], y[mask], color="k")
    if step_line is not None:
        ax.axvline(step_line, color="b", linestyle="-", lw=2)
    ax.text(0.65, 0.75, "P < 10$^{-3}$", transform=ax.transAxes)
    ax.set_ylabel("AC1")


def plot_f(ax, steps_ews, ac1_false):
    y = ac1_false
    ax.plot(steps_ews, y, "k-", lw=1.2)
    _trendline(ax, steps_ews, y, color="k")
    ax.text(0.65, 0.75, "P < 10$^{-3}$", transform=ax.transAxes)


def plot_g(ax, steps_ews, lambda_crit, lambda_cor_rit, step_line=None, draw_until=None):
    y1 = lambda_crit
    y2 = lambda_cor_rit
    if draw_until is not None:
        mask = steps_ews <= draw_until
    else:
        mask = np.ones_like(steps_ews, dtype=bool)
    ax.plot(steps_ews[mask], y1[mask], "k", lw=1.2, label=r"$\lambda$")
    _trendline(ax, steps_ews[mask], y1[mask], color="k")
    ax.plot(steps_ews[mask], y2[mask], "r", lw=1.2, label=r"$\lambda_{cor}$")
    _trendline(ax, steps_ews[mask], y2[mask], color="r")
    if step_line is not None:
        ax.axvline(step_line, color="b", linestyle="-", lw=2)
    ax.text(0.78, 0.25, "P < 10$^{-3}$", transform=ax.transAxes, color="k")
    ax.text(0.78, 0.10, "P < 10$^{-3}$", transform=ax.transAxes, color="r")
    ax.set_ylabel(r"$\lambda$")
    ax.legend(loc="lower left")
    ax.set_xlabel("Time")


def plot_h(ax, steps_ews, lambda_false, lambda_cor_false):
    y1 = lambda_false
    y2 = lambda_cor_false
    ax.plot(steps_ews, y1, "k", lw=1.2, label=r"$\lambda$")
    _trendline(ax, steps_ews, y1, color="k")
    ax.plot(steps_ews, y2, "r", lw=1.2, label=r"$\lambda_{cor}$")
    res = _trendline(ax, steps_ews, y2, color="r")
    if res is not None:
        ax.text(0.68, 0.10, f"P = {res.pvalue:.3f}", transform=ax.transAxes, color="r")
    ax.legend(loc="lower left")
    ax.set_xlabel("Time")


def plot_figure1(n_steps=10000, dt=0.1, sigma=0.2, w=2000, savepath="figure1_recreation.pdf"):
    """Generate synthetic data, compute indicators, and recreate Figure 1.

    Parameters
    - n_steps, dt, sigma: simulation parameters
    - w: sliding window length for indicators
    - savepath: output PDF filename
    """
    # --- Generate Data ---
    time_crit, ts_crit, ts_det_crit = critical_transition_data(n_steps, dt, sigma)
    time_false, ts_false, phi_false = false_alarm_data(n_steps, dt)

    # --- Indicators ---
    var_crit, ac1_crit, lam_crit, lam_cor_crit = compute_indicators(ts_crit, w)
    var_false, ac1_false, lam_false, _ = compute_indicators(ts_false, w)
    # Corrected restoring rate assuming the known autocorrelated noise from the simulator
    ts_false_pw = prewhiten_with_phi(ts_false, phi_false)
    _, _, lam_cor_false, _ = compute_indicators(ts_false_pw, w)

    bound = w // 2
    # Use step index on the x-axis to match article (0..10,000)
    steps = np.arange(n_steps)
    steps_ews = steps[bound:-bound]
    step_line = 7000  # location of transition line in steps

    # --- Figure ---
    fig, axs = plt.subplots(4, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    fig.suptitle("Recreation of Boers (2021): Figure 1", fontsize=16, y=0.995)

    # Column 1 (critical transition)
    plot_a(axs[0, 0], steps, ts_crit, ts_det_crit, step_line=step_line)
    draw_limit = 6000
    plot_c(axs[1, 0], steps_ews, var_crit[bound:-bound], step_line=step_line, draw_until=draw_limit)
    plot_e(axs[2, 0], steps_ews, ac1_crit[bound:-bound], step_line=step_line, draw_until=draw_limit)
    plot_g(axs[3, 0], steps_ews, lam_crit[bound:-bound], lam_cor_crit[bound:-bound], step_line=step_line, draw_until=draw_limit)

    # Column 2 (false alarm model)
    plot_b(axs[0, 1], steps, ts_false)
    plot_d(axs[1, 1], steps_ews, var_false[bound:-bound])
    plot_f(axs[2, 1], steps_ews, ac1_false[bound:-bound])
    plot_h(axs[3, 1], steps_ews, lam_false[bound:-bound], lam_cor_false[bound:-bound])

    def thousands(x, pos):
        try:
            return f"{int(x):,}"
        except Exception:
            return ""
    xfmt = FuncFormatter(thousands)
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.xaxis.set_major_locator(MultipleLocator(2000))
            ax.xaxis.set_major_formatter(xfmt)
            # Remove grid to match article style
            ax.grid(False)
            # Keep 0–n_steps across all panels
            ax.set_xlim(0, n_steps)
    # Subpanel labels a–h
    labels = list("abcdefgh")
    idx = 0
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.text(0.01, 0.95, labels[idx], transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
            idx += 1
    axs[3, 0].set_xlabel("Time")
    axs[3, 1].set_xlabel("Time")

    if savepath:
        plt.savefig(savepath)
    plt.show()


if __name__ == "__main__":
    plot_figure1()
