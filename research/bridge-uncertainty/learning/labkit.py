"""Tools for the operational-modal-analysis learning lab (oma_lab.ipynb).

You may read all of this, but try the lab exercises BEFORE reading the
`check_*` functions; they would give the answers away.

Contents
  bridge_model()      mass and stiffness matrices of a small 3-DOF "bridge"
  simulate_ambient()  acceleration response to random (ambient) excitation
  ssi_cov()           covariance-driven stochastic subspace identification
  stabilization_plot()
  seasonal_dataset()  a year of hourly "identified" frequencies with temperature
  check_lab1 ... check_lab5
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, signal

_RNG_SEED = 2026
# hidden "true" thermal parameters used by seasonal_dataset (do not peek!)
_TRUE = dict(a=-6.0e-4, b=4.0e-3, Tc=0.5, s=0.7, noise=0.002)


# ---------------------------------------------------------------- the structure
def bridge_model(k_scale: float = 1.0):
    """3-DOF chain: three 1000 kg masses connected by springs (fixed at one end).
    Returns M, K (SI units). k_scale multiplies all stiffnesses (temperature!)."""
    m = 1000.0
    k = np.array([4.0e5, 3.0e5, 2.0e5]) * k_scale
    M = np.diag([m, m, m])
    K = np.array([[k[0] + k[1], -k[1], 0.0],
                  [-k[1], k[1] + k[2], -k[2]],
                  [0.0, -k[2], k[2]]])
    return M, K


def _rayleigh(M, K, zeta=0.02, f1=1.0, f2=6.0):
    w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2
    a0, a1 = np.linalg.solve([[1 / (2 * w1), w1 / 2], [1 / (2 * w2), w2 / 2]], [zeta, zeta])
    return a0 * M + a1 * K


def simulate_ambient(M, K, duration_s=900.0, fs=50.0, noise_frac=0.05, seed=_RNG_SEED):
    """Acceleration at each mass under white-noise forces, plus sensor noise.
    Returns (acc array shape [3, N], fs)."""
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    C = _rayleigh(M, K)
    Mi = np.linalg.inv(M)
    A = np.block([[np.zeros((n, n)), np.eye(n)], [-Mi @ K, -Mi @ C]])
    B = np.vstack([np.zeros((n, n)), Mi])
    Cout = np.hstack([-Mi @ K, -Mi @ C])
    D = Mi
    N = int(duration_s * fs)
    t = np.arange(N) / fs
    u = rng.normal(0, 1000.0, size=(N, n))
    sys = signal.StateSpace(A, B, Cout, D)
    _, y, _ = signal.lsim(sys, u, t)
    y = y.T
    y += noise_frac * y.std(axis=1, keepdims=True) * rng.normal(size=y.shape)
    return y, fs


# ---------------------------------------------------------------- SSI-cov
def ssi_cov(y, fs, i=30, orders=range(2, 41, 2)):
    """Covariance-driven stochastic subspace identification (SSI-cov).

    Idea in 4 steps:
      1. Output covariances R_k = E[y(t+k) y(t)^T] contain the system's
         free-decay behaviour (Natural Excitation Technique idea).
      2. Stack them in a block Toeplitz matrix; its rank = model order.
      3. An SVD of that matrix gives the "observability matrix" O.
      4. A shift of O gives the discrete state matrix A; the eigenvalues of A
         give frequencies and damping ratios.
    Returns a list of (order, freqs_Hz, damping_ratios) for each model order.
    Model orders are scanned because the true order is unknown; physical modes
    are those that stay STABLE as the order increases (stabilization diagram).
    """
    l, N = y.shape
    y = y - y.mean(axis=1, keepdims=True)
    R = [y[:, k:] @ y[:, :N - k].T / (N - k) for k in range(2 * i)]
    T = np.block([[R[i + r - c] for c in range(i)] for r in range(i)])
    U, S, _ = np.linalg.svd(T)
    out = []
    for n in orders:
        O = U[:, :n] * np.sqrt(S[:n])
        A = np.linalg.pinv(O[:-l]) @ O[l:]
        lam = np.linalg.eigvals(A)
        mu = np.log(lam.astype(complex)) * fs
        f = np.abs(mu) / (2 * np.pi)
        z = -np.real(mu) / np.abs(mu)
        keep = (np.imag(mu) > 0) & (z > 0) & (z < 0.2)
        order = np.argsort(f[keep])
        out.append((n, f[keep][order], z[keep][order]))
    return out


def stabilization_plot(result, ax=None, fmax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    for n, f, z in result:
        ax.plot(f, np.full_like(f, n), "k.", ms=4)
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Model order")
    if fmax:
        ax.set_xlim(0, fmax)
    ax.set_title("Stabilization diagram: physical modes form vertical columns")
    return ax


# ---------------------------------------------------------------- seasonal data
def _softplus(z, s):
    return s * np.logaddexp(0.0, z / s)


def seasonal_dataset(seed=_RNG_SEED):
    """One year of hourly 'identified' mode-1 frequencies of the 3-DOF model.
    Stiffness depends on temperature through a hidden law; frequencies carry
    identification noise. Returns dict(t_hours, T, f)."""
    rng = np.random.default_rng(seed)
    h = np.arange(365 * 24)
    T = (8 + 11 * np.sin(2 * np.pi * (h / 24 - 110) / 365) + 4 * np.sin(2 * np.pi * h / 24 - 2)
         + np.cumsum(rng.normal(0, 0.25, h.size)) * 0.15)
    T = T - (T.mean() - 9)
    p = _TRUE
    kfac = (1 + p["a"] * (T - 10)) * (1 + p["b"] * _softplus(p["Tc"] - T, p["s"])) ** 2
    M, K0 = bridge_model()
    f1_0 = np.sqrt(linalg.eigh(K0, M, eigvals_only=True)[0]) / (2 * np.pi)
    f = f1_0 * np.sqrt(kfac)  # frequency scales with sqrt(stiffness)
    f = f * (1 + p["noise"] * rng.normal(size=f.size))
    return dict(t_hours=h, T=T, f=f)


# ---------------------------------------------------------------- checkers
def _ok(cond, msg_ok, msg_bad):
    print(("PASS  " if cond else "NOT YET  ") + (msg_ok if cond else msg_bad))
    return bool(cond)


def _true_freqs():
    M, K = bridge_model()
    return np.sqrt(linalg.eigh(K, M, eigvals_only=True)) / (2 * np.pi)


def check_lab1(freqs_hz):
    f = np.sort(np.asarray(freqs_hz, float))
    good = f.shape == (3,) and np.allclose(f, _true_freqs(), rtol=1e-6)
    return _ok(good, "natural frequencies correct.",
               "expected 3 frequencies in Hz. Did you take sqrt of the eigenvalues and divide by 2*pi?")


def check_lab2(est_hz, tol=0.03):
    e = np.sort(np.asarray(est_hz, float))
    if e.shape != (3,):
        return _ok(False, "", f"expected 3 peak frequencies, got {e.size}. Tune find_peaks (prominence/height).")
    err = np.abs(e / _true_freqs() - 1)
    return _ok(np.all(err < tol), f"peak-picking within {100*err.max():.1f}% of the truth.",
               f"errors {np.round(100*err,1)}%. Try a longer nperseg (finer resolution).")


def _true_damping():
    M, K = bridge_model()
    w = 2 * np.pi * _true_freqs()
    w1, w2 = 2 * np.pi * 1.0, 2 * np.pi * 6.0
    a0, a1 = np.linalg.solve([[1 / (2 * w1), w1 / 2], [1 / (2 * w2), w2 / 2]], [0.02, 0.02])
    return a0 / (2 * w) + a1 * w / 2


def check_lab3(freqs_hz, damping, tol_f=0.01, tol_z=0.35):
    f = np.asarray(freqs_hz, float); z = np.asarray(damping, float)
    if f.shape != (3,) or z.shape != (3,):
        return _ok(False, "", "give 3 frequencies and 3 damping ratios (one per stable column).")
    o = np.argsort(f); f, z = f[o], z[o]
    ef = np.abs(f / _true_freqs() - 1)
    ez = np.abs(z / _true_damping() - 1)
    return _ok(np.all(ef < tol_f) and np.all(ez < tol_z),
               f"SSI frequencies within {100*ef.max():.2f}%, damping within {100*ez.max():.0f}% of the truth.",
               f"frequency errors {np.round(100*ef,2)}%, damping errors {np.round(100*ez,0)}%. "
               "Pick poles that stay stable across many orders, and average them.")


def check_lab4(a_hat, Tc_hat):
    good_a = abs(a_hat - _TRUE["a"]) < 0.3 * abs(_TRUE["a"])
    good_T = abs(Tc_hat - _TRUE["Tc"]) < 1.5
    return _ok(good_a and good_T, "recovered the hidden thermal law (a and Tc).",
               "not close yet. Check your model (square root!), starting guess and bounds.")


def check_lab5(cov_random, cov_seasonal_physics, cov_seasonal_cubic):
    good = (abs(cov_random - 0.9) < 0.03 and cov_seasonal_physics > 0.85
            and cov_seasonal_cubic < cov_seasonal_physics - 0.1)
    return _ok(good, "random split ~90 %; seasonal: physics holds, cubic breaks. You reproduced the core finding.",
               "random-split coverage should be ~0.90 (check the ceil((n+1)(1-alpha)) quantile), "
               "and the cubic should do clearly worse than physics on the seasonal split.")
