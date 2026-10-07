#!/usr/bin/env python3
"""
==========================================================================
 Case study: PINN vs. purely data-driven MLP vs. numerical solver
 for power system transient stability (SMIB and IEEE/WSCC 9-bus systems)

 Companion code for: "Physics-Informed Neural Networks for Power System
 Stability Analysis: A Systematic Review ..." (IEEE Access, submitted)

 HOW TO RUN (see the PDF guide for step-by-step instructions):
     python case_study.py --mode test     # 2-5 min: checks the code runs
     python case_study.py --mode quick    # 15-60 min: first real results
     python case_study.py --mode full     # paper results (use a GPU)

 Everything is written to the folder  results/
     results.csv              one row per training run (all experiments)
     summary_table11.csv      numbers for Table 11 of the paper
     table11_rows.tex         the same numbers as LaTeX rows (copy-paste)
     summary_E2_data.csv      error vs number of labelled points
     summary_E4_horizon.csv   error vs time horizon
     summary_E3_cost.csv      training/data/inference cost and break-even
     summary_E5_9bus.csv      stability classification and CCT (9-bus)
     run_info.txt             software, hardware and all settings
     fig_*.png                figures for the paper (300 dpi)
     models/*.pt              trained networks
==========================================================================
"""
import argparse, os, sys, time, json, platform, csv, math
import numpy as np
import torch
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------
# 0. Settings
# --------------------------------------------------------------------------
MODES = {
    # test: 2-5 minutes, only checks that the code runs (results are NOT meaningful)
    "test": dict(seeds=[0], adam_epochs=100, lbfgs_iters=20, n_colloc=500,
                 smib_nd=[10, 100], smib_horizons=[5.0, 20.0],
                 nb_nd=[200], nb_adam_epochs=100, nb_lbfgs_iters=20),
    # quick: one seed, short training -> checks that everything works
    "quick": dict(seeds=[0], adam_epochs=1500, lbfgs_iters=200, n_colloc=2000,
                  smib_nd=[10, 100, 1000], smib_horizons=[5.0, 20.0],
                  nb_nd=[200], nb_adam_epochs=1500, nb_lbfgs_iters=200),
    # full: settings reported in the paper (Table 10)
    "full": dict(seeds=[0, 1, 2, 3, 4], adam_epochs=5000, lbfgs_iters=500, n_colloc=5000,
                 smib_nd=[10, 50, 100, 500, 1000], smib_horizons=[2.0, 5.0, 10.0, 20.0],
                 nb_nd=[200], nb_adam_epochs=5000, nb_lbfgs_iters=500),
}
HIDDEN_LAYERS, NEURONS, LR = 5, 50, 1e-3   # MLP: 5 hidden layers x 50 neurons, tanh
W_DATA, W_PHYS = 1.0, 1.0                  # loss weights w_d and w_r
BASE_ND, BASE_T = 100, 20.0                # base SMIB setting (E1, E3)

# SMIB parameters (Misyris et al., 2020): m*dd(delta) + d*d(delta) + B*sin(delta) - P = 0
SMIB = dict(m=0.4, d=0.15, B=0.2, P_min=0.08, P_max=0.18, delta0=0.0, omega0=0.0)

# 9-bus settings: three-phase fault at bus 7, cleared by tripping line 5-7
NB = dict(T=2.0, tc_min=0.05, tc_max=0.25, fault_bus=7, trip_line=(5, 7))

REF_TOL = 1e-10    # tolerance of the reference (ground-truth) Radau solver
TIME_TOL = 1e-6    # practical tolerance used to time the solver (fair speed comparison)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
OUT = "results"


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# 1. Neural network (same architecture for MLP baseline and PINN)
# --------------------------------------------------------------------------
class Net(torch.nn.Module):
    """y(t, p) = y0 + (t/T)^2 * N(t, p).  The (t/T)^2 factor makes the
    initial angle AND initial speed exact (hard-constrained initial conditions)."""

    def __init__(self, n_out, T, p_min, p_max, y0, scale):
        super().__init__()
        layers, n_in = [], 2
        for _ in range(HIDDEN_LAYERS):
            layers += [torch.nn.Linear(n_in, NEURONS), torch.nn.Tanh()]
            n_in = NEURONS
        layers += [torch.nn.Linear(NEURONS, n_out)]
        self.f = torch.nn.Sequential(*layers)
        self.T, self.p_min, self.p_max = T, p_min, p_max
        self.register_buffer("y0", torch.as_tensor(y0, dtype=torch.float32).reshape(1, -1))
        self.scale = scale

    def forward(self, t, p):
        tau = t / self.T
        z = torch.cat([2 * tau - 1, 2 * (p - self.p_min) / (self.p_max - self.p_min) - 1], dim=1)
        return self.y0 + self.scale * tau ** 2 * self.f(z)


def time_derivatives(net, t, p):
    """Return delta, d(delta)/dt and d2(delta)/dt2 using automatic differentiation."""
    t = t.clone().requires_grad_(True)
    y = net(t, p)
    dy, ddy = [], []
    for k in range(y.shape[1]):
        g = torch.autograd.grad(y[:, k].sum(), t, create_graph=True)[0]
        gg = torch.autograd.grad(g.sum(), t, create_graph=True)[0]
        dy.append(g); ddy.append(gg)
    return y, torch.cat(dy, 1), torch.cat(ddy, 1)


def train(net, t_d, p_d, y_d, residual_fn, t_r, p_r, adam_epochs, lbfgs_iters):
    """Adam then L-BFGS. residual_fn=None -> purely data-driven MLP."""
    net.to(DEVICE)

    def loss_fn():
        loss = W_DATA * ((net(t_d, p_d) - y_d) ** 2).mean()
        if residual_fn is not None:
            y, dy, ddy = time_derivatives(net, t_r, p_r)
            loss = loss + W_PHYS * (residual_fn(y, dy, ddy, t_r, p_r) ** 2).mean()
        return loss

    t0 = time.time()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for ep in range(adam_epochs):
        opt.zero_grad(); loss = loss_fn(); loss.backward(); opt.step()
    lbfgs = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=lbfgs_iters, history_size=50,
                              tolerance_grad=1e-9, tolerance_change=1e-12,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad(); l = loss_fn(); l.backward(); return l
    lbfgs.step(closure)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0, float(loss_fn().detach())


def predict(net, t, p):
    """Prediction of angle and speed (speed by autodiff) + inference time."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        y = net(t, p)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time() - t0
    _, dy, _ = time_derivatives(net, t, p)
    return y.detach().cpu().numpy(), dy.detach().cpu().numpy(), t_inf


def tens(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float32).reshape(len(a), -1), device=DEVICE)


# --------------------------------------------------------------------------
# 2. SMIB system
# --------------------------------------------------------------------------
def smib_solve(P, T, t_eval, tol=REF_TOL):
    m, d, B = SMIB["m"], SMIB["d"], SMIB["B"]
    sol = solve_ivp(lambda t, x: [x[1], (P - d * x[1] - B * np.sin(x[0])) / m], (0, T),
                    [SMIB["delta0"], SMIB["omega0"]], method="Radau", t_eval=t_eval, rtol=tol, atol=tol)
    return sol.y  # rows: delta, omega


def smib_residual(y, dy, ddy, t, P):
    m, d, B = SMIB["m"], SMIB["d"], SMIB["B"]
    return m * ddy + d * dy + B * torch.sin(y) - P


def smib_test_set(T):
    P_test = np.linspace(SMIB["P_min"], SMIB["P_max"], 25)
    ts = np.linspace(0, T, 1001)
    t0 = time.time()
    for P in P_test:                       # timing of the practical solver
        smib_solve(P, T, ts, tol=TIME_TOL)
    solver_ms = (time.time() - t0) / len(P_test) * 1e3
    Y = np.stack([smib_solve(P, T, ts) for P in P_test])     # ground truth
    return P_test, ts, Y, solver_ms


def make_smib_net(T):
    return Net(1, T, SMIB["P_min"], SMIB["P_max"], [SMIB["delta0"]], scale=1.0)


def smib_predict_grid(net, test):
    P_test, ts, Y, _ = test
    tt = tens(np.tile(ts, len(P_test))); pp = tens(np.repeat(P_test, len(ts)))
    d_hat, w_hat, t_inf = predict(net, tt, pp)
    return d_hat.reshape(len(P_test), -1), w_hat.reshape(len(P_test), -1), t_inf


def smib_run(method, nd, T, seed, cfg, test):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    P_test, ts, Y, solver_ms = test
    # labelled data: nd points taken from min(nd, 20) simulated trajectories
    n_traj = min(nd, 20)
    P_tr = rng.uniform(SMIB["P_min"], SMIB["P_max"], n_traj)
    t_grid = np.linspace(0, T, 2001)
    t0 = time.time()
    trajs = [smib_solve(P, T, t_grid, tol=TIME_TOL)[0] for P in P_tr]
    datagen_s = time.time() - t0
    k = rng.integers(0, n_traj, nd); j = rng.integers(0, len(t_grid), nd)
    t_d, p_d = tens(t_grid[j]), tens(P_tr[k])
    y_d = tens(np.array([trajs[a][b] for a, b in zip(k, j)]))
    # collocation points (PINN only), Latin-hypercube-like stratified sampling
    n = cfg["n_colloc"]
    t_r = tens((rng.permutation(n) + rng.random(n)) / n * T)
    p_r = tens(SMIB["P_min"] + (rng.permutation(n) + rng.random(n)) / n * (SMIB["P_max"] - SMIB["P_min"]))
    net = make_smib_net(T)
    res = smib_residual if method == "PINN" else None
    train_s, final_loss = train(net, t_d, p_d, y_d, res, t_r, p_r, cfg["adam_epochs"], cfg["lbfgs_iters"])
    # evaluation on 25 unseen operating points x 1001 time steps
    d_hat, w_hat, t_inf = smib_predict_grid(net, test)
    d_ref, w_ref = Y[:, 0], Y[:, 1]
    out = dict(rel_l2_delta=np.linalg.norm(d_hat - d_ref) / np.linalg.norm(d_ref),
               rel_l2_omega=np.linalg.norm(w_hat - w_ref) / np.linalg.norm(w_ref),
               max_abs_err_delta_rad=np.abs(d_hat - d_ref).max(),
               train_time_s=train_s, datagen_time_s=datagen_s,
               infer_ms_per_traj=t_inf / len(P_test) * 1e3, solver_ms_per_traj=solver_ms,
               final_loss=final_loss)
    return out, net, (d_hat, w_hat)


# --------------------------------------------------------------------------
# 3. IEEE / WSCC 9-bus system, classical generator model
#    (data: Anderson & Fouad; Sauer & Pai), 100 MVA base, 60 Hz
# --------------------------------------------------------------------------
LINES = [(1, 4, 0, 0.0576, 0), (4, 5, 0.010, 0.085, 0.176), (4, 6, 0.017, 0.092, 0.158),
         (5, 7, 0.032, 0.161, 0.306), (6, 9, 0.039, 0.170, 0.358), (7, 8, 0.0085, 0.072, 0.149),
         (8, 9, 0.0119, 0.1008, 0.209), (2, 7, 0, 0.0625, 0), (3, 9, 0, 0.0586, 0)]
LOADS = {5: 1.25 + 0.5j, 6: 0.9 + 0.3j, 8: 1.0 + 0.35j}
XD = np.array([0.0608, 0.1198, 0.1813]); H = np.array([23.64, 6.4, 3.01])
WS = 2 * np.pi * 60; MG = 2 * H / WS; DG = np.zeros(3)


def ybus(exclude=()):
    Y = np.zeros((9, 9), complex)
    for f, t, r, x, b in LINES:
        if (f, t) in exclude:
            continue
        y = 1 / complex(r, x); f -= 1; t -= 1
        Y[f, f] += y + 1j * b / 2; Y[t, t] += y + 1j * b / 2; Y[f, t] -= y; Y[t, f] -= y
    return Y


def power_flow():
    Y = ybus(); V = np.array([1.04, 1.025, 1.025] + [1.0] * 6); th = np.zeros(9)
    Psp = np.zeros(9); Qsp = np.zeros(9); Psp[1], Psp[2] = 1.63, 0.85
    for b, s in LOADS.items():
        Psp[b - 1] -= s.real; Qsp[b - 1] -= s.imag
    pq = [3, 4, 5, 6, 7, 8]

    def mismatch(x):
        t2 = np.r_[0, x[:8]]; V2 = V.copy(); V2[pq] = x[8:]
        Vc = V2 * np.exp(1j * t2); S = Vc * np.conj(Y @ Vc)
        return np.r_[(Psp - S.real)[1:], (Qsp - S.imag)[pq]]
    x = np.r_[th[1:], V[pq]]
    for _ in range(30):
        f0 = mismatch(x)
        if np.abs(f0).max() < 1e-11:
            break
        J = np.zeros((14, 14))
        for k in range(14):
            e = np.zeros(14); e[k] = 1e-7; J[:, k] = (mismatch(x + e) - f0) / 1e-7
        x = x - np.linalg.solve(J, f0)
    th[1:] = x[:8]; V[pq] = x[8:]
    Vc = V * np.exp(1j * th); S = Vc * np.conj(Y @ Vc)
    return V, th, Vc, S


def reduced_matrices():
    V, th, Vc, S = power_flow()
    I = np.conj(S[:3] / Vc[:3]); E = Vc[:3] + 1j * XD * I
    Em, d0, Pm = np.abs(E), np.angle(E), S[:3].real

    def augmented(Yn):
        Yl = Yn.copy()
        for b, s in LOADS.items():                       # constant-impedance loads
            Yl[b - 1, b - 1] += np.conj(s) / V[b - 1] ** 2
        Ya = np.zeros((12, 12), complex); Ya[3:, 3:] = Yl
        for i in range(3):                                # generator transient reactances
            y = 1 / (1j * XD[i]); Ya[i, i] += y; Ya[i, 3 + i] -= y; Ya[3 + i, i] -= y; Ya[3 + i, 3 + i] += y
        return Ya

    def kron(Ya):
        keep = [0, 1, 2]; rest = [k for k in range(Ya.shape[0]) if k not in keep]
        return Ya[np.ix_(keep, keep)] - Ya[np.ix_(keep, rest)] @ np.linalg.solve(Ya[np.ix_(rest, rest)], Ya[np.ix_(rest, keep)])
    Ya = augmented(ybus())
    fb = 3 + NB["fault_bus"] - 1
    keep = [k for k in range(12) if k != fb]              # bolted fault: faulted bus voltage = 0
    Yfault = kron(Ya[np.ix_(keep, keep)])
    Ypost = kron(augmented(ybus(exclude=[NB["trip_line"]])))
    return dict(Em=Em, d0=d0, Pm=Pm, Ypre=kron(Ya), Yfault=Yfault, Ypost=Ypost, V=V, th=th)


NBD = None


def pe_np(d, Yr):
    Em = NBD["Em"]; G, Bm = Yr.real, Yr.imag; dd = d[:, None] - d[None, :]
    return (Em[:, None] * Em[None, :] * (G * np.cos(dd) + Bm * np.sin(dd))).sum(1)


def nb_solve(tc, ts, tol=REF_TOL):
    T = NB["T"]
    f = lambda Yr: (lambda t, x: np.r_[x[3:], (NBD["Pm"] - pe_np(x[:3], Yr) - DG * x[3:]) / MG])
    s1 = solve_ivp(f(NBD["Yfault"]), (0, tc), np.r_[NBD["d0"], 0, 0, 0], method="Radau", rtol=tol, atol=tol, dense_output=True)
    s2 = solve_ivp(f(NBD["Ypost"]), (tc, T), s1.y[:, -1], method="Radau", rtol=tol, atol=tol, dense_output=True)
    x = np.where(ts < tc, s1.sol(np.minimum(ts, tc)), s2.sol(np.maximum(ts, tc)))
    return x  # rows: delta1..3 (rad), omega1..3 (rad/s deviation)


def is_unstable(delta):
    """Unstable if any two rotor angles separate by more than pi rad (180 deg)."""
    sep = np.abs(delta[:, None, :] - delta[None, :, :]).max()
    return bool(sep > np.pi)


def nb_cct():
    lo, hi = NB["tc_min"], NB["tc_max"] + 0.2
    ts = np.linspace(0, NB["T"], 2001)
    for _ in range(25):
        mid = (lo + hi) / 2
        if is_unstable(nb_solve(mid, ts)[:3]):
            hi = mid
        else:
            lo = mid
    return lo


def nb_residual_factory():
    Em = torch.tensor(NBD["Em"], dtype=torch.float32, device=DEVICE)
    Pm = torch.tensor(NBD["Pm"], dtype=torch.float32, device=DEVICE)
    M = torch.tensor(MG, dtype=torch.float32, device=DEVICE)
    D = torch.tensor(DG, dtype=torch.float32, device=DEVICE)
    mats = {k: (torch.tensor(NBD[k].real, dtype=torch.float32, device=DEVICE),
                torch.tensor(NBD[k].imag, dtype=torch.float32, device=DEVICE)) for k in ("Yfault", "Ypost")}

    def pe(d, key):
        G, Bm = mats[key]; dd = d[:, :, None] - d[:, None, :]
        return (Em[None, :, None] * Em[None, None, :] * (G * torch.cos(dd) + Bm * torch.sin(dd))).sum(2)

    def residual(y, dy, ddy, t, tc):
        Pe = torch.where(t < tc, pe(y, "Yfault"), pe(y, "Ypost"))
        return ddy + (D * dy - Pm + Pe) / M      # acceleration residual (rad/s^2)
    return residual


def nb_test_set():
    tcs = np.linspace(NB["tc_min"], NB["tc_max"], 21)
    ts = np.linspace(0, NB["T"], 1001)
    t0 = time.time()
    for tc in tcs:
        nb_solve(tc, ts, tol=TIME_TOL)
    solver_ms = (time.time() - t0) / len(tcs) * 1e3
    X = np.stack([nb_solve(tc, ts) for tc in tcs])
    return tcs, ts, X, solver_ms


def make_nb_net():
    return Net(3, NB["T"], NB["tc_min"], NB["tc_max"], NBD["d0"], scale=5.0)


def nb_predict_grid(net, test):
    tcs, ts, X, _ = test
    tt = tens(np.tile(ts, len(tcs))); pp = tens(np.repeat(tcs, len(ts)))
    d_hat, _, t_inf = predict(net, tt, pp)
    return d_hat.reshape(len(tcs), len(ts), 3).transpose(0, 2, 1), t_inf


def nb_run(method, nd, seed, cfg, test):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    tcs, ts, X, solver_ms = test
    T = NB["T"]; n_traj = 20
    tc_tr = rng.uniform(NB["tc_min"], NB["tc_max"], n_traj)
    t_grid = np.linspace(0, T, 2001)
    t0 = time.time()
    trajs = [nb_solve(tc, t_grid, tol=TIME_TOL)[:3] for tc in tc_tr]
    datagen_s = time.time() - t0
    k = rng.integers(0, n_traj, nd); j = rng.integers(0, len(t_grid), nd)
    t_d, p_d = tens(t_grid[j]), tens(tc_tr[k])
    y_d = tens(np.array([trajs[a][:, b] for a, b in zip(k, j)]))
    n = cfg["n_colloc"]
    t_r = tens((rng.permutation(n) + rng.random(n)) / n * T)
    p_r = tens(NB["tc_min"] + (rng.permutation(n) + rng.random(n)) / n * (NB["tc_max"] - NB["tc_min"]))
    net = make_nb_net()
    res = nb_residual_factory() if method == "PINN" else None
    # physics residual is an acceleration (large numbers) -> scale it to the data loss
    global W_PHYS
    w_old = W_PHYS
    if res is not None:
        W_PHYS = w_old * 1e-4
    train_s, final_loss = train(net, t_d, p_d, y_d, res, t_r, p_r, cfg["nb_adam_epochs"], cfg["nb_lbfgs_iters"])
    W_PHYS = w_old
    d_hat, t_inf = nb_predict_grid(net, test)
    d_ref = X[:, :3]
    stable_ref = np.array([not is_unstable(x) for x in d_ref])
    stable_hat = np.array([not is_unstable(x) for x in d_hat])
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    # relative angles (with respect to machine 1) are what matters for stability
    ra = lambda d: d[:, 1:] - d[:, :1]
    st = stable_ref
    # predicted CCT = largest clearing time still predicted stable before the first predicted-unstable case
    pred_cct = tcs[0]
    for tc, s in zip(tcs, stable_hat):
        if s:
            pred_cct = tc
        else:
            break
    out = dict(rel_l2_delta=rel(ra(d_hat[st]), ra(d_ref[st])) if st.any() else float("nan"),
               rel_l2_delta_all=rel(ra(d_hat), ra(d_ref)),
               max_abs_err_delta_rad=np.abs(ra(d_hat[st]) - ra(d_ref[st])).max() if st.any() else float("nan"),
               classification_accuracy=float((stable_hat == stable_ref).mean()),
               false_stable=int(((stable_hat) & (~stable_ref)).sum()),
               false_unstable=int(((~stable_hat) & (stable_ref)).sum()),
               predicted_cct_grid_s=pred_cct,
               train_time_s=train_s, datagen_time_s=datagen_s,
               infer_ms_per_traj=t_inf / len(tcs) * 1e3, solver_ms_per_traj=solver_ms, final_loss=final_loss)
    return out, net, d_hat


# --------------------------------------------------------------------------
# 4. Experiments, summaries and figures
# --------------------------------------------------------------------------
FIELDS = ["system", "method", "experiment", "n_data", "t_end", "seed", "rel_l2_delta", "rel_l2_omega",
          "rel_l2_delta_all", "max_abs_err_delta_rad", "classification_accuracy", "false_stable",
          "false_unstable", "predicted_cct_grid_s", "train_time_s", "datagen_time_s",
          "infer_ms_per_traj", "solver_ms_per_traj", "final_loss", "reference_cct_s"]


def load_rows():
    """Resume support: read runs that are already finished (e.g. after a Colab disconnect)."""
    path, rows = os.path.join(OUT, "results.csv"), []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if v in ("", None):
                    continue
                if k in ("system", "method", "experiment"):
                    d[k] = v
                elif k in ("n_data", "seed", "false_stable", "false_unstable"):
                    d[k] = int(float(v))
                else:
                    d[k] = float(v)
            rows.append(d)
    return rows


def find_row(rows, **kw):
    for r in rows:
        if all(r.get(a) == b for a, b in kw.items()):
            return r
    return None


def save_rows(rows):
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def ms(vals, fmt="{:.2e}"):
    v = np.array([x for x in vals if x == x], dtype=float)
    if len(v) == 0:
        return "n/a", float("nan"), float("nan")
    return (fmt + " ± " + fmt).format(v.mean(), v.std()), v.mean(), v.std()


def run_info(cfg, mode, extra):
    lines = [f"Mode: {mode}", f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"Python: {platform.python_version()}", f"PyTorch: {torch.__version__}",
             f"NumPy: {np.__version__}", f"SciPy: {__import__('scipy').__version__}",
             f"OS: {platform.platform()}", f"CPU: {platform.processor() or platform.machine()}, {os.cpu_count()} logical cores",
             f"Device used for training: {DEVICE}"]
    if DEVICE.type == "cuda":
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1]); lines.append(f"RAM: {kb / 1024 ** 2:.1f} GB")
    except Exception:
        pass
    lines += ["", "Settings:", json.dumps(cfg, indent=2),
              f"Network: {HIDDEN_LAYERS} hidden layers x {NEURONS} neurons, tanh, lr={LR}",
              f"Loss weights: w_d={W_DATA}, w_r={W_PHYS} (9-bus residual additionally scaled by 1e-4)",
              f"SMIB: {SMIB}", f"9-bus: {NB}", f"Reference solver: Radau IIA, rtol=atol={REF_TOL}",
              f"Solver timing: Radau IIA, rtol=atol={TIME_TOL}", ""] + extra
    with open(os.path.join(OUT, "run_info.txt"), "w") as f:
        f.write("\n".join(lines))


def figures(rows, smib_examples, nb_example, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "figure.dpi": 300})
    seeds_note = f"mean ± std over {len(cfg['seeds'])} seed(s)"
    # (a) SMIB trajectories
    if smib_examples:
        fig, axs = plt.subplots(1, len(smib_examples), figsize=(7.0, 2.4), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, (nd, (P_test, ts, Y, pinn, mlp)) in zip(axs, smib_examples.items()):
            for i, c in zip([0, 12, 24], ["C0", "C1", "C2"]):
                ax.plot(ts, Y[i, 0], "-", color=c, lw=1.2, label=f"Radau, P={P_test[i]:.3f}")
                ax.plot(ts, mlp[i], "--", color=c, lw=0.9)
                ax.plot(ts, pinn[i], ":", color=c, lw=1.4)
            ax.set_title(f"$N_d$ = {nd} labelled points"); ax.set_xlabel("Time (s)")
        axs[0].set_ylabel("Rotor angle $\\delta$ (rad)")
        axs[0].plot([], [], "k-", label="Radau (reference)"); axs[0].plot([], [], "k--", label="MLP")
        axs[0].plot([], [], "k:", label="PINN")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=6, fontsize=6, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, "fig_smib_trajectories.png")); plt.close(fig)
    # (b) error vs number of labelled points
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for meth, mk in (("MLP", "s--"), ("PINN", "o-")):
        nds = sorted({r["n_data"] for r in rows if r["experiment"] in ("E1", "E2") and r["system"] == "SMIB"})
        mu = [np.mean([r["rel_l2_delta"] for r in rows if r["system"] == "SMIB" and r["method"] == meth and r["n_data"] == n and r["t_end"] == BASE_T]) for n in nds]
        sd = [np.std([r["rel_l2_delta"] for r in rows if r["system"] == "SMIB" and r["method"] == meth and r["n_data"] == n and r["t_end"] == BASE_T]) for n in nds]
        ax.errorbar(nds, mu, yerr=sd, fmt=mk, ms=3, capsize=2, label=meth)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Number of labelled points $N_d$")
    ax.set_ylabel("Relative $L_2$ error of $\\delta$"); ax.legend(title=seeds_note, fontsize=6, title_fontsize=6)
    ax.grid(alpha=0.3, which="both"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_error_vs_data.png")); plt.close(fig)
    # (c) error vs horizon
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for meth, mk in (("MLP", "s--"), ("PINN", "o-")):
        Ts = sorted({r["t_end"] for r in rows if r["system"] == "SMIB" and r["n_data"] == BASE_ND})
        mu = [np.mean([r["rel_l2_delta"] for r in rows if r["system"] == "SMIB" and r["method"] == meth and r["n_data"] == BASE_ND and r["t_end"] == T]) for T in Ts]
        sd = [np.std([r["rel_l2_delta"] for r in rows if r["system"] == "SMIB" and r["method"] == meth and r["n_data"] == BASE_ND and r["t_end"] == T]) for T in Ts]
        ax.errorbar(Ts, mu, yerr=sd, fmt=mk, ms=3, capsize=2, label=meth)
    ax.set_yscale("log"); ax.set_xlabel("Time horizon $T$ (s)"); ax.set_ylabel("Relative $L_2$ error of $\\delta$")
    ax.legend(title=f"$N_d$ = {BASE_ND}; {seeds_note}", fontsize=6, title_fontsize=6); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_error_vs_horizon.png")); plt.close(fig)
    # (d) 9-bus trajectories
    if nb_example is not None:
        tcs, ts, X, preds = nb_example
        idx = [int(np.argmin(np.abs(tcs - 0.10))), int(np.argmin(np.abs(tcs - 0.20)))]
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
        for ax, i in zip(axs, idx):
            for g, c in ((1, "C0"), (2, "C1")):
                ax.plot(ts, np.degrees(X[i, g] - X[i, 0]), "-", color=c, lw=1.2, label=f"Radau $\\delta_{g+1}-\\delta_1$")
                for meth, ls in (("MLP", "--"), ("PINN", ":")):
                    ax.plot(ts, np.degrees(preds[meth][i, g] - preds[meth][i, 0]), ls, color=c, lw=1.0, label=f"{meth}")
            ax.axvline(tcs[i], color="gray", lw=0.6)
            state = "unstable" if is_unstable(X[i, :3]) else "stable"
            ax.set_title(f"$t_c$ = {tcs[i]:.2f} s (reference: {state})")
            ax.set_xlabel("Time (s)"); ax.set_ylim(-200, 400)
        axs[0].set_ylabel("Relative rotor angle (deg)"); axs[0].legend(fontsize=5, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_9bus_trajectories.png")); plt.close(fig)


def main():
    global OUT, NBD
    ap = argparse.ArgumentParser(description="PINN stability case study")
    ap.add_argument("--mode", choices=list(MODES), default="quick")
    ap.add_argument("--skip9bus", action="store_true", help="run only the SMIB experiments")
    ap.add_argument("--out", default="results", help="output folder (e.g. a Google Drive folder)")
    ap.add_argument("--restart", action="store_true", help="ignore finished runs and start from zero")
    a = ap.parse_args()
    cfg = MODES[a.mode]; OUT = a.out
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    log(f"Device: {DEVICE}  |  mode: {a.mode}  |  output folder: {os.path.abspath(OUT)}")
    if DEVICE.type == "cpu":
        log("NOTE: no GPU found. This works, but full mode will be slow (several hours).")

    # ---- list of SMIB runs ----
    jobs = []
    for s in cfg["seeds"]:
        for nd in cfg["smib_nd"]:
            jobs.append(("E1" if nd == BASE_ND else "E2", nd, BASE_T, s))
        for T in cfg["smib_horizons"]:
            if T != BASE_T:
                jobs.append(("E4", BASE_ND, T, s))
    n_total = 2 * len(jobs) + (0 if a.skip9bus else 2 * len(cfg["seeds"]) * len(cfg["nb_nd"]))
    log(f"Total training runs: {n_total}")
    rows, k, tests, smib_examples = ([] if a.restart else load_rows()), 0, {}, {}
    if rows:
        log(f"Resuming: {len(rows)} finished runs found in {OUT}/results.csv (use --restart to start again)")
    t_start = time.time()
    for exp, nd, T, s in jobs:
        if T not in tests:
            tests[T] = smib_test_set(T)
        preds = {}
        for meth in ("MLP", "PINN"):
            k += 1
            mfile = os.path.join(OUT, "models", f"smib_{meth}_nd{nd}_T{int(T)}_s{s}.pt")
            if find_row(rows, system="SMIB", method=meth, n_data=nd, t_end=T, seed=s) and os.path.exists(mfile):
                log(f"[{k}/{n_total}] SMIB {meth:4s} N_d={nd} T={T}s seed={s}: already done, skipped")
                net = make_smib_net(T).to(DEVICE); net.load_state_dict(torch.load(mfile, map_location=DEVICE))
                preds[meth] = smib_predict_grid(net, tests[T])[0]
                continue
            log(f"[{k}/{n_total}] SMIB {meth:4s} exp={exp} N_d={nd:4d} T={T:4.1f}s seed={s} ...")
            out, net, (dh, wh) = smib_run(meth, nd, T, s, cfg, tests[T])
            preds[meth] = dh
            rows.append(dict(system="SMIB", method=meth, experiment=exp, n_data=nd, t_end=T, seed=s, **out))
            torch.save(net.state_dict(), mfile)
            log(f"      rel. L2 error (angle) = {out['rel_l2_delta']:.3e}   train time = {out['train_time_s']:.1f} s"
                f"   elapsed = {(time.time() - t_start) / 60:.1f} min")
            save_rows(rows)
        if s == cfg["seeds"][0] and T == BASE_T and nd in (min(cfg["smib_nd"]), BASE_ND):
            P_test, ts, Y, _ = tests[T]
            smib_examples[nd] = (P_test, ts, Y, preds["PINN"], preds["MLP"])

    # ---- 9-bus runs (E5, E6) ----
    nb_example, extra = None, []
    if not a.skip9bus:
        NBD = reduced_matrices()
        cct = nb_cct()
        extra.append(f"9-bus initial rotor angles (deg): {np.degrees(NBD['d0']).round(2).tolist()}")
        extra.append(f"9-bus internal EMFs (p.u.): {NBD['Em'].round(4).tolist()}, Pm (p.u.): {NBD['Pm'].round(4).tolist()}")
        extra.append(f"9-bus reference critical clearing time (bisection): {cct:.4f} s")
        log(f"9-bus reference CCT = {cct:.4f} s")
        test = nb_test_set(); preds = {}
        for s in cfg["seeds"]:
            for nd in cfg["nb_nd"]:
                for meth in ("MLP", "PINN"):
                    k += 1
                    mfile = os.path.join(OUT, "models", f"9bus_{meth}_nd{nd}_s{s}.pt")
                    if find_row(rows, system="9-bus", method=meth, n_data=nd, seed=s) and os.path.exists(mfile):
                        log(f"[{k}/{n_total}] 9-bus {meth:4s} N_d={nd} seed={s}: already done, skipped")
                        if s == cfg["seeds"][0]:
                            net = make_nb_net().to(DEVICE); net.load_state_dict(torch.load(mfile, map_location=DEVICE))
                            preds[meth] = nb_predict_grid(net, test)[0]
                        continue
                    log(f"[{k}/{n_total}] 9-bus {meth:4s} N_d={nd} seed={s} ...")
                    out, net, dh = nb_run(meth, nd, s, cfg, test)
                    out["reference_cct_s"] = cct
                    rows.append(dict(system="9-bus", method=meth, experiment="E5/E6", n_data=nd, t_end=NB["T"], seed=s, **out))
                    torch.save(net.state_dict(), mfile)
                    if s == cfg["seeds"][0]:
                        preds[meth] = dh
                    log(f"      rel. L2 error (stable cases) = {out['rel_l2_delta']:.3e}   classification accuracy = "
                        f"{out['classification_accuracy']:.2f}   train time = {out['train_time_s']:.1f} s")
                    save_rows(rows)
        nb_example = (test[0], test[1], test[2], preds)

    # ---- summaries ----
    def sel(**kw):
        return [r for r in rows if all(r.get(a_) == b_ for a_, b_ in kw.items())]
    t11, tex = [], []
    for system, T in (("SMIB", BASE_T), ("9-bus", NB["T"])):
        base = sel(system=system, n_data=BASE_ND if system == "SMIB" else cfg["nb_nd"][0], t_end=T)
        if not base:
            continue
        solver = np.mean([r["solver_ms_per_traj"] for r in base])
        t11.append(dict(system=system, method="Radau", rel_l2_percent="--", max_err_rad="--", train_s="--", infer_ms=f"{solver:.1f}"))
        tex.append(f"{system} & Radau & -- & -- & -- & {solver:.1f} \\\\")
        for meth in ("MLP", "PINN"):
            rr = [r for r in base if r["method"] == meth]
            e = ms([100 * r["rel_l2_delta"] for r in rr], "{:.2f}")      # in percent
            mx = ms([r["max_abs_err_delta_rad"] for r in rr], "{:.3f}")
            tr = ms([r["train_time_s"] for r in rr], "{:.1f}"); inf = ms([r["infer_ms_per_traj"] for r in rr], "{:.2f}")
            t11.append(dict(system=system, method=meth, rel_l2_percent=e[0], max_err_rad=mx[0], train_s=tr[0], infer_ms=inf[0]))
            pm = lambda x: x.replace("±", "$\\pm$")
            tex.append("      & " + meth + " & " + pm(e[0]) + " & " + pm(mx[0]) + " & " + pm(tr[0]) + " & " + pm(inf[0]) + " \\\\")
        tex.append("\\midrule")
    with open(os.path.join(OUT, "summary_table11.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "method", "rel_l2_percent", "max_err_rad", "train_s", "infer_ms"])
        w.writeheader(); w.writerows(t11)
    with open(os.path.join(OUT, "table11_rows.tex"), "w") as f:
        f.write("% Paste between \\midrule and \\bottomrule of Table 11 (tab:results)\n" + "\n".join(tex[:-1]) + "\n")
    # E2 and E4
    with open(os.path.join(OUT, "summary_E2_data.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "n_data", "rel_l2_delta_mean", "rel_l2_delta_std", "rel_l2_omega_mean", "n_seeds"])
        for meth in ("MLP", "PINN"):
            for nd in cfg["smib_nd"]:
                rr = sel(system="SMIB", method=meth, n_data=nd, t_end=BASE_T)
                w.writerow([meth, nd, np.mean([r["rel_l2_delta"] for r in rr]), np.std([r["rel_l2_delta"] for r in rr]),
                            np.mean([r["rel_l2_omega"] for r in rr]), len(rr)])
    with open(os.path.join(OUT, "summary_E4_horizon.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "t_end_s", "rel_l2_delta_mean", "rel_l2_delta_std", "n_seeds"])
        for meth in ("MLP", "PINN"):
            for T in cfg["smib_horizons"]:
                rr = sel(system="SMIB", method=meth, n_data=BASE_ND, t_end=T)
                w.writerow([meth, T, np.mean([r["rel_l2_delta"] for r in rr]), np.std([r["rel_l2_delta"] for r in rr]), len(rr)])
    # E3 cost / break-even:  N* = (data generation + training) / (solver time - NN time) per query
    with open(os.path.join(OUT, "summary_E3_cost.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["system", "method", "datagen_s", "train_s", "solver_ms_per_traj", "nn_ms_per_traj", "speedup_per_query", "break_even_queries"])
        for system, T in (("SMIB", BASE_T), ("9-bus", NB["T"])):
            for meth in ("MLP", "PINN"):
                rr = sel(system=system, method=meth, t_end=T, n_data=BASE_ND if system == "SMIB" else cfg["nb_nd"][0])
                if not rr:
                    continue
                dg = np.mean([r["datagen_time_s"] for r in rr]); tr = np.mean([r["train_time_s"] for r in rr])
                so = np.mean([r["solver_ms_per_traj"] for r in rr]); nn = np.mean([r["infer_ms_per_traj"] for r in rr])
                be = (dg + tr) / max((so - nn) / 1e3, 1e-12)
                w.writerow([system, meth, f"{dg:.2f}", f"{tr:.1f}", f"{so:.2f}", f"{nn:.3f}", f"{so / nn:.0f}", f"{be:.0f}"])
    if not a.skip9bus:
        with open(os.path.join(OUT, "summary_E5_9bus.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["method", "rel_l2_stable_mean", "classification_accuracy_mean", "false_stable_total", "false_unstable_total", "predicted_cct_mean_s", "reference_cct_s"])
            for meth in ("MLP", "PINN"):
                rr = sel(system="9-bus", method=meth)
                w.writerow([meth, np.nanmean([r["rel_l2_delta"] for r in rr]), np.mean([r["classification_accuracy"] for r in rr]),
                            sum(r["false_stable"] for r in rr), sum(r["false_unstable"] for r in rr),
                            np.mean([r["predicted_cct_grid_s"] for r in rr]), rr[0]["reference_cct_s"]])
    run_info(cfg, a.mode, extra)
    figures(rows, smib_examples, nb_example, cfg)
    log(f"\nDONE in {(time.time() - t_start) / 60:.1f} min. All outputs are in: {os.path.abspath(OUT)}")
    log("Open summary_table11.csv first; see the PDF guide, Part F, for how to read every file.")


if __name__ == "__main__":
    main()
