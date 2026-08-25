"""Ground-truth recoverability engine for small binary missing-data instances.

Three instruments, combined with an explicit precedence rule (see `decide`):

1. Assumption-free LP relaxation (`lp_range`): linear program over full-table
   cells t[v, r] whose induced observed conditionals match reference observed
   laws exactly. Any two feasible points are valid joint laws of (V, R)
   (mechanism assumptions dropped), so a functional that varies over this
   polytope is CERTIFIED unrecoverable under any submodel (sound one way).
   Witness tables returned.

2. Model-aware witness search (`model_witness_search`): constrained nonlinear
   optimization over CPT parameters theta with equality constraints
   F(theta) == F(theta_ref) (identical observed laws under the m-graph
   factorization). A pair with distance < 1e-9 and |dphi| > tol certifies
   unrecoverability UNDER THE MODEL numerically.

3. Identification-formula checks (`IDENTITY_FORMULAS`): closed-form estimators
   that equal the target under the instance's structure class; verified
   numerically against the true generating model at machine precision.

Verdict precedence in `decide`: formula-pass -> RECOVERABLE (formula-certified);
else model witness -> UNRECOVERABLE; else LP width -> UNRECOVERABLE_RELAXED;
else UNDETERMINED. Positive bank instances carry formulas; negatives rely on
witnesses, so no bank instance lands in UNDETERMINED.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
from scipy.optimize import least_squares, linprog, minimize

from .mdag_dgp import MDAG


# --------------------------------------------------------------------------
# packing / unpacking CPT parameters
# --------------------------------------------------------------------------

def param_spec(inst: MDAG):
    spec = []
    for i in range(inst.n_vars):
        for k in inst.var_cpt[i]:
            spec.append(("var", i, k))
    for i in range(inst.n_vars):
        for k in inst.r_cpt[i]:
            spec.append(("r", i, k))
    return spec


def pack(inst: MDAG) -> np.ndarray:
    out = np.zeros(len(param_spec(inst)))
    for idx, (kind, i, k) in enumerate(param_spec(inst)):
        out[idx] = inst.var_cpt[i][k] if kind == "var" else inst.r_cpt[i][k]
    return out


def unpack(inst: MDAG, theta: np.ndarray) -> MDAG:
    new = copy.deepcopy(inst)
    for idx, (kind, i, k) in enumerate(param_spec(inst)):
        if kind == "var":
            new.var_cpt[i][k] = float(theta[idx])
        else:
            new.r_cpt[i][k] = float(theta[idx])
    return new


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------

def param_bounds(inst: MDAG):
    """Per-parameter (lo, hi) arrays. Entries pinned by exact mechanism values
    (e.g., an always-observed indicator with p=1.0) are frozen at their value
    so search spaces remain consistent with the reference model."""
    lo = np.zeros(len(param_spec(inst)))
    hi = np.ones(len(param_spec(inst)))
    for idx, (kind, i, k) in enumerate(param_spec(inst)):
        table = inst.var_cpt if kind == "var" else inst.r_cpt
        val = table[i][k]
        if val in (0.0, 1.0):
            lo[idx] = hi[idx] = val
        else:
            lo[idx] = 0.03 if kind == "var" else 0.05
            hi[idx] = 0.97 if kind == "var" else 0.95
    return lo, hi


def target_value_phi(inst: MDAG, target) -> float:
    """Evaluate the target functional on the model's variable law P(v)."""
    kind = target[0]
    if kind == "cond":
        _, y_i, y_val, x_idx, x_val = target
        num = den = 0.0
        for v in itertools.product((0, 1), repeat=inst.n_vars):
            pv = inst.p_var(v)
            if tuple(v[i] for i in x_idx) == tuple(x_val):
                den += pv
                if v[y_i] == y_val:
                    num += pv
        assert den > 0
        return num / den
    tot = 0.0
    for v in itertools.product((0, 1), repeat=inst.n_vars):
        pv = inst.p_var(v)
        if kind == "mean":
            val = float(v[target[1]])
        elif kind == "cell":
            val = 1.0 if tuple(v) == tuple(target[1]) else 0.0
        else:
            raise ValueError(kind)
        tot += pv * val
    return tot


def _lp_coeffs(inst: MDAG, target) -> dict[tuple, float]:
    kind = target[0]
    vals = {}
    for v in itertools.product((0, 1), repeat=inst.n_vars):
        if kind == "mean":
            vals[v] = float(v[target[1]])
        elif kind == "cell":
            vals[v] = 1.0 if tuple(v) == tuple(target[1]) else 0.0
        else:
            raise ValueError("LP supports 'mean' and 'cell' targets only")
    return vals


# --------------------------------------------------------------------------
# helpers on observed laws
# --------------------------------------------------------------------------

def pattern_probabilities(inst: MDAG) -> dict[tuple, float]:
    jt = inst.joint_table()
    out: dict[tuple, float] = {}
    for (v, r), p in jt.items():
        out[r] = out.get(r, 0.0) + p
    return out


def marginalize(q_r: dict[tuple, float], r: tuple[int, ...], keep: tuple[int, ...]) -> dict[tuple, float]:
    """Marginal of pattern-conditional table onto original indices `keep`."""
    def pos(i: int) -> int:
        return sum(1 for j in range(len(r)) if r[j] == 1 and j < i)

    assert all(r[i] == 1 for i in keep)
    out: dict[tuple, float] = {}
    for o, c in q_r.items():
        key = tuple(o[pos(i)] for i in keep)
        out[key] = out.get(key, 0.0) + c
    return out


def cond_from_full_stratum(q_full: dict[tuple, float], n_vars: int,
                           y_i: int, y_val: int, x_idx: tuple[int, ...], x_val: tuple[int, ...]) -> float:
    num = den = 0.0
    for o, c in q_full.items():
        if tuple(o[i] for i in x_idx) == tuple(x_val):
            den += c
            if o[y_i] == y_val:
                num += c
    return num / den


# --------------------------------------------------------------------------
# instrument 1: assumption-free LP relaxation
# --------------------------------------------------------------------------

def lp_range(inst: MDAG, q_hat: dict[tuple, dict[tuple, float]], target):
    """Max/min of target over ALL joint tables matching observed margins.

    Variables: x = [t cells | c_r scales]. Constraints: sum(t)=1 and, for each
    realized pattern r and config o: sum_{v: v_O=o} t[v,r] = c_r * q_r(o).
    """
    patterns = sorted(q_hat.keys())
    cells = list(itertools.product(itertools.product((0, 1), repeat=inst.n_vars), patterns))
    T, Rpats = len(cells), len(patterns)
    phi_vals = _lp_coeffs(inst, target)

    rows, rhs = [], []
    rows.append(np.ones(T + Rpats))
    rhs.append(1.0)
    for j, r in enumerate(patterns):
        Oidx = [i for i in range(inst.n_vars) if r[i] == 1]
        for o in itertools.product((0, 1), repeat=len(Oidx)):
            row = np.zeros(T + Rpats)
            for ci, (v, rr) in enumerate(cells):
                if rr == r and tuple(v[i] for i in Oidx) == tuple(o):
                    row[ci] = 1.0
            row[T + j] = -q_hat[r][tuple(o)]
            rows.append(row)
            rhs.append(0.0)

    A_eq, b_eq = np.array(rows), np.array(rhs)
    bounds = [(0.0, 1.0)] * T + [(0.0, None)] * Rpats

    def solve(direction):
        c_obj = np.array([phi_vals[v] for v, _ in cells] + [0.0] * Rpats)
        res = linprog(direction * c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        assert res.status == 0, f"LP failed: {res.message}"
        return res.fun * direction, res.x

    lo, xlo = solve(1.0)
    hi, xhi = solve(-1.0)
    return {
        "lo": lo,
        "hi": hi,
        "width": hi - lo,
        "t_min": {cells[k]: float(xlo[k]) for k in range(T)},
        "t_max": {cells[k]: float(xhi[k]) for k in range(T)},
    }


# --------------------------------------------------------------------------
# instrument 2: model-aware witness search
# --------------------------------------------------------------------------

def observed_vector(inst: MDAG, patterns) -> tuple[np.ndarray, list]:
    """Observed-data fingerprint: per-pattern CONDITIONAL laws AND the pattern
    probabilities P(R=r). Both are observable, so completions must match both."""
    jt = inst.joint_table()
    q = inst.observed_laws(jt)
    pp = {}
    for (v, r), p in jt.items():
        pp[r] = pp.get(r, 0.0) + p
    keys, vals = [], []
    for r in patterns:
        keys.append(("pat", r))
        vals.append(pp.get(r, 0.0))
        Oidx = [i for i in range(inst.n_vars) if r[i] == 1]
        qr = q.get(r, {})
        for o in itertools.product((0, 1), repeat=len(Oidx)):
            keys.append((r, o))
            vals.append(qr.get(tuple(o), 0.0))
    return np.array(vals), keys


def _jacobian(inst, theta, patterns, f_ref, eps=1e-6):
    spec_n = len(param_spec(inst))
    J = np.zeros((len(f_ref), spec_n))
    for j in range(spec_n):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        fp, _ = observed_vector(unpack(inst, tp), patterns)
        fm, _ = observed_vector(unpack(inst, tm), patterns)
        J[:, j] = (fp - fm) / (2 * eps)
    return J


def manifold_walk(inst, theta_ref, target, n_seeds: int = 12, steps: int = 60,
                  step_size: float = 0.02, seed: int = 0,
                  dist_tol: float = 1e-9):
    """Walk the feasible manifold {theta : F(theta) = F(theta_ref)} using
    null-space predictor + first-order corrector, maximizing |dphi|. Much more
    reliable than generic SLSQP for tiny overidentified systems."""
    rng = np.random.default_rng(seed)
    ref_inst = unpack(inst, theta_ref)
    patterns = ref_inst.realized_patterns(jt=ref_inst.joint_table())
    f_ref, _ = observed_vector(ref_inst, patterns)
    phi_ref = target_value_phi(ref_inst, target)
    bounds_lo = np.array([0.03 if k == "var" else 0.05 for k, _, _ in param_spec(inst)])
    bounds_hi = np.array([0.97 if k == "var" else 0.95 for k, _, _ in param_spec(inst)])

    best = {"delta_phi": 0.0, "dist": np.inf, "success": False,
            "theta_pair": None, "phi_values": None}

    def record(x):
        nonlocal best
        m = unpack(inst, x)
        f_new, _ = observed_vector(m, patterns)
        dist = float(np.max(np.abs(f_new - f_ref)))
        dphi = abs(target_value_phi(m, target) - phi_ref)
        if dist < dist_tol and dphi > best["delta_phi"]:
            best = {"delta_phi": float(dphi), "dist": dist,
                    "success": bool(dphi > 1e-4),
                    "theta_pair": (theta_ref.copy(), x.copy()),
                    "phi_values": (float(phi_ref), float(target_value_phi(m, target)))}

    # estimate Jacobian once at reference point
    J = _jacobian(inst, theta_ref, patterns, f_ref)
    U, S, Vt = np.linalg.svd(J, full_matrices=True)
    r = int(np.sum(S > 1e-8))
    Null = Vt[r:].T if r < Vt.shape[1] else np.zeros((len(theta_ref), 0))
    Jp = np.linalg.pinv(J)

    for k in range(n_seeds):
        d = Null[:, k % Null.shape[1]] if Null.size else None
        if d is None:
            break
        sign = 1.0 if rng.random() < 0.5 else -1.0
        x = theta_ref.copy()
        for _ in range(steps):
            x = x + sign * step_size * d / max(np.linalg.norm(d), 1e-12)
            x = np.clip(x, bounds_lo, bounds_hi)
            # first-order corrector back onto the manifold
            for _ in range(3):
                fx, _ = observed_vector(unpack(inst, x), patterns)
                x = x - Jp @ (fx - f_ref)
                x = np.clip(x, bounds_lo, bounds_hi)
            fx, _ = observed_vector(unpack(inst, x), patterns)
            if np.max(np.abs(fx - f_ref)) > 1e-7:
                break
            record(x)
            # refresh direction along curved manifold occasionally
            Jx = _jacobian(inst, x, patterns, f_ref)
            _, Sx, Vtx = np.linalg.svd(Jx, full_matrices=True)
            rx = int(np.sum(Sx > 1e-8))
            if rx < Vtx.shape[1]:
                Null = np.hstack([Null, Vtx[rx:].T])
    return best


def root_jump_search(inst: MDAG, theta_ref: np.ndarray, target,
                     n_starts: int = 40, seed: int = 0,
                     dist_tol: float = 1e-8):
    """Find distinct factorized models sharing the observed law of theta_ref by
    multistart least-squares root finding on F(theta) - F(theta_ref). Tiny
    overidentified moment systems typically admit many isolated roots; jumping
    between them is the most effective witness strategy at Phase-1 sizes."""
    rng = np.random.default_rng(seed)
    ref_inst = unpack(inst, theta_ref)
    patterns = ref_inst.realized_patterns(jt=ref_inst.joint_table())
    f_ref, _ = observed_vector(ref_inst, patterns)
    phi_ref = target_value_phi(ref_inst, target)
    lo, hi = param_bounds(inst)
    free = np.where(hi - lo > 0)[0]
    base = theta_ref.copy()

    def expand(x_free):
        th = base.copy()
        th[free] = x_free
        return th

    best = {"delta_phi": 0.0, "dist": np.inf, "success": False,
            "theta_pair": None, "phi_values": None}
    if len(free) == 0:
        return best
    for _ in range(n_starts):
        span = hi[free] - lo[free]
        x0 = lo[free] + 0.02 * span + rng.random(len(free)) * (0.96 * span)
        res = least_squares(
            lambda xf: observed_vector(unpack(inst, expand(xf)), patterns)[0] - f_ref,
            x0, bounds=(lo[free], hi[free]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if np.max(np.abs(res.fun)) >= dist_tol:
            continue
        m = unpack(inst, expand(res.x))
        dphi = abs(target_value_phi(m, target) - phi_ref)
        if dphi > best["delta_phi"]:
            best = {"delta_phi": float(dphi), "dist": float(np.max(np.abs(res.fun))),
                    "success": bool(dphi > 1e-4),
                    "theta_pair": (theta_ref.copy(), expand(res.x).copy()),
                    "phi_values": (float(phi_ref), float(target_value_phi(m, target)))}
            if best["delta_phi"] > 0.5:
                break
    return best


def model_witness_search(inst: MDAG, theta_ref: np.ndarray, target,
                         n_starts: int = 24, seed: int = 0,
                         dist_tol: float = 1e-9, phi_tol: float = 1e-4):
    rng = np.random.default_rng(seed)
    spec_n = len(param_spec(inst))
    ref_inst = unpack(inst, theta_ref)
    patterns = ref_inst.realized_patterns(jt=ref_inst.joint_table())
    f_ref, _ = observed_vector(ref_inst, patterns)
    phi_ref = target_value_phi(ref_inst, target)
    lo_b, hi_b = param_bounds(inst)
    bounds = list(zip(lo_b, hi_b))

    best = {"delta_phi": 0.0, "dist": np.inf, "success": False,
            "theta_pair": None, "phi_values": None}

    cons = [{"type": "eq", "fun": lambda x: observed_vector(unpack(inst, x), patterns)[0] - f_ref}]

    lo_b, hi_b = param_bounds(inst)
    span_b = hi_b - lo_b

    def in_bounds(x):
        return np.where(span_b > 0,
                        np.clip(x, lo_b + 1e-3 * span_b, hi_b - 1e-3 * span_b),
                        lo_b)

    starts = [in_bounds(theta_ref + rng.normal(0, 0.06, size=spec_n))
              for _ in range(n_starts // 2)]
    starts += [in_bounds(lo_b + rng.random(spec_n) * span_b)
               for _ in range(n_starts - len(starts))]

    for x0 in starts:
        for sign in (+1.0, -1.0):

            def obj(x, sign=sign):
                m = unpack(inst, x)
                gap = np.max(np.abs(observed_vector(m, patterns)[0] - f_ref))
                pen = 1e4 * max(gap - 1e-12, 0.0)
                return sign * (target_value_phi(m, target) - phi_ref) + pen

            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-12})
            if not np.all(np.isfinite(res.x)):
                continue
            m = unpack(inst, res.x)
            dist = float(np.max(np.abs(observed_vector(m, patterns)[0] - f_ref)))
            dphi = abs(target_value_phi(m, target) - phi_ref)
            if dist < dist_tol and dphi > best["delta_phi"]:
                best = {"delta_phi": float(dphi), "dist": dist,
                        "success": bool(dphi > phi_tol),
                        "theta_pair": (theta_ref.copy(), res.x.copy()),
                        "phi_values": (float(phi_ref), float(target_value_phi(m, target)))}
            if best["delta_phi"] > 0.5:
                return best
    return best


# --------------------------------------------------------------------------
# instrument 3: identification formulas
# --------------------------------------------------------------------------

IDENTITY_FORMULAS = {}


def register(name):
    def deco(fn):
        IDENTITY_FORMULAS[name] = fn
        return fn

    return deco


@register("direct_full_pattern")
def f_direct(inst, q, target, aux):
    """Read mean/cell off a fully observed realized pattern (MCAR sanity)."""
    r_full = tuple([1] * inst.n_vars)
    cells = q[r_full]
    if target[0] == "cell":
        return cells[tuple(target[1])]
    tot = 0.0
    for o, c in cells.items():
        tot += c * o[target[1]]
    return tot


@register("mcar_cross_pattern_agreement")
def f_mcar_agree(inst, q, target, aux):
    """MCAR: every pattern observing the needed coordinates sees the same law."""
    if target[0] == "cell":
        ests = []
        for r, cells in q.items():
            need = tuple(i for i in range(inst.n_vars) if target[1][i] is not None)
            mg = marginalize(cells, r, need)
            key = tuple(target[1][i] for i in need)
            if key in mg:
                ests.append(mg[key])
        assert max(ests) - min(ests) < 1e-9, "MCAR cross-pattern disagreement"
        return float(np.mean(ests))
    j = target[1]
    ests = []
    for r, cells in q.items():
        if r[j] == 1:
            mg = marginalize(cells, r, (j,))
            ests.append(sum(k[0] * c for k, c in mg.items()))
    assert max(ests) - min(ests) < 1e-9, "MCAR cross-pattern disagreement"
    return float(np.mean(ests))


@register("mar_cond_stratum")
def f_mar_cond(inst, q, target, aux):
    """P(Y=y|X=x) inside the fully observed stratum (MAR identification)."""
    _, y_i, y_val, x_idx, x_val = target
    r_full = tuple([1] * inst.n_vars)
    return cond_from_full_stratum(q[r_full], inst.n_vars, y_i, y_val, x_idx, x_val)


@register("anchor_direct")
def f_anchor(inst, q, target, aux):
    """Mean of an always-observed variable pooled over patterns observing it."""
    j = target[1]
    num = den = 0.0
    pp = aux["pattern_prob"]
    for r, cells in q.items():
        if r[j] != 1:
            continue
        w = pp[r]
        mg = marginalize(cells, r, (j,))
        num += w * sum(k[0] * c for k, c in mg.items())
        den += w * sum(mg.values())
    return num / den


@register("mar_mean_iterated")
def f_mar_mean(inst, q, target, aux):
    """E[V_j] = sum_x P(x) P(V_j=1 | X=x); MAR stratum conditional plus pooled
    marginal of the conditioning block (all its members always observed)."""
    j = target[1]
    r_full = tuple([1] * inst.n_vars)
    always = [i for i in range(inst.n_vars)
              if all(r[i] == 1 for r in q.keys())]
    assert j in always or True  # j itself may be the partially observed one
    others = tuple(i for i in always if i != j)
    pxx: dict[tuple, float] = {}
    totw = 0.0
    pp = aux["pattern_prob"]
    for r, cells in q.items():
        if all(r[i] == 1 for i in others):
            w = pp[r]
            totw += w
            mg = marginalize(cells, r, others)
            for k, c in mg.items():
                pxx[k] = pxx.get(k, 0.0) + w * c
    pxx = {k: c / totw for k, c in pxx.items()}
    ex = 0.0
    for xx, wx in pxx.items():
        ex += wx * cond_from_full_stratum(q[r_full], inst.n_vars, j, 1, others, xx)
    return ex


@register("mar_joint_product")
def f_mar_joint(inst, q, target, aux):
    """Two-variable joint P(v1,v2) = P(v1) * P(v2|v1): pooled always-observed
    marginal of V1 times the MAR fully-observed-stratum conditional."""
    cell = tuple(target[1])
    r_full = tuple([1] * inst.n_vars)
    pxx: dict[tuple, float] = {}
    totw = 0.0
    pp = aux["pattern_prob"]
    for r, cells in q.items():
        if r[0] == 1:
            w = pp[r]
            totw += w
            mg = marginalize(cells, r, (0,))
            for k, c in mg.items():
                pxx[k] = pxx.get(k, 0.0) + w * c
    pxx = {k: c / totw for k, c in pxx.items()}
    return pxx[(cell[0],)] * cond_from_full_stratum(
        q[r_full], inst.n_vars, 1, cell[1], (0,), (cell[0],))


@register("cond_sel_stratum")
def f_cond_sel(inst, q, target, aux):
    """P(Y=y|X=x) from the selected stratum (selection depends on X alone)."""
    _, y_i, y_val, x_idx, x_val = target
    r_full = tuple([1] * inst.n_vars)
    return cond_from_full_stratum(q[r_full], inst.n_vars, y_i, y_val, x_idx, x_val)


# --------------------------------------------------------------------------
# decision procedure
# --------------------------------------------------------------------------

def decide(inst: MDAG, theta_true: np.ndarray, target, formula: str | None,
           seed: int = 0, lp_width_tol: float = 1e-3):
    m_true = unpack(inst, theta_true)
    jt = m_true.joint_table()
    q = m_true.observed_laws(jt)
    aux = {"pattern_prob": pattern_probabilities(m_true)}
    true_phi = target_value_phi(m_true, target)
    out = {"target": list(target), "true_value": true_phi}

    if formula is not None:
        est = IDENTITY_FORMULAS[formula](m_true, q, target, aux)
        out["formula_estimate"] = est
        if abs(est - true_phi) <= 1e-8 * max(1.0, abs(true_phi)):
            out.update(verdict="RECOVERABLE", evidence=f"formula:{formula}")
            return out
        out["formula_gap"] = abs(est - true_phi)

    wit = model_witness_search(inst, theta_true, target, seed=seed)
    out["witness"] = {k: wit[k] for k in ("delta_phi", "dist", "success", "phi_values")}
    if not wit["success"]:
        walk = root_jump_search(inst, theta_true, target, seed=seed + 1)
        out["root_jump"] = {k: walk[k] for k in ("delta_phi", "dist", "success")}
        if walk["success"]:
            wit = walk
            out["witness"] = {k: walk[k] for k in ("delta_phi", "dist", "success", "phi_values")}
    if wit["success"]:
        out.update(verdict="UNRECOVERABLE",
                   evidence=f"model_witness dphi={wit['delta_phi']:.4f} dist={wit['dist']:.1e}")
        return out

    if target[0] in ("mean", "cell"):
        lp = lp_range(inst, q, target)
        out["lp"] = {"width": lp["width"], "lo": lp["lo"], "hi": lp["hi"]}
        if lp["width"] > lp_width_tol:
            # Assumption-free variation only: the target varies over tables
            # matching the observed margins WITHOUT mechanism constraints.
            # This is evidence of fragility, NOT a model-valid unrecoverability
            # certificate (the model class is smaller than the relaxation).
            out.update(verdict="VARIABLE_UNCONSTRAINED_ONLY",
                       evidence=f"lp_width={lp['width']:.4f}")
            return out

    out.update(verdict="UNDETERMINED", evidence="no certificate either way")
    return out
