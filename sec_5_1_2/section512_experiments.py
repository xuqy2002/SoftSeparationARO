"""Section 5.1.2 continuous-first-stage runtime experiments.
"""


from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import time
import traceback
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse
import gurobipy as gp
from gurobipy import GRB

try:
    from . import benchmark as _benchmark
except ImportError:
    import benchmark as _benchmark


class _RuntimeAPI:

    def __getattr__(self, name):
        return globals()[name]


_BENCHMARK_RUNTIME = _RuntimeAPI()

VERSION = "section512-runtime-v3"
@dataclass(frozen=True)
class Settings:
    gap: float = 1e-3
    time_limit: float = 1800.0
    reference_gap: float = 1e-4
    reference_time_limit: object = 3600.0
    reference_max_iter: int = 30000
    ccg_max_iter: int = 30000
    ours_max_iter: int = 30000
    mh_steps_per_update: int = 1

    check_every: int = 20
    check_time_limit: object = 300.0
    check_final: bool = True
    progress_every: int = 1
    solver_tol: float = 1e-4
    resume: bool = True

CFG = Settings()

W = 1e-3
ELLIPSOID_DIMS = (3, 10, 30, 50, 70, 100)
SCENARIO_DIMS = (3, 30, 100)
SCENARIO_SIZES = (1000, 5000, 10000, 50000, 100000)
BUDGET_DIMS = (10, 30, 50, 70, 100)
BUDGET_GAMMAS = (0.1, 0.3, 0.5, 0.7, 0.9)
REPS = tuple(range(1, 6))

BASE = Path(__file__).resolve().parent
UNIFIED_NOTEBOOK_PATH = BASE / "Section_5_1_2_Runtime_Experiments.ipynb"
OUTPUT_DIR = BASE / "results"
REFERENCE_CACHE_DIR = OUTPUT_DIR / "reference_cache"
RUN_DIR = OUTPUT_DIR / VERSION


# ## 1. Common instance and uncertainty generation
# 
# For fixed (dimension, replication), instance data are identical across tables and I.
# Finite scenario sets use different reproducible seeds for each I and are not nested.
# For every dimension, the ellipsoid has center $0.5\mathbf 1$, independently sampled
# semi-axes in $[0.10,0.50]$, and an independent Haar rotation. Surface sampling
# uses the Jacobian rejection correction `min(a)*||u/a||`, Budget experiments use $\mathbf 1^\top\xi\leq\Gamma d$.
# In[ ]:


def ellipsoid_parameters(dim, rng):
    if int(dim) != dim or dim < 1:
        raise ValueError("dim must be a positive integer")
    center = np.full(dim, 0.5)
    axes = rng.uniform(0.10, 0.50, dim)
    # Gaussian QR with positive R diagonal gives Haar O(d); fix orientation for SO(d).
    rotation, triangular = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.where(np.diag(triangular) < 0, -1.0, 1.0)
    rotation *= signs
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    return center, axes, rotation

def ellipsoid_from_unit(inst, unit):
    return inst["center"] + np.asarray(unit) @ inst["transform"].T

def ellipsoid_to_unit(inst, point):
    return ((np.asarray(point)-inst["center"]) @ inst["rotation"]) / inst["axes"]

def ellipsoid_coordinate_radii(inst):
    return np.linalg.norm(inst["transform"], axis=1)

def ellipsoid_support_point(inst, direction):
    unit_direction = inst["transform"].T @ np.asarray(direction)
    norm = np.linalg.norm(unit_direction)
    return (ellipsoid_from_unit(inst, unit_direction/norm)
            if norm > 1e-12 else inst["center"].copy())

def make_instance(dim, rep):
    seed = 100000 * int(dim) + 1000 * int(rep)
    rng = np.random.default_rng(seed)
    fixed = rng.integers(300, 400, dim).astype(float)
    capacity = rng.integers(10, 30, dim).astype(float)
    transport = rng.integers(20, 40, (dim, dim)).astype(float)
    base = rng.integers(200, 300, dim).astype(float)
    sensitivity = 0.1 * rng.integers(1, 5, dim) * base
    geometry_rng = np.random.default_rng(np.random.SeedSequence([seed, 20260916]))
    center, axes, rotation = ellipsoid_parameters(dim, geometry_rng)
    return dict(dim=dim, rep=rep, seed=seed, c=np.r_[fixed / 20, capacity],
                transport=transport, base=base, sensitivity=sensitivity,
                emergency=float(rng.integers(100, 201)), y_upper=20.0, slope=40.0,
                center=center, axes=axes, rotation=rotation, transform=rotation*axes)

def sample_surface(inst, rng, size=1):
    chunks = []
    remaining = int(size)
    while remaining:
        count = min(max(remaining * 2, 8), 8192)
        u = rng.normal(size=(count, inst["dim"]))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        accept = rng.random(count) <= np.min(inst["axes"]) * np.linalg.norm(u / inst["axes"], axis=1)
        take = u[accept][:remaining]
        chunks.append(ellipsoid_from_unit(inst, take))
        remaining -= len(take)
    return np.concatenate(chunks)

def make_scenarios(inst, count):
    return sample_surface(inst, np.random.default_rng(inst["seed"] + int(count) + 17), count)

def demand(inst, xi):
    return inst["base"] + inst["sensitivity"] * np.asarray(xi)

def project_x(x, inst):
    # Closest point on each triangle: bottom, right edge, or sloping edge.
    d, K, upper = inst["dim"], inst["slope"], inst["y_upper"]
    a, b = np.asarray(x[:d]), np.asarray(x[d:])
    line_y = np.clip((a + K * b) / (1 + K*K), 0, upper)
    candidates = np.stack([
        np.column_stack([np.clip(a, 0, upper), np.zeros(d)]),
        np.column_stack([np.full(d, upper), np.clip(b, 0, K*upper)]),
        np.column_stack([line_y, K*line_y])], axis=1)
    feasible = (a >= 0) & (a <= upper) & (b >= 0) & (b <= K*a)
    distance = np.sum((candidates - np.column_stack([a, b])[:, None, :])**2, axis=2)
    chosen = candidates[np.arange(d), np.argmin(distance, axis=1)]
    chosen[feasible] = np.column_stack([a, b])[feasible]
    return np.r_[chosen[:, 0], chosen[:, 1]]

def finite_number(value):
    try:
        return math.isfinite(float(value)) and abs(float(value)) < 1e90
    except (TypeError, ValueError):
        return False

def bound_gap(lower, upper):
    if not (finite_number(lower) and finite_number(upper)):
        return math.inf
    if lower > upper + 1e-7 * max(1.0, abs(upper)):
        return math.inf
    return max(0.0, upper - lower) / max(1.0, abs(lower))

def deadline_after(seconds):
    return math.inf if seconds is None else time.perf_counter() + float(seconds)

def remaining(deadline):
    return max(0.0, deadline - time.perf_counter())

class BudgetExpired(RuntimeError):
    pass

def optimize_before(model, deadline):
    left = remaining(deadline)
    if left <= 0:
        raise BudgetExpired("Time budget exhausted")
    model.Params.TimeLimit = left if math.isfinite(left) else GRB.INFINITY
    model.optimize()

def new_model(name):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = CFG.solver_tol
    model.Params.OptimalityTol = CFG.solver_tol
    model.Params.MIPGap = CFG.reference_gap
    model.Params.MIPGapAbs = 1e-8
    return model

def solver_bounds(model):
    value = float(model.ObjVal) if model.SolCount else math.nan
    try:
        bound = float(model.ObjBound)
    except (AttributeError, gp.GurobiError):
        bound = value if model.Status == GRB.OPTIMAL else math.nan
    if model.Status == GRB.OPTIMAL and not finite_number(bound):
        bound = value
    return value, bound


# ## 2. Shared exact recourse, separation, and reference solvers
# ### Separation and external checks
# 



class Recourse:
    def __init__(self, inst, *, deterministic=False, select_min_l1=False):
        self.inst = inst
        self.deterministic = bool(deterministic)
        self.select_min_l1 = bool(select_min_l1)
        d = inst["dim"]
        self.model = new_model("exact_recourse_lp")
        if self.deterministic:
            # A fixed cold-start simplex selection makes dual-guided proposals history-independent.
            self.model.Params.Method = 1
            self.model.Params.Threads = 1
            self.model.Params.Seed = 0
        self.u = self.model.addVars(d, lb=0, ub=inst["emergency"], name="u")
        self.v = self.model.addVars(d, lb=0, ub=inst["emergency"], name="v")
        for i in range(d):
            for j in range(d):
                self.model.addConstr(self.v[j] - self.u[i] <= inst["transport"][i, j])
        self.model.update()

    def solve(self, x, xi, deadline=math.inf):
        d = self.inst["dim"]
        objective = gp.quicksum(float(demand(self.inst, xi)[j])*self.v[j] for j in range(d))
        objective -= gp.quicksum(float(x[d+i])*self.u[i] for i in range(d))
        self.model.setObjective(objective, GRB.MAXIMIZE)
        if self.deterministic:
            self.model.update()
            self.model.reset()
        optimize_before(self.model, deadline)
        if self.model.Status != GRB.OPTIMAL:
            if self.model.Status == GRB.TIME_LIMIT:
                raise BudgetExpired("Recourse solve timed out; MH transition not applied")
            raise RuntimeError("Recourse not optimal: " + str(self.model.Status))
        value = float(self.model.ObjVal)
        if not self.select_min_l1:
            self.last_scenario_gradient = self.inst["sensitivity"] * np.array([self.v[j].X for j in range(d)])
            return value, np.r_[np.zeros(d), [-self.u[i].X for i in range(d)]]
        # Select a minimum-L1 recourse subgradient on the original LP optimal face.
        face = self.model.addConstr(objective == value, name="optimal_face")
        try:
            self.model.setObjective(gp.quicksum(self.u.values()), GRB.MINIMIZE)
            self.model.reset()
            optimize_before(self.model, deadline)
            if self.model.Status != GRB.OPTIMAL:
                if self.model.Status == GRB.TIME_LIMIT:
                    raise BudgetExpired("Dual selection timed out; MH transition not applied")
                raise RuntimeError("Dual selection not optimal: " + str(self.model.Status))
            selected_value = float(objective.getValue())
            u = np.array([self.u[i].X for i in range(d)])
            v = np.array([self.v[j].X for j in range(d)])
            self.last_scenario_gradient = self.inst["sensitivity"] * v
            return selected_value, np.r_[np.zeros(d), -u]
        finally:
            self.model.remove(face)
            self.model.setObjective(objective, GRB.MAXIMIZE)
            self.model.update()

    def close(self):
        self.model.dispose()

def FiniteSeparation(inst, scenarios):
    return _benchmark.FiniteSeparation(_BENCHMARK_RUNTIME, inst, scenarios)


def EllipsoidKKTSeparation(inst):
    return _benchmark.EllipsoidKKTSeparation(_BENCHMARK_RUNTIME, inst)


def Separation(inst, scenarios=None):
    return _benchmark.separation(_BENCHMARK_RUNTIME, inst, scenarios)


def first_stage_model(inst, name):
    return _benchmark.first_stage_model(_BENCHMARK_RUNTIME, inst, name)


def add_scenario(model, x, eta, inst, xi, label):
    return _benchmark.add_scenario(
        _BENCHMARK_RUNTIME, model, x, eta, inst, xi, label)


def solve_ccg(inst, scenarios=None, target=None, time_limit=None, max_iter=None,
              verbose=False, reference=None):
    return _benchmark.solve_ccg(
        _BENCHMARK_RUNTIME, inst, scenarios, target, time_limit, max_iter,
        verbose, reference)


def extensive_reference(inst, scenarios):
    return _benchmark.extensive_reference(
        _BENCHMARK_RUNTIME, inst, scenarios)


REFERENCE_FORMULATION = "continuous_transport_emergency_uniform100_200_rotated_ellipsoid_v3"

def reference_key(inst, scenarios):
    # Independent of method, step size, temperature, output rule, and run VERSION.
    digest = hashlib.sha256(REFERENCE_FORMULATION.encode())
    for key in ("dim", "emergency", "y_upper", "slope"):
        digest.update(json.dumps([key,inst[key]]).encode())
    for key in ("c", "transport", "base", "sensitivity", "center", "axes", "rotation"):
        value = np.ascontiguousarray(inst[key],dtype=np.float64)
        digest.update(json.dumps([key,list(value.shape)]).encode())
        digest.update(value.tobytes())
    if scenarios is None:
        digest.update(b"ellipsoid")
    else:
        value = np.ascontiguousarray(scenarios,dtype=np.float64)
        digest.update(json.dumps(["finite",list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()

def usable_reference(saved):
    if not finite_number(saved.get("upper")):
        return False
    lo, hi = saved.get("lower"), saved["upper"]
    return not (finite_number(lo) and lo > hi+1e-7*max(1.0,abs(hi)))

def load_reference_file(path):
    try:
        saved = json.loads(path.read_text())
        return saved if isinstance(saved,dict) else None
    except (OSError,ValueError):
        return None

def legacy_reference_files(inst, scenarios):
    return []

def prepare_reference_result(saved, path, reused):
    result = dict(saved)
    result.setdefault("original_reference_status",result.get("status"))
    result["status"] = "reached" if bound_gap(result.get("lower"),result.get("upper"))<=CFG.reference_gap else "reference_unavailable"
    result["reference_reused"] = reused
    result["reference_cache_path"] = str(path)
    return result


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (tuple,list,np.ndarray)): return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    return value

PERSISTENCE_EXCLUDED_KEYS = {
    "x", "x_t", "x_average", "reference_solution", "x0",
    "metric_diagonal",
    "reference_cache_path", "reference_origin", "ccg_reuse_source",
    "output_dir", "directory",
}

def compact_record(value):
    if isinstance(value, dict):
        return {str(key): compact_record(item) for key, item in value.items()
                if key not in PERSISTENCE_EXCLUDED_KEYS}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [compact_record(item) for item in value]
    return value

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(json_safe(compact_record(data)), indent=2, allow_nan=False))
    temporary.replace(path)


ALGORITHM_NAMES = {"ours": "Algorithm 1", "ccg": "C&CG", "ldr": "LDR"}


def algorithm_record(algorithm, iteration, algorithm_sec, gap, **identifiers):
    return dict(
        identifiers,
        algorithm=algorithm,
        iteration=int(iteration) if iteration is not None else 1,
        algorithm_sec=algorithm_sec,
        gap=gap,
    )


def print_progress(algorithm, iteration, algorithm_sec, gap):
    print(
        f"algorithm={algorithm}, iteration={iteration}, "
        f"alg={algorithm_sec:.6f}s, gap={gap:.6g}",
        flush=True,
    )


def saved_result_record(method, result, **identifiers):
    gap = result.get("best_checked_gap")
    if not finite_number(gap):
        gap = result.get("gap")
    return algorithm_record(
        ALGORITHM_NAMES[method],
        result.get("iterations", 1),
        result.get("runtime_sec"),
        gap,
        **identifiers,
    )

def find_cached_reference(inst, scenarios):
    """return a usable matching reference"""
    key = reference_key(inst,scenarios)
    path = REFERENCE_CACHE_DIR / ("reference_"+key+".json")
    candidates = []
    saved = load_reference_file(path)
    if saved and saved.get("reference_fingerprint")==key and usable_reference(saved):
        candidates.append((path,saved))
    if not candidates:
        for old_path in legacy_reference_files(inst,scenarios):
            saved = load_reference_file(old_path)
            if not saved or not usable_reference(saved): continue
            expected = "ellipsoid" if scenarios is None else "finite"
            if saved.get("uncertainty") != expected: continue
            if saved.get("scenario_count") != (0 if scenarios is None else len(scenarios)): continue
            candidates.append((old_path,saved))
    if not candidates:
        return None
    origin,saved = min(candidates,key=lambda item:(
        bound_gap(item[1].get("lower"),item[1].get("upper"))>CFG.reference_gap,
        item[1]["upper"]))
    result = prepare_reference_result(saved,origin,True)
    result.update(reference_fingerprint=key,reference_formulation=REFERENCE_FORMULATION)
    result.setdefault("reference_origin",str(origin))
    return result

def get_reference(inst, scenarios):
    key = reference_key(inst,scenarios)
    path = REFERENCE_CACHE_DIR / ("reference_"+key+".json")
    # All newly solved references use CCG; finite separation enumerates every scenario LP.
    result = solve_ccg(inst,scenarios,target=CFG.reference_gap,
        time_limit=CFG.reference_time_limit,max_iter=CFG.reference_max_iter,verbose=True)
    result["reference_mode"] = "ellipsoid_ccg" if scenarios is None else "finite_ccg"
    result = prepare_reference_result(result,path,False)
    result.update(scenario_count=0 if scenarios is None else len(scenarios),
                  uncertainty="ellipsoid" if scenarios is None else "finite",
                  reference_fingerprint=key,reference_formulation=REFERENCE_FORMULATION)
    return result


# ## 3. Optimization and proposal rules
# 
# The implementation uses the fixed diagonal preconditioner and the paper's fixed
# update schedule, and linearly weighted average throughout.
# 
# 
# The proposal is either independent uniform sampling or the dual-guided vMF
# proposal.  The latter uses the selected recourse dual to define a direction
# and includes the Hastings correction.  
# 


def initial_x(inst, deadline, init_mode):
    d = inst["dim"]
    if init_mode == "midpoint":
        y = np.full(d, 0.5*inst["y_upper"])
        z = 0.5*inst["slope"]*y
        return np.r_[y, z]
    if init_mode not in ("support_demand_lp", "support_demand_lp_ball"):
        raise ValueError("unknown initialization rule")
    if init_mode == "support_demand_lp_ball":
        if inst.get("uncertainty") != "budget":
            raise ValueError("support_demand_lp_ball is only defined for budget sets")
        xi = budget_ball_support_point(inst, inst["sensitivity"])
    else:
        xi = ellipsoid_support_point(inst, inst["sensitivity"])
    model, x, eta = first_stage_model(inst,"single_scenario_initializer")
    try:
        add_scenario(model,x,eta,inst,xi,0)
        optimize_before(model,deadline)
        if not model.SolCount:
            raise RuntimeError("Initialization LP has no solution")
        return project_x(np.array([x[i].X for i in range(2*d)]),inst)
    finally:
        model.dispose()

def step_coefficient(inst, alpha0, metric_diagonal):
    if alpha0 is not None:
        if alpha0 <= 0: raise ValueError("alpha0 must be positive")
        return float(alpha0)
    d = inst["dim"]
    ranges = np.r_[np.full(d, inst["y_upper"]),
                   np.full(d, inst["y_upper"]*inst["slope"])]
    bounds = np.r_[np.abs(inst["c"][:d]),
        np.maximum(np.abs(inst["c"][d:]), np.abs(inst["c"][d:]-inst["emergency"]))]
    return np.sqrt(np.sum(metric_diagonal*ranges**2)) / (
        np.sqrt(np.sum(bounds**2/metric_diagonal))*np.sqrt(np.log1p(CFG.ours_max_iter)))

def stopping_reference(reference):
    if reference.get("status") == "reached" and finite_number(reference.get("lower")):
        return float(reference["lower"])
    if finite_number(reference.get("upper")):
        return float(reference["upper"])
    return None

def stopping_gap(candidate_upper, baseline):
    if not finite_number(candidate_upper) or not finite_number(baseline):
        return math.inf
    return max(0.0,(candidate_upper-baseline)/max(1.0,abs(baseline)))


def draw_vmf_direction(center, kappa, rng):
    """Wood rejection sampler on S^(d-1), followed by a Householder rotation."""
    center = np.asarray(center, dtype=float)
    d = len(center)
    if d < 2:
        raise ValueError("dual_vmf needs dimension >= 2")
    if kappa == 0:
        direction = rng.normal(size=d)
        return direction/np.linalg.norm(direction)
    # Stable form of (sqrt(4*kappa**2+(d-1)**2)-2*kappa)/(d-1).
    b = (d-1)/(np.hypot(2*kappa, d-1)+2*kappa)
    x0 = (1-b)/(1+b)
    c = kappa*x0+(d-1)*np.log1p(-x0*x0)
    while True:
        z = rng.beta((d-1)/2, (d-1)/2)
        w = (1-(1+b)*z)/(1-(1-b)*z)
        score = kappa*w+(d-1)*np.log1p(-x0*w)-c
        if np.log(max(rng.random(),np.finfo(float).tiny)) <= score:
            break
    tangent = rng.normal(size=d-1)
    tangent /= np.linalg.norm(tangent)
    point = np.r_[w, np.sqrt(max(0.,1-w*w))*tangent]
    axis = np.zeros(d)
    axis[0] = 1.
    delta = axis-center
    norm2 = float(delta@delta)
    if norm2 > 1e-24:
        point -= 2*delta*float(delta@point)/norm2
    return point/np.linalg.norm(point)

def dual_proposal_gradient(inst, recourse_gradient):
    # Use the cold-start LP selection of u; positive demands determine maximal feasible v.
    u = -np.asarray(recourse_gradient)[inst["dim"]:]
    v = np.minimum(inst["emergency"], np.min(inst["transport"]+u[:,None],axis=0))
    return inst["sensitivity"]*v

class ScenarioProposal:
    """Independent uniform or dual-guided vMF proposal."""
    def __init__(self, inst, scenarios, rng, mode, kappa=None):
        self.inst, self.scenarios, self.rng, self.mode = inst, scenarios, rng, mode
        self.kappa = 5.0*inst["dim"] if kappa is None else float(kappa)
        self.last_phase = "init"
        self.vmf_units = None
        self.vmf_log_forward = None
        if self.mode == "dual_vmf" and scenarios is not None:
            self.vmf_units = ellipsoid_to_unit(inst, scenarios)
            if not np.allclose(np.linalg.norm(self.vmf_units, axis=1), 1., atol=1e-7):
                raise ValueError("dual_vmf finite scenarios must lie on the ellipsoid surface")

    def uniform(self):
        if self.scenarios is None:
            return sample_surface(self.inst, self.rng, 1)[0]
        return self.scenarios[self.rng.integers(len(self.scenarios))].copy()

    def dual_center(self, scenario_gradient):
        direction = self.inst["transform"].T @ scenario_gradient
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            direction = np.zeros(self.inst["dim"])
            direction[0] = 1.
            return direction
        return direction/norm

    def discrete_log_weights(self, center):
        logits = self.kappa*(self.vmf_units@center)
        maximum = np.max(logits)
        return logits-maximum-np.log(np.exp(logits-maximum).sum())

    def dual_draw(self, scenario_gradient):
        center = self.dual_center(scenario_gradient)
        if self.scenarios is None:
            unit = draw_vmf_direction(center, self.kappa, self.rng)
            return ellipsoid_from_unit(self.inst, unit)
        logs = self.discrete_log_weights(center)
        index = self.rng.choice(len(logs), p=np.exp(logs))
        self.vmf_log_forward = float(logs[index])
        return self.scenarios[index].copy()

    def hastings_correction(self, current, candidate, grad_current, grad_candidate):
        if self.mode == "uniform":
            return 0.
        forward_center = self.dual_center(grad_current)
        reverse_center = self.dual_center(grad_candidate)
        if self.scenarios is not None:
            indices = np.flatnonzero(np.all(self.scenarios == current, axis=1))
            if not len(indices):
                raise ValueError("Current scenario is absent from finite set")
            reverse_logs = self.discrete_log_weights(reverse_center)
            return float(reverse_logs[indices[0]]-self.vmf_log_forward)
        u = ellipsoid_to_unit(self.inst, current)
        up = ellipsoid_to_unit(self.inst, candidate)
        log_area_ratio = (np.log(np.linalg.norm(up/self.inst["axes"]))
                          - np.log(np.linalg.norm(u/self.inst["axes"])))
        return float(log_area_ratio+self.kappa*(reverse_center@u-forward_center@up))

    def draw(self, current, scenario_gradient, iteration):
        if self.mode == "dual_vmf":
            self.last_phase = "dual_vmf"
            return self.dual_draw(scenario_gradient)
        self.last_phase = "uniform"
        return self.uniform()

def fixed_preconditioner(inst):
    """Fixed relative gradient scales; no samples, iterates, or references."""
    d = inst["dim"]
    bounds = np.r_[np.abs(inst["c"][:d]),
        np.maximum(np.abs(inst["c"][d:]), np.abs(inst["c"][d:]-inst["emergency"]))]
    bounds = np.maximum(bounds, 1e-8)
    # Normalize the common scale, leaving alpha0 as the global step coefficient.
    diagonal = bounds / np.exp(np.mean(np.log(bounds)))
    diagonal.setflags(write=False)
    return diagonal

def project_x_metric(x, inst, diagonal):
    """Exact diagonal-metric projection onto the product of capacity triangles."""
    d, K, upper = inst["dim"], inst["slope"], inst["y_upper"]
    x, diagonal = np.asarray(x, dtype=float), np.asarray(diagonal, dtype=float)
    if diagonal.shape != (2*d,) or not np.all(np.isfinite(diagonal) & (diagonal > 0)):
        raise ValueError("Metric diagonal must be finite, positive, and have length 2*dim")
    a, b = x[:d], x[d:]
    hy, hz = diagonal[:d], diagonal[d:]
    line_y = np.clip((hy*a + K*hz*b)/(hy + K*K*hz), 0, upper)
    candidates = np.stack([
        np.column_stack([np.clip(a, 0, upper), np.zeros(d)]),
        np.column_stack([np.full(d, upper), np.clip(b, 0, K*upper)]),
        np.column_stack([line_y, K*line_y])], axis=1)
    pairs = np.column_stack([a, b])
    distance = np.sum(np.column_stack([hy, hz])[:, None, :]*(candidates-pairs[:, None, :])**2, axis=2)
    chosen = candidates[np.arange(d), np.argmin(distance, axis=1)]
    feasible = (a >= 0) & (a <= upper) & (b >= 0) & (b <= K*a)
    chosen[feasible] = pairs[feasible]
    return np.r_[chosen[:, 0], chosen[:, 1]]


def dual_averaging_step(anchor, gradient_sum, alpha, inst, metric_diagonal):
    return project_x_metric(
        anchor-alpha*gradient_sum/metric_diagonal, inst, metric_diagonal)


def validate_algorithm_settings(init_mode, proposal_mode, alpha0, w, output_rule,
                                kappa):
    if init_mode not in ("support_demand_lp", "support_demand_lp_ball", "midpoint"):
        raise ValueError("invalid init_mode")
    if proposal_mode not in ("uniform", "dual_vmf"):
        raise ValueError("proposal_mode must be 'uniform' or 'dual_vmf'")
    if kappa is not None and (not math.isfinite(kappa) or not 0 <= kappa <= 1e4):
        raise ValueError("kappa must be None or finite and in [0, 10000]")
    if type(CFG.mh_steps_per_update) is not int or CFG.mh_steps_per_update < 1:
        raise ValueError("mh_steps_per_update must be a positive integer")
    if alpha0 is not None and not (math.isfinite(alpha0) and alpha0 > 0):
        raise ValueError("alpha0 must be None or positive and finite")
    if not math.isfinite(w) or w <= 0:
        raise ValueError("w must be finite and positive")
  
def sampled_timeout_check(inst, x):
    sample_seed = int(inst["seed"]) + 915031
    points = sample_surface(inst, np.random.default_rng(sample_seed), 10000)
    evaluator = FiniteSeparation(inst, points)
    try:
        result = evaluator.solve(x, math.inf)
    finally:
        evaluator.close()
    if not result["complete"] or not finite_number(result["value"]):
        raise RuntimeError("The 10,000-point fallback evaluation did not complete")
    result.update(upper=math.inf, evaluation_method="sampled_10000",
                  sample_seed=sample_seed, sample_count=10000)
    return result

def checked_separation(evaluator, inst, scenarios, x, deadline):
    # complete the full LP enumeration for external check
    if scenarios is not None:
        deadline = math.inf
    try:
        result = evaluator.solve(x, deadline)
    except BudgetExpired:
        result = dict(value=math.nan, upper=math.inf, xi=None,
                      solver_status=GRB.TIME_LIMIT, complete=False,
                      evaluation_method="kkt_timeout" if scenarios is None else "enumeration_incomplete")
    if scenarios is None and result["solver_status"] == GRB.TIME_LIMIT:
        fallback = sampled_timeout_check(inst, x)
        fallback.update(kkt_solver_status=result["solver_status"], kkt_value=result["value"])
        return fallback
    return result

def check_scope(method):
    if method == "sampled_10000":
        return "sampled_scenario_benchmark"
    if method == "enumeration":
        return "finite_scenario_benchmark"
    if method == "enumeration_incomplete":
        return "evaluation_incomplete"
    return "separation_objval_benchmark"

def check_details(result):
    return {key: result[key] for key in (
        "complete", "solver_status", "evaluated_count", "scenario_count",
        "sample_count", "sample_seed", "kkt_solver_status", "kkt_value") if key in result}

def solve_ours(inst, scenarios, reference, seed, trace_path=None, *, init_mode, w,
               proposal_mode, alpha0, output_rule, dual_vmf_kappa=None):
    from collections import deque

    validate_algorithm_settings(
        init_mode, proposal_mode, alpha0, w, output_rule, dual_vmf_kappa)
    dual_averaging_start_iter = 100 if inst.get("uncertainty") == "budget" else 1
    reference_value = stopping_reference(reference)
    if reference_value is None:
        raise ValueError("No finite reference bound is available")
    if proposal_mode == "dual_vmf":
        min_demand = (np.asarray(inst["base"])
            if inst.get("uncertainty") == "budget"
            else inst["base"]+inst["sensitivity"]*inst["center"]
                 -np.abs(inst["sensitivity"])*ellipsoid_coordinate_radii(inst))
        if not np.all(min_demand > 0):
            raise ValueError("dual_vmf requires positive demands over the ellipsoid")

    start = time.perf_counter()
    excluded = excluded_logging = 0.0
    iteration = checks = skipped = 0
    mh_transitions = mh_accepted = mh_oracle_solves = 0
    oracle = screen = sep = None
    x = xi = output_x = averaged = None
    last_upper = last_eval_value = math.inf
    last_gap = best_checked_gap = math.inf
    best_gap_iteration = None
    best_gap_evaluation_method = "not_checked"
    evaluation_method, fallback_checks, evaluation_details = "not_checked", 0, {}
    status, last_checked = "max_iter", -1

    metric_diagonal = fixed_preconditioner(inst)
    alpha0 = step_coefficient(inst, alpha0, metric_diagonal)

    weighted_sum = np.zeros(2*inst["dim"])
    weight_sum = 0.0
    average_points = deque()
    da_anchor = None
    da_gradient_sum = None
    trace = []
    rng = np.random.default_rng(seed)

    def elapsed():
        return time.perf_counter()-start-excluded-excluded_logging

    def algorithm_deadline():
        return start+excluded+excluded_logging+CFG.time_limit

    proposer = ScenarioProposal(
        inst, scenarios, rng, proposal_mode, dual_vmf_kappa)

    def accept_check_result(result):
        nonlocal best_checked_gap, best_gap_iteration, best_gap_evaluation_method
        nonlocal last_gap, last_upper, last_eval_value
        nonlocal evaluation_method, fallback_checks, evaluation_details
        evaluation_method = result["evaluation_method"]
        evaluation_details = check_details(result)
        if evaluation_method == "sampled_10000":
            fallback_checks += 1
        if evaluation_method == "enumeration_incomplete":
            return False
        if finite_number(result["upper"]):
            last_upper = float(inst["c"]@output_x)+result["upper"]
        if finite_number(result["value"]):
            last_eval_value = float(inst["c"]@output_x)+result["value"]
            last_gap = stopping_gap(last_eval_value, reference_value)
            if finite_number(last_gap) and last_gap < best_checked_gap:
                best_checked_gap = last_gap
                best_gap_iteration = iteration
                best_gap_evaluation_method = evaluation_method
        return last_gap <= CFG.gap

    def validate(force=False):
        nonlocal excluded, screen, sep, checks, skipped
        nonlocal last_gap, last_upper, last_checked, last_eval_value
        nonlocal evaluation_method, evaluation_details
        check_start = time.perf_counter()
        checks += 1
        last_checked = iteration
        last_upper = last_gap = last_eval_value = math.inf
        evaluation_method = "pending"
        evaluation_details = {}
        try:
            check_deadline = deadline_after(CFG.check_time_limit)
            if screen is None:
                screen = Recourse(inst)
            q, _ = screen.solve(output_x, xi, check_deadline)
            lower_candidate = float(inst["c"]@output_x)+q
            screen_gap = (lower_candidate-reference_value)/max(1.0, abs(reference_value))
            if not force and screen_gap > CFG.gap:
                skipped += 1
                evaluation_method = "screened_out"
                return False
            if sep is None:
                sep = Separation(inst, scenarios)
                if scenarios is None:
                    sep.model.Params.MIPFocus = 3
                    sep.model.Params.OBBT = 1
            result = checked_separation(sep, inst, scenarios, output_x, check_deadline)
            return accept_check_result(result)
        except BudgetExpired:
            if scenarios is None:
                return accept_check_result(sampled_timeout_check(inst, output_x))
            evaluation_method = "enumeration_incomplete"
            return False
        finally:
            excluded += time.perf_counter()-check_start

    try:
        x = initial_x(inst, algorithm_deadline(), init_mode)
        xi = proposer.uniform()
        oracle = Recourse(inst, deterministic=proposal_mode == "dual_vmf",
                          select_min_l1=True)

        for t in range(1, CFG.ours_max_iter+1):
            if elapsed() >= CFG.time_limit:
                raise BudgetExpired()
            qi, gi = oracle.solve(x, xi, algorithm_deadline())
            scenario_gradient = (dual_proposal_gradient(inst, gi)
                                 if proposal_mode == "dual_vmf"
                                 else oracle.last_scenario_gradient.copy())
            mh_oracle_solves += 1
            for _ in range(CFG.mh_steps_per_update):
                if elapsed() >= CFG.time_limit:
                    raise BudgetExpired()
                candidate = proposer.draw(xi, scenario_gradient, t)
                qj, gj = oracle.solve(x, candidate, algorithm_deadline())
                mh_oracle_solves += 1
                candidate_gradient = (dual_proposal_gradient(inst, gj)
                                      if proposal_mode == "dual_vmf"
                                      else oracle.last_scenario_gradient.copy())
                correction = proposer.hastings_correction(
                    xi, candidate, scenario_gradient, candidate_gradient)
                log_ratio = (qj-qi)/w+correction
                if math.log(max(rng.random(), np.finfo(float).tiny)) <= min(0.0, log_ratio):
                    xi, qi, gi = candidate, qj, gj
                    scenario_gradient = candidate_gradient
                    mh_accepted += 1
                mh_transitions += 1

            alpha = alpha0/math.sqrt(t)
            grad = inst["c"]+gi
            dual_averaging_active = t >= dual_averaging_start_iter
            if dual_averaging_active:
                if t == dual_averaging_start_iter:
                    da_anchor = x.copy()
                    da_gradient_sum = np.zeros_like(x)
                da_gradient_sum += grad
                x = dual_averaging_step(
                    da_anchor, da_gradient_sum, alpha, inst, metric_diagonal)
            else:
                x = project_x_metric(
                    x-alpha*grad/metric_diagonal, inst, metric_diagonal)

            average_weight = float(t)
            weighted_sum += average_weight*x
            weight_sum += average_weight
            if output_rule == "half_linear_average":
                average_points.append((t, average_weight, x.copy()))
                while len(average_points) > (t+1)//2:
                    _, old_weight, old_x = average_points.popleft()
                    weighted_sum -= old_weight*old_x
                    weight_sum -= old_weight
            averaged = weighted_sum/weight_sum
            iteration = t
            output_x = averaged.copy()
            last_upper = last_gap = last_eval_value = math.inf
            last_checked = -1
            evaluation_method = "not_checked"
            evaluation_details = {}

            if iteration % CFG.check_every == 0 and validate():
                status = "reached"
            if iteration % CFG.progress_every == 0 or status == "reached":
                entry = algorithm_record(
                    "Algorithm 1", iteration, elapsed(), best_checked_gap)
                trace.append(entry)
                logging_start = time.perf_counter()
                try:
                    print_progress(
                        entry["algorithm"], entry["iteration"],
                        entry["algorithm_sec"], entry["gap"])
                    if trace_path is not None:
                        write_json(trace_path, trace)
                finally:
                    excluded_logging += time.perf_counter()-logging_start
            if status == "reached":
                break

    except BudgetExpired:
        status = "time_limit"
    finally:
        if output_x is not None and status != "reached" and CFG.check_final:
            try:
                if validate(force=True):
                    status = "reached"
            except Exception:
                pass
        if oracle is not None:
            oracle.close()
        cleanup_start = time.perf_counter()
        if screen is not None:
            screen.close()
        if sep is not None:
            sep.close()
        excluded += time.perf_counter()-cleanup_start

    algorithm_time = elapsed()
    if status == "reached" and algorithm_time > CFG.time_limit:
        status = "time_limit"
    if status == "reached":
        status = "benchmark_reached"
    certified_gap = bound_gap(reference.get("lower"), last_upper)
    benchmark_gap = stopping_gap(last_eval_value, reference_value)
    logging_start = time.perf_counter()
    try:
        if trace_path is not None:
            write_json(trace_path, trace)
    finally:
        excluded_logging += time.perf_counter()-logging_start
    return dict(
        status=status, runtime_sec=algorithm_time,
        wall_time_sec=time.perf_counter()-start, excluded_check_time_sec=excluded,
        objective=(last_eval_value if finite_number(last_eval_value) else None),
        upper=last_upper,
        stopping_bound=("sample_max" if evaluation_method == "sampled_10000"
                        else "enumeration_max" if evaluation_method == "enumeration"
                        else "ObjVal"),
        evaluation_lower=last_eval_value,
        timing_policy="exclude_external_validation_and_logging",
        lower=reference.get("lower"), gap=last_gap, iterations=iteration,
        best_checked_gap=best_checked_gap, best_gap_iteration=best_gap_iteration,
        best_gap_evaluation_method=best_gap_evaluation_method,
        stopping_reference_value=reference_value,
        certification_scope=check_scope(evaluation_method),
        certified_gap=certified_gap, benchmark_gap=benchmark_gap,
        evaluation_method=evaluation_method, fallback_checks=fallback_checks,
        evaluation_details=evaluation_details, last_checked_iteration=last_checked,
        alpha0=alpha0, w=w, init_mode=init_mode, recourse_mode="lp",
        metric_rule="normalized_gradient_bounds", checks=checks,
        screen_skips=skipped, output_rule=output_rule,
        proposal_mode=proposal_mode, proposal_phase=proposer.last_phase,
        dual_vmf_kappa=proposer.kappa,
        hastings_corrected=proposal_mode == "dual_vmf",
        mh_steps_per_update=CFG.mh_steps_per_update,
        mh_transitions=mh_transitions, mh_accepted=mh_accepted,
        mh_oracle_solves=mh_oracle_solves)


# ## 4. Benchmark interfaces
#
# C&CG, exact separation, and LDR are implemented in benchmark.py. 


def add_soc_le(model, intercept, coefficients, rhs, inst):
    return _benchmark.add_soc_le(
        _BENCHMARK_RUNTIME, model, intercept, coefficients, rhs, inst)


def solve_ldr(inst, scenarios, reference):
    return _benchmark.solve_ldr(
        _BENCHMARK_RUNTIME, inst, scenarios, reference)


# ## 5. Unified execution, independent case records, and summaries
# 
# 



def summarize_records():
    rows = [json.loads(p.read_text()) for p in sorted(RUN_DIR.glob("case_*.json"))]
    if not rows: return pd.DataFrame(),pd.DataFrame()
    raw = pd.DataFrame(rows)
    raw.to_csv(RUN_DIR/"raw.csv",index=False)
    grouped = []
    columns = ["table", "dim", "scenario_size", "algorithm"]
    for key,group in raw.groupby(columns,dropna=False):
        times = pd.to_numeric(group["algorithm_sec"],errors="coerce")
        gaps = pd.to_numeric(group["gap"],errors="coerce")
        reached = (gaps <= CFG.gap) & (times <= CFG.time_limit)
        row = dict(zip(columns,key))
        row.update(record_count=len(group), reached_count=int(reached.sum()),
                   algorithm_sec_mean=times.mean(),algorithm_sec_std=times.std(ddof=1),
                   gap_mean=gaps.mean(),gap_max=gaps.max())
        grouped.append(row)
    summary = pd.DataFrame(grouped)
    summary.to_csv(RUN_DIR/"summary.csv",index=False)
    for table in ("table1_ellipsoid","table2_scenarios"):
        summary[summary.table.eq(table)].to_csv(RUN_DIR/(table+"_summary.csv"),index=False)
    return raw,summary

def reuse_ccg_result(inst, scenarios, reference, label):
    """Reuse observed UB hitting times; never manufacture an unrecorded crossing."""
    baseline = reference.get("upper")
    if not finite_number(baseline):
        return dict(status="reference_unavailable",runtime_sec=math.nan,gap=math.nan,ccg_reused=True)
    fingerprint = reference_key(inst,scenarios)
    denominator = max(1.0,abs(baseline))
    def gap(upper):
        return max(0.0,(upper-baseline)/denominator) if finite_number(upper) else math.inf
    def observed_result(row, origin, resolution, formulation):
        runtime, upper = row.get("runtime_sec"),row.get("upper")
        if not finite_number(runtime) or runtime < 0 or gap(upper)>CFG.gap:
            return None
        lower = row.get("lower")
        return dict(status="reached" if runtime<=CFG.time_limit else "time_limit",
            runtime_sec=float(runtime),lower=lower,upper=upper,objective=upper,gap=gap(upper),
            solver_gap=bound_gap(lower,upper),iterations=row.get("iteration",row.get("iterations")),
            reference_objective=baseline,ccg_reused=True,
            ccg_reuse_source=str(origin),ccg_timing_resolution=resolution,
            first_hitting_time_known=(resolution=="iteration_history"),
            ccg_formulation=formulation,instance_fingerprint=fingerprint)
    history = reference.get("ccg_history")
    if reference.get("reference_mode") in ("ellipsoid_ccg","finite_ccg") and history:
        for row in sorted(history,key=lambda v:v.get("runtime_sec",math.inf)):
            result = observed_result(row,reference.get("reference_cache_path","reference"),
                                     "iteration_history",reference.get("ccg_formulation","legacy_unspecified"))
            if result is not None:
                return result
    reference_names = {Path(str(reference.get(k,""))).name for k in
                       ("reference_cache_path","reference_origin") if reference.get(k)}
    records = []
    for path in OUTPUT_DIR.glob("*/case_"+label+"_ccg.json"):
        saved = load_reference_file(path)
        if not saved or saved.get("instance_seed")!=inst["seed"]:
            continue
        saved_key = saved.get("instance_fingerprint")
        if saved_key is not None:
            if saved_key!=fingerprint: continue
        elif Path(str(saved.get("reference_cache_path",""))).name not in reference_names:
            continue
        origin_parent = Path(str(reference.get("reference_origin",reference.get("reference_cache_path","")))).parent
        records.append((path.parent!=origin_parent,str(path),path,saved))
    for _,__,path,saved in sorted(records,key=lambda v:(v[0],v[1])):
        history = saved.get("ccg_history")
        if history:
            for row in sorted(history,key=lambda v:v.get("runtime_sec",math.inf)):
                result = observed_result(row,path,"iteration_history",saved.get("ccg_formulation","legacy_unspecified"))
                if result is not None: return result
        result = observed_result(saved,path,"recorded_endpoint",saved.get("ccg_formulation","legacy_unspecified"))
        if result is not None: return result
    return dict(status="reuse_timing_unavailable",runtime_sec=math.nan,gap=math.nan,
        objective=math.nan,ccg_reused=True,reference_objective=baseline,
        ccg_reuse_reason="No matching recorded UB hitting time; CCG was not rerun.",instance_fingerprint=fingerprint)

def _run_ellipsoid_scenario_cases(cases, *, alpha_0=1.0, kappa=None,
                                  output_name="ellipsoid_scenario"):
    """Run Algorithm 1, C&CG, and LDR together for the supplied cases."""
    global RUN_DIR
    if not math.isfinite(alpha_0) or alpha_0 <= 0:
        raise ValueError("alpha_0 must be finite and positive")
    if kappa is not None and (not math.isfinite(kappa) or kappa < 0):
        raise ValueError("kappa must be None or finite and nonnegative")
    validate_algorithm_settings(
        "support_demand_lp", "dual_vmf", alpha_0, W, "linear_average", kappa)
    if not (CFG.check_every>=1 and CFG.progress_every>=1):
        raise ValueError("Positive temperature and check intervals required")
    alpha_tag = f"{float(alpha_0):g}"
    kappa_tag = "5dim" if kappa is None else f"{float(kappa):g}"
    RUN_DIR = OUTPUT_DIR / output_name / VERSION / f"alpha_{alpha_tag}_kappa_{kappa_tag}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cases = list(cases)
    for number,(table,dim,I,rep) in enumerate(cases,1):
        label = f"{table}_d{dim}_I{I}_rep{rep}"
        pending = []
        for method in ("ours", "ccg", "ldr"):
            path = RUN_DIR/f"case_{label}_{method}.json"
            saved = json.loads(path.read_text()) if CFG.resume and path.exists() else {}
            required = {"algorithm", "iteration", "algorithm_sec", "gap"}
            if (not required.issubset(saved)
                    or saved.get("algorithm_sec") is None
                    or saved.get("gap") is None):
                pending.append(method)
        if not pending: continue
        inst = make_instance(dim,rep)
        scenarios = None if I==0 else make_scenarios(inst,I)
        try:
            reference = get_reference(inst,scenarios)
        except Exception as exc:
            reference = dict(status="reference_unavailable",lower=math.nan,upper=math.nan,error=repr(exc))
        for method in pending:
            attempt_start = time.perf_counter()
            try:
                if method=="ours":
                    if stopping_reference(reference) is None:
                        result = dict(status="reference_unavailable",runtime_sec=math.nan)
                    else:
                        result = solve_ours(inst,scenarios,reference,inst["seed"]+I+29,
                            RUN_DIR/f"trace_{label}.json", init_mode="support_demand_lp",
                            w=W, proposal_mode="dual_vmf", alpha0=alpha_0,
                            output_rule="linear_average",
                            dual_vmf_kappa=kappa)
                elif method=="ccg":
                    result = solve_ccg(inst,scenarios,target=CFG.gap,
                        time_limit=CFG.time_limit,max_iter=CFG.ccg_max_iter,
                        verbose=True,reference=reference)
                    result["ccg_reused"] = False
                elif method=="ldr":
                    result = solve_ldr(inst,scenarios,reference)
                else:
                    raise ValueError("Unknown method: "+method)
            except Exception as exc:
                result = dict(
                    runtime_sec=time.perf_counter()-attempt_start,
                    iterations=0,
                    gap=math.nan,
                )
            saved = saved_result_record(
                method, result, table=table, dim=dim,
                scenario_size=I if I else "NA", rep=rep)
            write_json(RUN_DIR/f"case_{label}_{method}.json",saved)
            summarize_records()
    return summarize_records()

def run_ellipsoid_experiment(alpha_0=None, kappa=None):
    alpha_0 = 1.0 if alpha_0 is None else float(alpha_0)
    cases = (("table1_ellipsoid", dim, 0, rep)
             for dim in ELLIPSOID_DIMS for rep in REPS)
    return _run_ellipsoid_scenario_cases(
        cases, alpha_0=alpha_0, kappa=kappa, output_name="ellipsoid")


def run_scenario_experiment(alpha_0=None, kappa=None):
    alpha_0 = 1.0 if alpha_0 is None else float(alpha_0)
    cases = (("table2_scenarios", dim, count, rep)
             for dim in SCENARIO_DIMS for count in SCENARIO_SIZES for rep in REPS)
    return _run_ellipsoid_scenario_cases(
        cases, alpha_0=alpha_0, kappa=kappa, output_name="scenario")


def run_budget_experiment(alpha_0=None, kappa=None):
    """Run the budget-set table with its fixed initialization and averaging."""
    import sys
    alpha_0 = 2.0 if alpha_0 is None else float(alpha_0)
    previous_cfg = CFG
    try:
        return _run_budget_impl(
            sys.modules[__name__], dims=BUDGET_DIMS, gammas=BUDGET_GAMMAS,
            reps=REPS, output_root=OUTPUT_DIR / "budget",
            alpha0=alpha_0, dual_vmf_kappa=kappa)
    finally:
        globals()["CFG"] = previous_cfg


def run_all_runtime_comparisons(alpha_0=None, kappa=None):
    """Run all three Section 5.1.2 comparisons using common overrides."""
    return {
        "ellipsoid": run_ellipsoid_experiment(alpha_0=alpha_0, kappa=kappa),
        "scenario": run_scenario_experiment(alpha_0=alpha_0, kappa=kappa),
        "budget": run_budget_experiment(alpha_0=alpha_0, kappa=kappa),
    }


# Budget uncertainty implementation
"""Budget-set implementation used by the Section 5.1.2 runtime module."""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

BLOCK_ID = "continuous-budget-experiment-v1"
FORMULATION = "continuous-budget-dual-milp-gabrel-lpm-v3"
EXCLUDED_BLOCK_IDS = {BLOCK_ID, "budget-sampled-benchmark-v1"}


def budget_parts(dim, gamma):
    if int(dim) != dim or dim < 1 or not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Require integer dim >= 1 and 0 <= Gamma <= 1")
    budget = float(gamma) * int(dim)
    if abs(budget - round(budget)) < 1e-12:
        budget = float(round(budget))
    whole = int(math.floor(budget))
    return budget, whole, budget - whole


def budget_sample(inst, rng, size=1):
    """Uniform saturated vertices, as in the mixed-integer budget experiment."""
    d = inst["dim"]
    _, whole, fraction = budget_parts(d, inst["gamma"])
    result = np.zeros((int(size), d))
    for row in result:
        indices = rng.permutation(d)
        row[indices[:whole]] = 1.0
        if fraction > 0:
            row[indices[whole]] = fraction
    return result


def budget_support_point(inst, direction):
    """Support of the full budget polytope (also handles negative coefficients)."""
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (inst["dim"],) or not np.all(np.isfinite(direction)):
        raise ValueError("Invalid support direction")
    left = inst["budget"]
    point = np.zeros(inst["dim"])
    for j in np.argsort(-direction, kind="stable"):
        if direction[j] <= 0 or left <= 0:
            break
        point[j] = min(1.0, left)
        left -= point[j]
    return point



def budget_ball_support_point(inst, direction):
    """Support point of the equal-norm ball on the saturated budget hyperplane.

    Saturated budget vertices share center (B/d)1 and squared radius
    floor(B) + (B-floor(B))^2 - B^2/d.  The ball is restricted to
    1^T xi = B, so only the zero-sum component of a direction matters.
    """
    d, budget = int(inst["dim"]), float(inst["budget"])
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (d,) or not np.all(np.isfinite(direction)):
        raise ValueError("Invalid affine-ball support direction")
    checked_budget, whole, fraction = budget_parts(d, inst["gamma"])
    if abs(checked_budget-budget) > 1e-10*max(1.0, abs(budget)):
        raise ValueError("Inconsistent budget and Gamma")
    center = np.full(d, budget/d)
    radius_squared = whole + fraction*fraction - budget*budget/d
    if radius_squared < -1e-10:
        raise ValueError("Invalid equal-norm budget radius")
    tangent_direction = direction-np.mean(direction)
    tangent_norm = np.linalg.norm(tangent_direction)
    if tangent_norm <= 1e-12 or radius_squared <= 0:
        return center
    return center + math.sqrt(max(0.0, radius_squared))*tangent_direction/tangent_norm

def budget_log_partition(log_weights, whole, fraction):
    """Suffix DP for k ones and, if needed, one fractional coordinate."""
    log_weights = np.asarray(log_weights, dtype=float)
    d, fractional = len(log_weights), int(fraction > 0)
    table = np.full((d+1, whole+1, fractional+1), -np.inf)
    table[d, 0, 0] = 0.0
    for j in range(d-1, -1, -1):
        table[j] = table[j+1]
        if whole:
            table[j, 1:, :] = np.logaddexp(
                table[j, 1:, :], log_weights[j]+table[j+1, :-1, :])
        if fractional:
            table[j, :, 1] = np.logaddexp(
                table[j, :, 1], fraction*log_weights[j]+table[j+1, :, 0])
    return table


def budget_weighted_vertex(log_weights, whole, fraction, table, rng):
    """Exact draw from exp(log_weights @ xi) on saturated budget vertices."""
    d, remaining, fractional = len(log_weights), whole, int(fraction > 0)
    point = np.zeros(d)
    for j in range(d):
        if remaining == 0 and fractional == 0:
            break
        if fractional == 0 and remaining == d-j:
            point[j:] = 1.0
            break
        logs = np.array([
            table[j+1, remaining, fractional],
            log_weights[j]+table[j+1, remaining-1, fractional] if remaining else -np.inf,
            fraction*log_weights[j]+table[j+1, remaining, 0] if fractional else -np.inf,
        ])
        probabilities = np.exp(logs-np.max(logs))
        probabilities /= probabilities.sum()
        choice = int(rng.choice(3, p=probabilities))
        if choice == 1:
            point[j], remaining = 1.0, remaining-1
        elif choice == 2:
            point[j], fractional = fraction, 0
    return point


def budget_reference_key(inst, scenarios=None):
    if scenarios is not None:
        raise ValueError("This block solves the full budget set, not a finite scenario pool")
    digest = hashlib.sha256(FORMULATION.encode())
    for key in ("dim", "budget", "emergency", "y_upper", "slope"):
        digest.update(json.dumps([key, inst[key]]).encode())
    for key in ("c", "transport", "base", "sensitivity"):
        value = np.ascontiguousarray(inst[key], dtype=np.float64)
        digest.update(json.dumps([key, list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()

def install_budget_geometry(module):
    """Install explicit uncertainty-type dispatch into the unified module."""
    if getattr(module, "_budget_geometry_installed", False):
        return module
    original_make = module.make_instance
    original_proposal = module.ScenarioProposal
    original_reference_key = module.reference_key
    original_sample_surface = module.sample_surface

    def make_instance(dim, rep, gamma):
        inst = original_make(dim, rep)
        budget, _, _ = budget_parts(dim, gamma)
        for key in ("axes", "rotation", "transform"):
            inst.pop(key, None)
        inst.update(center=np.full(dim, budget/dim), gamma=float(gamma),
                    budget=budget, uncertainty="budget")
        if np.any(inst["sensitivity"] < 0):
            raise ValueError("Saturated-vertex sampling requires nonnegative demand sensitivity")
        return inst

    class BudgetProposal:
        def __init__(self, inst, scenarios, rng, mode, kappa=None):
            if scenarios is not None or mode not in ("uniform", "dual_vmf"):
                raise ValueError("Budget block supports uniform or dual_vmf saturated-vertex proposals")
            self.inst, self.rng = inst, rng
            self.mode = mode
            self.kappa = 0.5*inst["dim"] if kappa is None else float(kappa)
            if not math.isfinite(self.kappa) or not 0 <= self.kappa <= 1e4:
                raise ValueError("dual_vmf_kappa must be finite and in [0, 10000]")
            self.budget, self.whole, self.fraction = budget_parts(inst["dim"], inst["gamma"])
            # All saturated vertices have the same distance from (B/d)*1.
            self.radius = math.sqrt(max(0.0, self.whole+self.fraction**2-self.budget**2/inst["dim"]))
            self.last_phase = "uniform_budget_vertex"
            self.forward_log_probability = None
            self.last_candidate = None

        def uniform(self):
            return budget_sample(self.inst, self.rng, 1)[0]

        def draw(self, current, scenario_gradient, iteration):
            if self.mode == "uniform":
                self.last_phase = "uniform_budget_vertex"
                return self.uniform()
            weights, table = self.distribution(scenario_gradient)
            candidate = budget_weighted_vertex(weights, self.whole, self.fraction, table, self.rng)
            self.forward_log_probability = self.log_probability(candidate, weights, table)
            self.last_candidate = candidate.copy()
            self.last_phase = "dual_vmf_budget_vertex"
            return candidate

        def distribution(self, scenario_gradient):
            gradient = np.asarray(scenario_gradient, dtype=float)
            if gradient.shape != (self.inst["dim"],) or not np.all(np.isfinite(gradient)):
                raise ValueError("Invalid scenario gradient for budget dual_vmf")
            # Only the zero-sum component changes scores on the fixed-budget face.
            direction = gradient-np.mean(gradient)
            length = np.linalg.norm(direction)
            if length <= 1e-12 or self.radius <= 1e-12 or self.kappa == 0:
                weights = np.zeros_like(gradient)
            else:
                weights = self.kappa*direction/(length*self.radius)
                # Subtracting a common score leaves probabilities unchanged at fixed B.
                weights -= np.max(weights)
            table = budget_log_partition(weights, self.whole, self.fraction)
            return weights, table

        def log_probability(self, point, weights, table):
            point = np.asarray(point, dtype=float)
            if point.shape != (self.inst["dim"],):
                raise ValueError("Invalid budget vertex shape")
            allowed = (point == 0) | (point == 1)
            if self.fraction > 0:
                allowed |= point == self.fraction
            if (not np.all(allowed) or np.count_nonzero(point == 1) != self.whole
                    or (self.fraction > 0 and np.count_nonzero(point == self.fraction) != 1)):
                raise ValueError("Proposal probability requires a saturated budget vertex")
            return float(weights@point-table[0, self.whole, int(self.fraction > 0)])

        def hastings_correction(self, current, candidate, grad_current, grad_candidate):
            if self.mode == "uniform":
                return 0.0
            if self.last_candidate is None or not np.array_equal(candidate, self.last_candidate):
                raise ValueError("Hastings correction must follow the corresponding draw")
            weights, table = self.distribution(grad_candidate)
            reverse = self.log_probability(current, weights, table)
            return float(reverse-self.forward_log_probability)

    def proposal_dispatch(inst, scenarios, rng, mode, kappa=None):
        if inst.get("uncertainty") == "budget":
            return BudgetProposal(inst, scenarios, rng, mode, kappa)
        return original_proposal(inst, scenarios, rng, mode, kappa)

    def reference_key_dispatch(inst, scenarios=None):
        if inst.get("uncertainty") == "budget":
            return budget_reference_key(inst, scenarios)
        return original_reference_key(inst, scenarios)

    def sample_surface_dispatch(inst, rng, size=1):
        if inst.get("uncertainty") == "budget":
            return budget_sample(inst, rng, size)
        return original_sample_surface(inst, rng, size)

    module.make_budget_instance = make_instance
    module.ScenarioProposal = proposal_dispatch
    module.reference_key = reference_key_dispatch
    module.sample_surface = sample_surface_dispatch
    module.budget_ball_support_point = budget_ball_support_point
    original_details = module.check_details

    def check_details(result):
        details = original_details(result)
        details.update({key: value for key, value in result.items()
                        if key == "big_m_source" or key.startswith("gabrel_")})
        return details

    module.check_details = check_details
    module._budget_geometry_installed = True
    return module



def budget_ccg_timing(reference, target, time_limit):
    return _benchmark.budget_ccg_timing(reference, target, time_limit)


def _run_budget_impl(module, *, dims=(10, 30, 50, 70, 100),
                          gammas=(0.1, 0.3, 0.5, 0.7, 0.9),
                          reps=(1, 2, 3, 4, 5), output_root=None,
                          alpha0=2.0, dual_vmf_kappa=None):
    """Run/cache CCG references, Ours and LDR with continuous first-stage decisions."""
    m = install_budget_geometry(module)
    m.CFG = replace(m.CFG, ours_max_iter=1000, check_every=10)
    init_mode = "support_demand_lp_ball"
    w = 1e-3
    proposal_mode = "dual_vmf"
    output_rule = "half_linear_average"
    reuse = False
    m.validate_algorithm_settings(
        init_mode, proposal_mode, alpha0, w, output_rule, dual_vmf_kappa)
    dims, gammas, reps = list(dims), list(gammas), list(reps)
    if not dims or not gammas or not reps or any(int(r) != r or r < 1 for r in reps):
        raise ValueError("Use nonempty dimensions/Gammas and positive integer rep indices")
    for dim in dims:
        for gamma in gammas:
            budget_parts(dim, gamma)
    root = Path(output_root) if output_root is not None else m.OUTPUT_DIR / "budget"
    config = dict(formulation=FORMULATION, budget_version=FORMULATION, version=m.VERSION,
                  settings=asdict(m.CFG),
                  algorithm=dict(init_mode=init_mode, alpha0=alpha0, w=w,
                                 proposal_mode=proposal_mode, output_rule=output_rule,
                                 dual_vmf_kappa=dual_vmf_kappa),
                  dims=dims, gammas=gammas, reps=reps,
                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Code-only changes to this adapter must not hide compatible completed cases.
    identity_config = json.loads(json.dumps(
        {key: value for key, value in config.items() if key != "adapter_sha256"},
        sort_keys=True))
    identifier = hashlib.sha256(json.dumps(identity_config, sort_keys=True).encode()).hexdigest()[:12]
    directory = root / identifier
    directory.mkdir(parents=True, exist_ok=True)
    cache = root / "reference_cache"
    cache.mkdir(exist_ok=True)
    records, references = [], []

    def save_tables():
        refs = m.pd.DataFrame(references)
        raw = m.pd.DataFrame(records)
        raw.to_csv(directory/"raw.csv", index=False)
        for method, algorithm in m.ALGORITHM_NAMES.items():
            if len(raw):
                raw.loc[raw["algorithm"] == algorithm].to_csv(
                    directory/f"{method}_raw.csv", index=False)
        refs.to_csv(directory/"reference_times.csv", index=False)
        if len(raw):
            summary = raw.groupby(["dim", "gamma", "algorithm"], as_index=False).agg(
                records=("rep", "size"), algorithm_sec_mean=("algorithm_sec", "mean"),
                algorithm_sec_std=("algorithm_sec", "std"), gap_mean=("gap", "mean"))
            summary.to_csv(directory/"summary.csv", index=False)
            for method, algorithm in m.ALGORITHM_NAMES.items():
                summary.loc[summary["algorithm"] == algorithm].to_csv(
                    directory/f"{method}_summary.csv", index=False)
        else:
            summary = m.pd.DataFrame()
        return raw, summary, refs

    for dim in dims:
        for gamma in gammas:
            for rep in reps:
                inst = m.make_budget_instance(int(dim), int(rep), float(gamma))
                label = f"d{dim}_Gamma{gamma:g}_rep{rep}"
                metadata = dict(dim=dim, gamma=gamma, budget=inst["budget"], rep=rep,
                                instance_seed=inst["seed"], uncertainty="budget")
                ref_path = cache / ("reference_" + budget_reference_key(inst) + ".json")
                ref = m.load_reference_file(ref_path) if reuse else None
                reused = bool(ref and m.usable_reference(ref))
                compute_sec = 0.0
                if not reused:
                    start = time.perf_counter()
                    try:
                        ref = m.solve_ccg(inst, None, target=m.CFG.reference_gap,
                                          time_limit=m.CFG.reference_time_limit,
                                          max_iter=m.CFG.reference_max_iter, verbose=True)
                    except Exception as exc:
                        ref = dict(status="error", error=repr(exc), traceback=m.traceback.format_exc(),
                                   runtime_sec=time.perf_counter()-start, lower=None, upper=None,
                                   objective=None, gap=math.nan)
                    compute_sec = time.perf_counter() - start
                    ref.update(reference_mode="budget_ccg", ccg_formulation=FORMULATION,
                               reference_formulation=FORMULATION, big_m_source="gabrel_lp_m",
                               **metadata)
                reference_certified = (
                    m.bound_gap(ref.get("lower"), ref.get("upper")) <= m.CFG.reference_gap
                )
                reference_usable = m.usable_reference(ref)
                # Keep the reference runtime from its original CCG solve, even when reused.
                reference_row = m.algorithm_record(
                    "C&CG reference", ref.get("iterations", 1),
                    ref.get("runtime_sec"), ref.get("solver_gap", ref.get("gap")),
                    dim=dim, gamma=gamma, rep=rep)
                references.append(reference_row)
                m.write_json(directory/f"reference_{label}.json", reference_row)
                # Reference keeps its full 0.01% solve time; CCG uses the 0.1% UB hit.
                timing = (budget_ccg_timing(ref, m.CFG.gap, m.CFG.time_limit) if reference_certified else
                          dict(status="reference_unavailable", runtime_sec=math.nan, gap=math.nan))
                timing_record = m.saved_result_record(
                    "ccg", timing, dim=dim, gamma=gamma, rep=rep)
                records.append(timing_record)
                m.write_json(directory/f"ccg_{label}.json", timing_record)
                save_tables()
                for method in ("ours", "ldr"):
                    path = directory / f"{method}_{label}.json"
                    result = m.load_reference_file(path) if reuse else None
                    if not result or result.get("status") in ("error", "reference_unavailable"):
                        if not reference_usable:
                            result = dict(status="reference_unavailable", runtime_sec=math.nan, gap=math.nan)
                        else:
                            try:
                                result = (m.solve_ours(inst, None, ref, inst["seed"]+29,
                                                      directory/f"trace_{label}.json",
                                                      init_mode=init_mode, w=w,
                                                      proposal_mode=proposal_mode,
                                                      alpha0=alpha0,
                                                      output_rule=output_rule,
                                                      dual_vmf_kappa=dual_vmf_kappa)
                                          if method == "ours" else m.solve_ldr(inst, None, ref))
                            except Exception as exc:
                                result = dict(status="error", error=repr(exc), traceback=m.traceback.format_exc(),
                                              runtime_sec=math.nan, gap=math.nan)
                        result.update(metadata, method=method, big_m_source="gabrel_lp_m",
                                      reference_usable=reference_usable,
                                      reference_certified=reference_certified)
                        saved = m.saved_result_record(
                            method, result, dim=dim, gamma=gamma, rep=rep)
                        m.write_json(path, saved)
                    else:
                        saved = result
                    records.append(saved)
                    save_tables()
    raw, summary, reference_times = save_tables()
    return dict(directory=directory, raw=raw, summary=summary, reference_times=reference_times)
