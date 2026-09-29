from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from IPython.display import display

try:
    from . import benchmark as _benchmark
except ImportError:
    import benchmark as _benchmark


class _RuntimeAPI:
    def __getattr__(self, name):
        return globals()[name]


_BENCHMARK_RUNTIME = _RuntimeAPI()

VERSION = "section52-reproducibility-v1"
PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "results" / VERSION
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Section 5.2 settings.
W = 0.001
PROPOSAL_MODE = "uniform"         # "uniform" or "dual_vmf"
DUAL_VMF_KAPPA = None             # None selects dim/2
TARGET_GAP = 1e-3                 # Reported benchmark threshold: 0.1%
REFERENCE_GAP = 1e-4             
MAX_CCG_ITERATIONS = 1000
TIME_LIMIT = 1800.0
EXACT_CHECK_TIME_LIMIT = 1800.0
MASTER_MIP_GAP = 1e-4
USE_WARM_START = True
WARMUP_ITERATIONS = 100
EPSILON_FALSE = 0.1
EPSILON_OPT = 200.0
EPSILON_CERT = 100.0
CERTIFICATION_RHO = 0.9
MAX_CERTIFICATION_ATTEMPTS = 5_000
MAX_TOTAL_ORACLE_CALLS = 100_000
REPS = tuple(range(1, 6))
SOLVER_THREADS = 0                 # Gurobi automatic thread selection
# Ellipsoidal KKT complementarity: "tight_big_m" or "sos1".
ELLIPSOID_COMPLEMENTARITY_MODE = "tight_big_m"
REUSE = True


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))
    temporary.replace(path)


def random_rotation(dim, rng):
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q @ np.diag(signs)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def make_instance(dim, rep, uncertainty, gamma=None):
    """Section 5.2 data, with the Section 5.1 random ranges."""
    seed = 1_000_000 + 10_000 * int(dim) + int(rep)
    rng = np.random.default_rng(seed)
    instance = {
        "dim": int(dim),
        "rep": int(rep),
        "seed": int(seed),
        "uncertainty": str(uncertainty),
        "fixed_cost": rng.integers(300, 401, size=dim).astype(float),
        "capacity_cost": rng.integers(10, 31, size=dim).astype(float),
        "transport": rng.integers(20, 41, size=(dim, dim)).astype(float),
        "emergency": rng.integers(100, 201, size=dim).astype(float),
        "base": rng.integers(200, 401, size=dim).astype(float),
        "sensitivity": None,
        "K": rng.integers(800, 1201, size=dim).astype(float),
    }
    instance["sensitivity"] = rng.uniform(0.1, 0.5, size=dim) * instance["base"]
    if uncertainty in {"ellipsoid", "scenario"}:
        axes = rng.uniform(0.10, 0.50, size=dim)
        rotation = random_rotation(dim, rng)
        instance.update(center=np.full(dim, 0.5), axes=axes,
                        rotation=rotation, A=rotation @ np.diag(axes))
    elif uncertainty == "budget":
        if gamma is None:
            raise ValueError("gamma is required for budget uncertainty")
        budget = float(gamma) * dim
        if abs(budget - round(budget)) > 1e-10:
            raise ValueError("This exact sampler/separator requires integer Gamma*dim")
        instance.update(gamma=float(gamma), budget=int(round(budget)))
    else:
        raise ValueError("unknown uncertainty set")
    return instance


def sample_ellipsoid_boundary(instance, rng, count=1):
    dim = instance["dim"]
    directions = rng.normal(size=(int(count), dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return instance["center"] + directions @ instance["A"].T


def make_scenarios(instance, count, seed_offset=37):
    rng = np.random.default_rng(instance["seed"] + int(count) + int(seed_offset))
    return sample_ellipsoid_boundary(instance, rng, count)


def sample_budget_extreme(instance, rng):
    xi = np.zeros(instance["dim"], dtype=float)
    if instance["budget"]:
        xi[rng.choice(instance["dim"], size=instance["budget"], replace=False)] = 1.0
    return xi


class ExactRecourseLP:
    """Exact recourse LP with minimum-L1 optimal-face dual selection."""
    def __init__(self, instance, face_tolerance=1e-8):
        self.instance = instance
        self.face_tolerance = float(face_tolerance)
        d = instance["dim"]
        model = gp.Model("exact_recourse_dual")
        model.Params.OutputFlag = 0
        model.Params.Threads = SOLVER_THREADS
        self.u = model.addVars(d, lb=0.0, name="u")
        self.v = model.addVars(d, lb=0.0, name="v")
        for j in range(d):
            self.v[j].UB = float(instance["emergency"][j])
        for i in range(d):
            for j in range(d):
                model.addConstr(self.v[j] - self.u[i] <= float(instance["transport"][i, j]))
        model.update()
        self.model = model
        self.secondary_failures = 0

    def solve(self, z, xi, select_dual=True):
        z = np.asarray(z, dtype=float)
        xi = np.asarray(xi, dtype=float)
        demand = self.instance["base"] + self.instance["sensitivity"] * xi
        primary = (gp.quicksum(float(demand[j]) * self.v[j] for j in range(self.instance["dim"]))
                   - gp.quicksum(float(z[i]) * self.u[i] for i in range(self.instance["dim"])))
        self.model.setObjective(primary, GRB.MAXIMIZE)
        self.model.reset(1)
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"primary recourse LP status={self.model.Status}")
        value = float(self.model.ObjVal)
        if not select_dual:
            return {"value": value, "xi": xi}

        primary_u = np.array([self.u[i].X for i in range(self.instance["dim"])])
        primary_v = np.array([self.v[j].X for j in range(self.instance["dim"])])
        face = self.model.addConstr(primary >= value - self.face_tolerance * max(1.0, abs(value)),
                                    name="primary_optimal_face")
        self.model.setObjective(
            gp.quicksum(self.u[i] for i in range(self.instance["dim"])),
            GRB.MINIMIZE,
        )
        selection_method = "optimal_face_min_l1"
        self.model.reset(1)
        self.model.optimize()
        if self.model.Status == GRB.OPTIMAL:
            u = np.array([self.u[i].X for i in range(self.instance["dim"])])
            v = np.array([self.v[j].X for j in range(self.instance["dim"])])
            secondary = True
        else:
            self.secondary_failures += 1
            u, v, secondary = primary_u, primary_v, False
            selection_method = "primary_fallback"
        self.model.remove(face)
        self.model.update()
        # Valid cut: eta + u^T z >= demand^T v.  The RHS is computed from
        # the selected feasible dual solution, never from the primary ObjVal.
        intercept = float(demand @ v)
        selected_value = float(intercept - z @ u)
        return {"value": value, "selected_value": selected_value,
                "intercept": intercept, "u": u, "v": v, "xi": xi,
                "secondary_optimal": secondary, "gradient_l1": float(np.sum(u)),
                "dual_selection": selection_method}

    def close(self):
        self.model.dispose()


def worst_total_demand(instance, scenarios=None):
    """Exact max of total demand over the active uncertainty set."""
    base = np.asarray(instance["base"], dtype=float)
    sensitivity = np.asarray(instance["sensitivity"], dtype=float)
    uncertainty = instance["uncertainty"]
    if uncertainty == "scenario":
        if scenarios is None or len(scenarios) == 0:
            raise ValueError("Finite-scenario capacity bound requires nonempty scenarios")
        scenarios = np.asarray(scenarios, dtype=float)
        return float(np.max(np.sum(base[None, :] + sensitivity[None, :] * scenarios, axis=1)))
    if uncertainty == "ellipsoid":
        # xi = center + A*s, ||s||_2 <= 1.
        return float(np.sum(base) + sensitivity @ instance["center"]
                     + np.linalg.norm(instance["A"].T @ sensitivity))
    if uncertainty == "budget":
        # max sensitivity^T xi over 0 <= xi <= 1, sum xi <= budget.
        budget = min(float(instance["budget"]), float(instance["dim"]))
        weights = np.sort(np.maximum(sensitivity, 0.0))[::-1]
        whole = int(math.floor(budget))
        fraction = budget - whole
        support = float(np.sum(weights[:whole]))
        if fraction > 1e-12 and whole < len(weights):
            support += fraction * float(weights[whole])
        return float(np.sum(base) + support)
    raise ValueError("unknown uncertainty set")


def first_stage_value(instance, y, z):
    return float(instance["fixed_cost"] @ y + instance["capacity_cost"] @ z)


def add_recourse_block(master, instance, y, z, eta, xi, label):
    """Primal recourse block shared by warm start, the tree, and C&CG."""
    d = instance["dim"]
    demand = instance["base"] + instance["sensitivity"] * np.asarray(xi)
    flow = master.addVars(d, d, lb=0.0, name=f"flow_{label}")
    emergency = master.addVars(d, lb=0.0, name=f"emergency_{label}")
    for i in range(d):
        master.addConstr(gp.quicksum(flow[i, j] for j in range(d)) <= z[i])
    for j in range(d):
        master.addConstr(gp.quicksum(flow[i, j] for i in range(d)) + emergency[j]
                         >= float(demand[j]))
    master.addConstr(
        eta >= gp.quicksum(float(instance["transport"][i, j]) * flow[i, j]
                           for i in range(d) for j in range(d))
        + gp.quicksum(float(instance["emergency"][j]) * emergency[j] for j in range(d))
    )


def make_master(instance, name, scenarios=None, relax_integrality=False):
    d = instance["dim"]
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.Threads = SOLVER_THREADS
    model.Params.MIPGap = MASTER_MIP_GAP
    y_type = GRB.CONTINUOUS if relax_integrality else GRB.BINARY
    y = model.addVars(d, lb=0.0, ub=1.0, vtype=y_type, name="y")
    z = model.addVars(d, lb=0.0, name="z")
    eta = model.addVar(lb=0.0, name="eta")
    for i in range(d):
        model.addConstr(z[i] <= float(instance["K"][i]) * y[i])
    model.setObjective(
        gp.quicksum(float(instance["fixed_cost"][i]) * y[i]
                    + float(instance["capacity_cost"][i]) * z[i] for i in range(d)) + eta,
        GRB.MINIMIZE,
    )
    return model, y, z, eta


def initial_scenario(instance, scenarios=None):
    if instance["uncertainty"] == "scenario":
        return np.asarray(scenarios[0], dtype=float)
    if instance["uncertainty"] == "ellipsoid":
        return np.asarray(instance["center"], dtype=float)
    return np.zeros(instance["dim"], dtype=float)


# Benchmark entry points. Their implementations live in benchmark.py.
def _configure_benchmark():
    _benchmark.configure(_BENCHMARK_RUNTIME)


def make_separator(instance, scenarios=None):
    _configure_benchmark()
    return _benchmark.make_separator(instance, scenarios)


def exact_robust_objective(instance, y, z, scenarios=None, time_limit=TIME_LIMIT):
    _configure_benchmark()
    return _benchmark.exact_robust_objective(
        instance, y, z, scenarios, time_limit=float(time_limit))


def solve_ccg_reference(instance, scenarios=None, target_gap=REFERENCE_GAP,
                        max_iterations=MAX_CCG_ITERATIONS, time_limit=TIME_LIMIT):
    _configure_benchmark()
    return _benchmark.solve_ccg_reference(
        instance, scenarios, target_gap=float(target_gap),
        max_iterations=int(max_iterations), time_limit=float(time_limit))


def solve_ldr(instance, reference_value, scenarios=None):
    _configure_benchmark()
    return _benchmark.solve_ldr(
        instance, reference_value, scenarios, time_limit=float(TIME_LIMIT))


def benders_cut_signature(intercept, u):
    return round(float(intercept), 8), tuple(np.round(np.asarray(u, dtype=float), 8))


def demand_support_scenario(instance, scenarios=None):
    """Maximize total demand over the active uncertainty set."""
    weights = np.asarray(instance["sensitivity"], dtype=float)
    uncertainty = instance["uncertainty"]
    if uncertainty == "scenario":
        if scenarios is None or len(scenarios) == 0:
            raise ValueError("support_lp requires a nonempty scenario set")
        scenarios = np.asarray(scenarios, dtype=float)
        return scenarios[int(np.argmax(scenarios @ weights))].copy()
    if uncertainty == "ellipsoid":
        direction = np.asarray(instance["A"], dtype=float).T @ weights
        norm = np.linalg.norm(direction)
        if norm <= 1e-14:
            return np.asarray(instance["center"], dtype=float).copy()
        return (np.asarray(instance["center"], dtype=float)
                + np.asarray(instance["A"], dtype=float) @ (direction / norm))
    if uncertainty == "budget":
        budget = min(float(instance["budget"]), float(instance["dim"]))
        point = np.zeros(instance["dim"], dtype=float)
        order = np.argsort(weights)[::-1]
        whole = int(math.floor(budget))
        if whole:
            point[order[:whole]] = 1.0
        if budget > whole and whole < instance["dim"]:
            point[order[whole]] = budget - whole
        return point
    raise ValueError("unknown uncertainty set")


def continuous_support_lp_start(instance, xi, time_limit):
    """Solve the relaxed first-stage LP for one demand-support scenario."""
    model, y, z, eta = make_master(
        instance, "continuous_support_demand_lp", relax_integrality=True)
    add_recourse_block(model, instance, y, z, eta, xi, "support")
    model.Params.Method = 1
    model.Params.TimeLimit = max(0.01, float(time_limit))
    try:
        model.optimize()
        if model.SolCount == 0:
            raise RuntimeError(f"support-demand LP status={model.Status}")
        return np.array([z[i].X for i in range(instance["dim"])])
    finally:
        model.dispose()


def solve_continuous_relaxation_warmup(
        instance, recourse, proposal_kernel, rng, scenarios=None,
        iterations=None):
    """Run the support-LP initialized continuous warm-up."""
    if iterations is None:
        iterations = WARMUP_ITERATIONS
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("warm-up iterations must be a positive integer")
    start = time.perf_counter()
    time_limit = 300.0
    d = instance["dim"]
    capacities = np.asarray(instance["K"], dtype=float)
    xi = demand_support_scenario(instance, scenarios)
    z_value = continuous_support_lp_start(
        instance, xi, max(0.01, time_limit - (time.perf_counter() - start)))

    # For fixed z, positivity of the opening costs implies that the relaxed
    # optimum uses y_i=z_i/K_i.  The continuous relaxation can therefore be
    # written over z alone with the effective linear cost below.
    effective_cost = (np.asarray(instance["capacity_cost"], dtype=float)
                      + np.asarray(instance["fixed_cost"], dtype=float)
                      / np.maximum(capacities, 1e-12))
    cuts, signatures, trace = [], set(), []
    weighted_sum = np.zeros(d, dtype=float)
    weight_sum = 0.0

    def add_cut(selected):
        signature = benders_cut_signature(selected["intercept"], selected["u"])
        if signature in signatures:
            return False
        signatures.add(signature)
        u_value = np.asarray(selected["u"], dtype=float).copy()
        intercept = float(selected["intercept"])
        cuts.append({"u": u_value, "intercept": intercept,
                     "xi": np.asarray(selected["xi"], dtype=float).copy(),
                     "dual_selection": selected.get("dual_selection")})
        return True

    initial = recourse.solve(z_value, xi, True)
    add_cut(initial)
    status = "completed"
    for iteration in range(1, iterations + 1):
        if time.perf_counter() - start >= time_limit:
            status = "time_limit"
            break
        current = recourse.solve(z_value, xi, True)
        accepted = 0
        proposal, forward_log_probability = proposal_kernel.draw(current)
        candidate = recourse.solve(z_value, proposal, True)
        reverse_log_probability = proposal_kernel.log_probability(current["xi"], candidate)
        log_ratio = ((candidate["value"] - current["value"]) / W
                     + reverse_log_probability - forward_log_probability)
        if math.log(max(rng.random(), np.finfo(float).tiny)) <= min(0.0, log_ratio):
            current = candidate
            accepted = 1
        xi = np.asarray(current["xi"], dtype=float).copy()
        cut_added = add_cut(current)
        alpha = 1.0 / math.sqrt(iteration)
        previous_z = z_value.copy()
        z_value = np.clip(
            z_value - alpha * (effective_cost - np.asarray(current["u"], dtype=float)),
            0.0, capacities)
        weighted_sum += alpha * z_value
        weight_sum += alpha
        averaged_z = weighted_sum / weight_sum
        trace.append({"iteration": iteration, "alpha": alpha,
                      "movement": float(np.linalg.norm(z_value - previous_z)),
                      "selected_recourse": float(current["value"]),
                      "accepted": accepted, "cut_added": cut_added,
                      "cuts": len(cuts),
                      "runtime_sec": time.perf_counter() - start})

    averaged_z = weighted_sum / weight_sum if weight_sum > 0 else z_value.copy()
    start_eta = max([0.0] + [float(cut["intercept"] - cut["u"] @ averaged_z)
                             for cut in cuts])
    mip_start = {"y": (averaged_z > 1e-8).astype(float),
                 "z": averaged_z, "eta": start_eta}
    return {"enabled": True, "status": status,
            "runtime_sec": time.perf_counter() - start,
            "iterations": len(trace), "alpha0": 1.0,
            "method": "soft_projected_subgradient",
            "initialization": "support_lp",
            "cuts": cuts, "start": mip_start,
            "last_xi": xi, "trace": trace}




def draw_uniform_scenario(instance, rng, scenarios=None):
    if instance["uncertainty"] == "scenario":
        return np.asarray(scenarios[int(rng.integers(0, len(scenarios)))], dtype=float)
    if instance["uncertainty"] == "ellipsoid":
        return sample_ellipsoid_boundary(instance, rng, 1)[0]
    return sample_budget_extreme(instance, rng)


def sample_vmf_direction(mu, kappa, rng):
    """Exact Wood rejection sampler on the unit sphere."""
    mu = np.asarray(mu, dtype=float)
    dimension = len(mu)
    norm = np.linalg.norm(mu)
    if norm <= 1e-14:
        mu = np.zeros(dimension)
        mu[0] = 1.0
    else:
        mu = mu / norm
    if kappa <= 1e-14:
        sample = rng.normal(size=dimension)
        return sample / np.linalg.norm(sample)
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa**2 + (dimension - 1.0)**2)) / (dimension - 1.0)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (dimension - 1.0) * math.log(1.0 - x0**2)
    while True:
        beta = rng.beta(0.5 * (dimension - 1.0), 0.5 * (dimension - 1.0))
        axial = (1.0 - (1.0 + b) * beta) / (1.0 - (1.0 - b) * beta)
        if (kappa * axial + (dimension - 1.0) * math.log(1.0 - x0 * axial) - c
                >= math.log(max(rng.random(), np.finfo(float).tiny))):
            break
    tangent = rng.normal(size=dimension - 1)
    tangent /= np.linalg.norm(tangent)
    canonical = np.concatenate(([axial], math.sqrt(max(0.0, 1.0 - axial**2)) * tangent))
    first = np.zeros(dimension)
    first[0] = 1.0
    difference = first - mu
    if np.linalg.norm(difference) <= 1e-14:
        return canonical
    difference /= np.linalg.norm(difference)
    return canonical - 2.0 * difference * (difference @ canonical)


def budget_log_partition(weights, cardinality):
    """Suffix DP for log sum exp(weights^T xi) over binary xi with fixed cardinality."""
    weights = np.asarray(weights, dtype=float)
    dimension = len(weights)
    table = np.full((dimension + 1, cardinality + 1), -math.inf)
    table[dimension, 0] = 0.0
    for index in range(dimension - 1, -1, -1):
        table[index, 0] = 0.0
        for remaining in range(1, min(cardinality, dimension - index) + 1):
            table[index, remaining] = np.logaddexp(
                table[index + 1, remaining],
                weights[index] + table[index + 1, remaining - 1],
            )
    return table


def sample_weighted_budget_vertex(weights, cardinality, table, rng):
    point = np.zeros(len(weights), dtype=float)
    remaining = int(cardinality)
    for index in range(len(weights)):
        positions = len(weights) - index
        if remaining == 0:
            break
        if remaining == positions:
            point[index:] = 1.0
            remaining = 0
            break
        log_select = weights[index] + table[index + 1, remaining - 1]
        probability = math.exp(min(0.0, log_select - table[index, remaining]))
        if rng.random() <= probability:
            point[index] = 1.0
            remaining -= 1
    if remaining != 0:
        raise RuntimeError("Budget proposal DP failed to sample the requested cardinality")
    return point


class ScenarioProposal:
    """Uniform or dual-guided vMF proposal with exact Hastings probabilities."""
    def __init__(self, instance, scenarios, rng):
        if PROPOSAL_MODE not in ("uniform", "dual_vmf"):
            raise ValueError("PROPOSAL_MODE must be 'uniform' or 'dual_vmf'")
        self.instance = instance
        self.scenarios = None if scenarios is None else np.asarray(scenarios, dtype=float)
        self.rng = rng
        self.mode = PROPOSAL_MODE
        self.kappa = (0.5 * instance["dim"] if DUAL_VMF_KAPPA is None
                      else float(DUAL_VMF_KAPPA))
        if not math.isfinite(self.kappa) or self.kappa < 0.0:
            raise ValueError("DUAL_VMF_KAPPA must be finite and nonnegative")
        self.scenario_units = None
        self.scenario_lookup = None
        if instance["uncertainty"] == "scenario":
            if self.scenarios is None or len(self.scenarios) == 0:
                raise ValueError("Finite-scenario proposal requires nonempty scenarios")
            centered = self.scenarios - instance["center"]
            sphere = np.linalg.solve(instance["A"], centered.T).T
            lengths = np.linalg.norm(sphere, axis=1)
            self.scenario_units = sphere / np.maximum(lengths[:, None], 1e-14)
            self.scenario_lookup = {
                np.ascontiguousarray(row, dtype=np.float64).tobytes(): index
                for index, row in enumerate(self.scenarios)
            }
        self.budget_cardinality = int(instance.get("budget", 0))
        self.budget_radius = math.sqrt(max(
            0.0,
            self.budget_cardinality
            - self.budget_cardinality**2 / instance["dim"],
        ))

    def uniform(self):
        return draw_uniform_scenario(self.instance, self.rng, self.scenarios)

    def scenario_gradient(self, selected):
        return np.asarray(self.instance["sensitivity"], dtype=float) * np.asarray(selected["v"], dtype=float)

    @staticmethod
    def normalized(direction):
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm <= 1e-14:
            fallback = np.zeros(len(direction))
            fallback[0] = 1.0
            return fallback
        return direction / norm

    def ellipsoid_parameters(self, selected):
        gradient = self.scenario_gradient(selected)
        return self.normalized(self.instance["A"].T @ gradient)

    def finite_log_probabilities(self, selected):
        mu = self.ellipsoid_parameters(selected)
        logits = self.kappa * (self.scenario_units @ mu)
        maximum = float(np.max(logits))
        log_normalizer = maximum + math.log(float(np.sum(np.exp(logits - maximum))))
        return logits - log_normalizer

    def budget_parameters(self, selected):
        gradient = self.scenario_gradient(selected)
        centered = gradient - np.mean(gradient)
        norm = np.linalg.norm(centered)
        if norm <= 1e-14 or self.budget_radius <= 1e-14 or self.kappa == 0.0:
            weights = np.zeros_like(centered)
        else:
            weights = self.kappa * centered / (norm * self.budget_radius)
            weights -= np.max(weights)
        return weights, budget_log_partition(weights, self.budget_cardinality)

    def point_index(self, point):
        key = np.ascontiguousarray(point, dtype=np.float64).tobytes()
        if key not in self.scenario_lookup:
            raise ValueError("Finite-scenario proposal received a point outside the scenario set")
        return self.scenario_lookup[key]

    def draw(self, selected):
        if self.mode == "uniform":
            return self.uniform(), 0.0
        uncertainty = self.instance["uncertainty"]
        if uncertainty == "ellipsoid":
            mu = self.ellipsoid_parameters(selected)
            sphere = sample_vmf_direction(mu, self.kappa, self.rng)
            point = self.instance["center"] + self.instance["A"] @ sphere
            return point, float(self.kappa * (mu @ sphere))
        if uncertainty == "scenario":
            log_probabilities = self.finite_log_probabilities(selected)
            probabilities = np.exp(log_probabilities)
            index = int(self.rng.choice(len(probabilities), p=probabilities))
            return self.scenarios[index].copy(), float(log_probabilities[index])
        if uncertainty == "budget":
            weights, table = self.budget_parameters(selected)
            point = sample_weighted_budget_vertex(
                weights, self.budget_cardinality, table, self.rng
            )
            log_probability = float(weights @ point - table[0, self.budget_cardinality])
            return point, log_probability
        raise ValueError("unknown uncertainty set")

    def log_probability(self, point, selected):
        if self.mode == "uniform":
            return 0.0
        uncertainty = self.instance["uncertainty"]
        point = np.asarray(point, dtype=float)
        if uncertainty == "ellipsoid":
            sphere = np.linalg.solve(self.instance["A"], point - self.instance["center"])
            sphere = self.normalized(sphere)
            return float(self.kappa * (self.ellipsoid_parameters(selected) @ sphere))
        if uncertainty == "scenario":
            return float(self.finite_log_probabilities(selected)[self.point_index(point)])
        if uncertainty == "budget":
            weights, table = self.budget_parameters(selected)
            return float(weights @ point - table[0, self.budget_cardinality])
        raise ValueError("unknown uncertainty set")


@dataclass(frozen=True)
class SingleTreeConfig:
    use_warm_start: bool = USE_WARM_START
    warmup_iterations: int = WARMUP_ITERATIONS
    temperature: float = W
    epsilon_false: float = EPSILON_FALSE
    epsilon_opt: float = EPSILON_OPT
    epsilon_cert: float = EPSILON_CERT
    certification_rho: float = CERTIFICATION_RHO
    max_certification_attempts: int = MAX_CERTIFICATION_ATTEMPTS
    max_total_oracle_calls: int = MAX_TOTAL_ORACLE_CALLS
    time_limit: float = TIME_LIMIT


def certification_constants(config):
    if config.use_warm_start and (
            not isinstance(config.warmup_iterations, int)
            or config.warmup_iterations <= 0):
        raise ValueError("warmup_iterations must be a positive integer")
    if not (0.0 < config.epsilon_false < 1.0):
        raise ValueError("epsilon_false must lie in (0,1)")
    if config.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not (0.0 < config.epsilon_cert < config.epsilon_opt):
        raise ValueError("epsilon_cert must lie strictly between zero and epsilon_opt")
    if not (0.0 < config.certification_rho < 1.0):
        raise ValueError("certification_rho must lie in (0,1)")
    return {
        "rho": float(config.certification_rho),
        "log_rho": math.log(config.certification_rho),
        "master_gap_target": config.epsilon_opt - config.epsilon_cert,
    }


def initial_log_ratio(label, kappa, epsilon_false):
    return (math.log(label) + math.log(label + 1.0)
            + math.log(kappa) + math.log(kappa + 1.0)
            - math.log(epsilon_false))


def continuation_calls_required(log_ratio, log_rho):
    if log_ratio <= 0.0:
        return 0
    return int(math.ceil(log_ratio / (-log_rho)))


def solve_single_tree_streamlined(instance, scenarios, config, seed=0):
    """Single-tree Algorithm 3 with callback-level sequential certification."""
    constants = certification_constants(config)
    started = time.perf_counter()
    deadline = started + config.time_limit
    rng = np.random.default_rng(seed)
    random.seed(seed)

    model, y, capacity, eta = make_master(
        instance, "algorithm3_single_tree", scenarios)
    model.Params.LazyConstraints = 1
    # A single callback thread has exclusive access to each persistent chain.
    model.Params.Threads = 1
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = float(constants["master_gap_target"])
    model.Params.FeasibilityTol = 1e-9
    recourse = ExactRecourseLP(instance, face_tolerance=0.0)
    proposal_kernel = ScenarioProposal(instance, scenarios, rng)

    states, cuts, cuts_by_signature = {}, [], {}
    trace, certified_history = [], []
    next_label = 0
    best_certified = None
    certified_upper = math.inf
    attempt_counter = oracle_calls = mh_transitions = 0
    continuation_clean_calls = rejected_candidates = 0
    revisited_integer_attempts = 0
    same_integer_changed_continuous_attempts = 0
    termination_reason = certificate_failure = callback_error = None

    def elapsed():
        return time.perf_counter() - started

    def cut_expression(record):
        return eta + gp.quicksum(
            float(record["u"][i]) * capacity[i]
            for i in range(instance["dim"]))

    def register_cut(selected, source):
        signature = benders_cut_signature(selected["intercept"], selected["u"])
        if signature in cuts_by_signature:
            return cuts_by_signature[signature], False
        record = {
            "u": np.asarray(selected["u"], dtype=float).copy(),
            "intercept": float(selected["intercept"]),
            "xi": np.asarray(selected["xi"], dtype=float).copy(),
            "source": source,
        }
        cuts.append(record)
        cuts_by_signature[signature] = record
        return record, True

    def add_initial_constraint(selected, source):
        record, is_new = register_cut(selected, source)
        if is_new:
            model.addConstr(
                cut_expression(record) >= float(record["intercept"]),
                name=f"{source}_cut_{len(cuts)}")
        return record

    def set_start(start):
        if start is None:
            return
        z_value = np.asarray(start["capacity"], dtype=float)
        theta_value = max(
            [float(start["theta"])]
            + [float(record["intercept"] - record["u"] @ z_value)
               for record in cuts])
        for i in range(instance["dim"]):
            y[i].Start = float(start["y"][i])
            capacity[i].Start = float(z_value[i])
        eta.Start = theta_value

    def binary_key(candidate):
        return tuple(int(round(value)) for value in candidate["y"])

    global W
    W = float(config.temperature)
    if config.use_warm_start:
        warmup = solve_continuous_relaxation_warmup(
            instance, recourse, proposal_kernel, rng, scenarios=scenarios,
            iterations=config.warmup_iterations)
        chain_seed = np.asarray(warmup["last_xi"], dtype=float).copy()
        for selected in warmup["cuts"]:
            add_initial_constraint(selected, "warmup")
    else:
        warmup = {"enabled": False, "status": "disabled", "runtime_sec": 0.0,
                  "iterations": 0, "cuts": [], "start": None, "trace": []}
        chain_seed = np.asarray(initial_scenario(instance, scenarios), dtype=float)

    initial = recourse.solve(
        np.zeros(instance["dim"]), initial_scenario(instance, scenarios), True)
    add_initial_constraint(initial, "initial")
    if warmup["start"] is not None:
        set_start({
            "y": np.asarray(warmup["start"]["y"], dtype=float),
            "capacity": np.asarray(warmup["start"]["z"], dtype=float),
            "theta": float(warmup["start"]["eta"]),
        })
    model.update()

    def ensure_state(candidate):
        nonlocal next_label
        key = binary_key(candidate)
        if key not in states:
            next_label += 1
            states[key] = {
                "xi": chain_seed.copy(),
                "last_capacity": np.asarray(candidate["capacity"], dtype=float).copy(),
                "kappa": 0,
                "label": next_label,
                "candidate_visits": 0,
                "continuous_signatures": set(),
            }
        return key, states[key]

    def limit_reason():
        if elapsed() >= config.time_limit:
            return "time_limit"
        if oracle_calls >= config.max_total_oracle_calls:
            return "oracle_call_cap"
        return None

    def oracle_call(candidate, attempt, phase):
        nonlocal oracle_calls, mh_transitions
        reason = limit_reason()
        if reason is not None:
            return None, None, {"stop_reason": reason}
        key, state = ensure_state(candidate)
        local_kappa = int(state["kappa"])
        movement = float(np.linalg.norm(
            np.asarray(candidate["capacity"], dtype=float)
            - state["last_capacity"]))
        current = recourse.solve(candidate["capacity"], state["xi"], True)
        proposal, forward_log = proposal_kernel.draw(current)
        proposed = recourse.solve(candidate["capacity"], proposal, True)
        reverse_log = proposal_kernel.log_probability(current["xi"], proposed)
        log_acceptance = ((proposed["value"] - current["value"])
                          / config.temperature + reverse_log - forward_log)
        accepted = 0
        if math.log(max(rng.random(), np.finfo(float).tiny)) <= min(0.0, log_acceptance):
            current, accepted = proposed, 1
        state["xi"] = np.asarray(current["xi"], dtype=float).copy()
        state["last_capacity"] = np.asarray(candidate["capacity"], dtype=float).copy()
        state["kappa"] = local_kappa + 1
        states[key] = state
        oracle_calls += 1
        mh_transitions += 1
        margin = float(current["selected_value"] - candidate["theta"])
        violated = margin > config.epsilon_cert
        metadata = {
            "stop_reason": None, "key": key, "label": int(state["label"]),
            "kappa_after": int(state["kappa"]), "transitions": 1,
        }
        trace.append({
            "event": "oracle_call", "attempt": attempt, "phase": phase,
            "oracle_call": oracle_calls, "chain_label": state["label"],
            "local_kappa": state["kappa"], "movement": movement,
            "transitions": 1, "accepted_transitions": accepted,
            "recourse_value": float(current["value"]),
            "selected_cut_value": float(current["selected_value"]),
            "theta": float(candidate["theta"]), "cut_margin": margin,
            "violated": bool(violated), "elapsed_sec": elapsed(),
        })
        return current, violated, metadata

    def cut_refutes_certified(record, accepted):
        margin = (float(record["intercept"])
                  - float(record["u"] @ accepted["capacity"])
                  - float(accepted["master_theta"]))
        return margin > config.epsilon_cert, margin

    def callback(cb_model, where):
        nonlocal attempt_counter, continuation_clean_calls, rejected_candidates
        nonlocal best_certified, certified_upper, termination_reason
        nonlocal certificate_failure, callback_error, revisited_integer_attempts
        nonlocal same_integer_changed_continuous_attempts
        if where != GRB.Callback.MIPSOL:
            return
        try:
            reason = limit_reason()
            if reason is not None:
                termination_reason = reason
                cb_model.terminate()
                return
            if attempt_counter >= config.max_certification_attempts:
                termination_reason = "certification_attempt_cap"
                cb_model.terminate()
                return
            candidate = {
                "y": np.array([round(cb_model.cbGetSolution(y[i]))
                               for i in range(instance["dim"])]),
                "capacity": np.array([cb_model.cbGetSolution(capacity[i])
                                      for i in range(instance["dim"])]),
                "theta": float(cb_model.cbGetSolution(eta)),
                "solver_objective": float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)),
                "solver_bound": float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJBND)),
            }
            attempt_counter += 1
            attempt = attempt_counter
            key, state = ensure_state(candidate)
            distance = float(np.linalg.norm(
                np.asarray(candidate["capacity"], dtype=float)
                - state["last_capacity"]))
            integer_revisit = state["candidate_visits"] > 0
            continuous_changed = integer_revisit and distance > 1e-9
            state["candidate_visits"] += 1
            state["continuous_signatures"].add(tuple(np.round(
                np.asarray(candidate["capacity"], dtype=float), 9)))
            states[key] = state
            revisited_integer_attempts += int(integer_revisit)
            same_integer_changed_continuous_attempts += int(continuous_changed)
            trace.append({
                "event": "attempt_started", "attempt": attempt,
                "binary_key": key, "chain_label": int(state["label"]),
                "integer_revisit": bool(integer_revisit),
                "continuous_changed_for_same_integer": bool(continuous_changed),
                "distance_from_previous_continuous": distance,
                "solver_objective": candidate["solver_objective"],
                "solver_bound": candidate["solver_bound"],
                "theta": candidate["theta"], "elapsed_sec": elapsed(),
            })

            def reject(selected, phase, metadata):
                nonlocal rejected_candidates, termination_reason, certificate_failure
                record, is_new = register_cut(selected, phase)
                cb_model.cbLazy(
                    cut_expression(record) >= float(record["intercept"]))
                rejected_candidates += 1
                contradictions = []
                for accepted_candidate in certified_history:
                    contradicts, margin = cut_refutes_certified(
                        record, accepted_candidate)
                    if contradicts:
                        contradictions.append({
                            "certified_attempt": accepted_candidate["attempt"],
                            "margin": float(margin),
                        })
                trace.append({
                    "event": "candidate_rejected", "attempt": attempt,
                    "phase": phase, "chain_label": metadata["label"],
                    "local_kappa": metadata["kappa_after"],
                    "cut_was_new": bool(is_new),
                    "cut_margin": float(
                        selected["selected_value"] - candidate["theta"]),
                    "contradictions": contradictions, "elapsed_sec": elapsed(),
                })
                if contradictions:
                    certificate_failure = {
                        "detected_at_attempt": attempt,
                        "contradictions": contradictions,
                    }
                    termination_reason = "certificate_failure"
                    cb_model.terminate()

            selected, violated, metadata = oracle_call(candidate, attempt, "fresh")
            if metadata["stop_reason"] is not None:
                termination_reason = metadata["stop_reason"]
                cb_model.terminate()
                return
            if violated:
                reject(selected, "fresh", metadata)
                return

            label = metadata["label"]
            checkpoint_kappa = metadata["kappa_after"]
            log_ratio = initial_log_ratio(
                label, checkpoint_kappa, config.epsilon_false)
            planned_continuations = continuation_calls_required(
                log_ratio, constants["log_rho"])
            false_acceptance_budget = (
                config.epsilon_false
                / (label * (label + 1.0)
                   * checkpoint_kappa * (checkpoint_kappa + 1.0)))
            trace.append({
                "event": "certification_checkpoint", "attempt": attempt,
                "chain_label": label, "checkpoint_kappa": checkpoint_kappa,
                "initial_log_ratio": log_ratio, "rho": constants["rho"],
                "planned_continuation_calls": planned_continuations,
                "false_acceptance_budget": false_acceptance_budget,
                "elapsed_sec": elapsed(),
            })

            continuation_calls = 0
            while log_ratio > 0.0:
                selected, violated, continuation = oracle_call(
                    candidate, attempt, "continuation")
                if continuation["stop_reason"] is not None:
                    termination_reason = continuation["stop_reason"]
                    cb_model.terminate()
                    return
                continuation_calls += 1
                if violated:
                    reject(selected, "continuation", continuation)
                    return
                continuation_clean_calls += 1
                log_ratio += constants["log_rho"]

            master_objective = (first_stage_value(
                instance, candidate["y"], candidate["capacity"])
                + candidate["theta"])
            candidate_upper = master_objective + config.epsilon_cert
            accepted = {
                "attempt": attempt, "y": candidate["y"].copy(),
                "capacity": candidate["capacity"].copy(),
                "z": candidate["capacity"].copy(),
                "theta": candidate["theta"] + config.epsilon_cert,
                "master_theta": candidate["theta"],
                "master_objective": float(master_objective),
                "objective": float(candidate_upper), "chain_label": label,
                "checkpoint_kappa": checkpoint_kappa,
                "continuation_calls": continuation_calls,
            }
            certified_history.append(accepted)
            if candidate_upper < certified_upper:
                certified_upper = float(candidate_upper)
                best_certified = accepted.copy()
            trace.append({
                "event": "candidate_certified", "attempt": attempt,
                "chain_label": label, "checkpoint_kappa": checkpoint_kappa,
                "continuation_calls": continuation_calls,
                "certified_objective": candidate_upper,
                "certified_upper": certified_upper,
                "solver_bound": candidate["solver_bound"],
                "conditional_gap_at_acceptance": max(
                    0.0, certified_upper - candidate["solver_bound"]),
                "elapsed_sec": elapsed(),
            })
        except Exception as exception:
            callback_error = repr(exception)
            termination_reason = "callback_error"
            cb_model.terminate()

    solver_status, node_count = None, math.nan
    conditional_lower, solver_incumbent = -math.inf, math.inf
    secondary_lp_failures = 0
    try:
        model.Params.TimeLimit = max(0.01, deadline - time.perf_counter())
        model.optimize(callback)
        solver_status = int(model.Status)
        node_count = float(model.NodeCount)
        try:
            conditional_lower = float(model.ObjBound)
        except gp.GurobiError:
            conditional_lower = -math.inf
        if model.SolCount > 0:
            solver_incumbent = float(model.ObjVal)
    finally:
        secondary_lp_failures = recourse.secondary_failures
        recourse.close()
        model.dispose()

    certified_gap = (
        max(0.0, certified_upper - conditional_lower)
        if math.isfinite(certified_upper) and math.isfinite(conditional_lower)
        else math.inf)
    if certificate_failure is not None:
        status = "certificate_failure"
    elif termination_reason is not None:
        status = termination_reason
    elif best_certified is None:
        status = ("no_certified_solution_infeasible"
                  if solver_status == GRB.INFEASIBLE
                  else f"no_certified_solution_status_{solver_status}")
    elif certified_gap <= config.epsilon_opt + 1e-9:
        status = "certified_gap_reached"
    elif solver_status == GRB.OPTIMAL:
        status = "solver_optimal_gap_accounting_failed"
    elif solver_status == GRB.TIME_LIMIT:
        status = "time_limit"
    else:
        status = f"solver_status_{solver_status}"

    algorithm_sec = elapsed()
    return {
        "method": "ours", "status": status,
        "certified": status == "certified_gap_reached",
        "has_certified_incumbent": best_certified is not None,
        "error": callback_error, "runtime_sec": algorithm_sec,
        "single_tree": True, "optimize_calls": 1,
        "solver_status": solver_status, "node_count": node_count,
        "iterations": attempt_counter,
        "certification_attempts": attempt_counter,
        "rejected_candidates": rejected_candidates,
        "certified_candidates": len(certified_history),
        "revisited_integer_attempts": revisited_integer_attempts,
        "same_integer_changed_continuous_attempts": (
            same_integer_changed_continuous_attempts),
        "integer_patterns_with_multiple_continuous_points": sum(
            len(value["continuous_signatures"]) > 1
            for value in states.values()),
        "max_continuous_points_per_integer": max(
            (len(value["continuous_signatures"]) for value in states.values()),
            default=0),
        "continuation_clean_calls": continuation_clean_calls,
        "oracle_calls": oracle_calls, "mh_transitions": mh_transitions,
        "cuts": len(cuts), "markov_states": len(states),
        "chain_labels_created": next_label,
        "certified_upper": certified_upper,
        "conditional_global_lower_bound": conditional_lower,
        "certified_absolute_gap": certified_gap,
        "lower_bound_scope": "conditional_on_fixed_rho",
        "solver_incumbent": solver_incumbent,
        "certificate_failure": certificate_failure,
        "solution": best_certified, "trace": trace,
        "constants": constants,
        "continuous_relaxation_warmup": {
            "enabled": bool(config.use_warm_start),
            "status": warmup["status"],
            "runtime_sec": float(warmup["runtime_sec"]),
            "iterations": int(warmup["iterations"]),
            "cuts": len(warmup["cuts"]),
            "method": warmup.get("method"),
            "alpha0": warmup.get("alpha0"),
            "initialization": warmup.get("initialization"),
        },
        "state_summary": {
            str(key): {
                "label": int(value["label"]),
                "kappa": int(value["kappa"]),
                "candidate_visits": int(value["candidate_visits"]),
                "distinct_continuous_points": len(value["continuous_signatures"]),
            }
            for key, value in states.items()
        },
        "secondary_lp_failures": secondary_lp_failures,
        "recourse_oracle": "exact primary LP + optimal-face min-L1 LP",
        "proposal_mode": PROPOSAL_MODE,
        "dual_vmf_kappa": proposal_kernel.kappa,
        "configuration": {
            "use_warm_start": config.use_warm_start,
            "warmup_iterations": config.warmup_iterations,
            "temperature": config.temperature,
            "epsilon_false": config.epsilon_false,
            "epsilon_opt": config.epsilon_opt,
            "epsilon_cert": config.epsilon_cert,
            "certification_rho": config.certification_rho,
        },
    }


def solve_ours(instance, reference=None, scenarios=None, target_gap=TARGET_GAP,
               time_limit=TIME_LIMIT, seed=0, perform_exact_checks=True,
               use_warm_start=None, warmup_iterations=None):
    """Run single-tree Algorithm 3, then optionally perform a benchmark check."""
    config = SingleTreeConfig(
        use_warm_start=(USE_WARM_START if use_warm_start is None
                        else bool(use_warm_start)),
        warmup_iterations=(WARMUP_ITERATIONS if warmup_iterations is None
                           else int(warmup_iterations)),
        temperature=float(W), epsilon_false=float(EPSILON_FALSE),
        epsilon_opt=float(EPSILON_OPT), epsilon_cert=float(EPSILON_CERT),
        certification_rho=float(CERTIFICATION_RHO),
        max_certification_attempts=int(MAX_CERTIFICATION_ATTEMPTS),
        max_total_oracle_calls=int(MAX_TOTAL_ORACLE_CALLS),
        time_limit=float(time_limit),
    )
    total_started = time.perf_counter()
    result = solve_single_tree_streamlined(instance, scenarios, config, seed)
    result.update({
        "gap": math.inf, "posthoc_robust_objective": math.nan,
        "excluded_check_time_sec": 0.0, "last_exact_check_status": "disabled",
        "reference_certified": bool(reference and reference.get("certified", False)),
        "reference_kind": ("not_used" if reference is None
                           else reference.get("reference_kind", "unknown")),
    })
    solution = result["solution"]
    if perform_exact_checks and reference is not None and solution is not None:
        reference_value = float(reference["objective"])
        check_started = time.perf_counter()
        robust_value, detail = exact_robust_objective(
            instance, solution["y"], solution["capacity"], scenarios,
            time_limit=EXACT_CHECK_TIME_LIMIT)
        result["excluded_check_time_sec"] = time.perf_counter() - check_started
        result["last_exact_check_status"] = detail["status"]
        result["posthoc_robust_objective"] = robust_value
        if detail["complete"] and math.isfinite(reference_value):
            result["gap"] = max(
                0.0, (robust_value - reference_value)
                / max(1.0, abs(reference_value)))
    result["total_wall_time_sec"] = time.perf_counter() - total_started
    result["benchmark_target_gap"] = float(target_gap)
    result["stopping_rule"] = (
        "single-tree absolute-gap certificate conditional on fixed rho; "
        "reference and exact separation are post-hoc benchmarks only")
    print(
        f"ours {instance['uncertainty']} d={instance['dim']} "
        f"rep={instance['rep']} attempts={result['certification_attempts']}: "
        f"algorithm={result['runtime_sec']:.3f}s, "
        f"gap={100.0 * result['gap']:.6f}%",
        flush=True,
    )
    return result




















def case_key(instance, scenarios=None):
    digest = hashlib.sha256()
    digest.update(json.dumps([
        "ellipsoid_complementarity_mode", ELLIPSOID_COMPLEMENTARITY_MODE,
        "aggregate_capacity_feasibility", False,
    ]).encode())
    for key in ("dim", "rep", "seed", "uncertainty"):
        digest.update(json.dumps([key, instance[key]]).encode())
    for key in ("fixed_cost", "capacity_cost", "transport", "emergency", "base",
                "sensitivity", "K"):
        digest.update(np.asarray(instance[key], dtype=np.float64).tobytes())
    for key in ("center", "axes", "rotation", "A", "gamma", "budget"):
        if key in instance:
            value = instance[key]
            digest.update(np.asarray(value).tobytes() if isinstance(value, np.ndarray)
                          else json.dumps([key, value]).encode())
    if scenarios is not None:
        digest.update(np.asarray(scenarios, dtype=np.float64).tobytes())
    return digest.hexdigest()


def run_case(instance, scenarios=None, reuse=REUSE, methods=("ours", "ldr")):
    methods = tuple(methods)
    allowed_methods = {"ours", "ldr"}
    if (not methods or len(set(methods)) != len(methods)
            or not set(methods).issubset(allowed_methods)):
        raise ValueError('methods must be ("ours",), ("ldr",), or ("ours", "ldr")')
    label = f"{instance['uncertainty']}_d{instance['dim']}_rep{instance['rep']}"
    if scenarios is not None:
        label += f"_I{len(scenarios)}"
    if instance["uncertainty"] == "budget":
        label += f"_Gamma{instance['gamma']:g}"
    metadata = {"label": label, "uncertainty": instance["uncertainty"],
                "dim": instance["dim"], "rep": instance["rep"],
                "scenario_count": len(scenarios) if scenarios is not None else None,
                "gamma": instance.get("gamma")}
    folder = OUTPUT_ROOT / label
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = case_key(instance, scenarios)
    effective_kappa = (0.5 * instance["dim"] if DUAL_VMF_KAPPA is None
                       else float(DUAL_VMF_KAPPA))
    proposal_configuration = {
        "mode": PROPOSAL_MODE,
        "kappa": effective_kappa if PROPOSAL_MODE == "dual_vmf" else None,
        "algorithm": "single_tree_sequential_certification",
        "use_warm_start": bool(USE_WARM_START),
        "warmup_iterations": int(WARMUP_ITERATIONS),
        "temperature": float(W),
        "epsilon_false": float(EPSILON_FALSE),
        "epsilon_opt": float(EPSILON_OPT),
        "epsilon_cert": float(EPSILON_CERT),
        "certification_rho": float(CERTIFICATION_RHO),
    }
    ours_fingerprint = hashlib.sha256(
        (fingerprint + json.dumps(proposal_configuration, sort_keys=True)).encode()
    ).hexdigest()
    proposal_tag = ("uniform" if PROPOSAL_MODE == "uniform"
                    else f"dual_vmf_kappa{effective_kappa:g}")
    reference_path = folder / "reference.json"
    ours_path = folder / f"ours_single_tree_{proposal_tag}.json"
    ldr_path = folder / "ldr.json"
    reference = None
    if reuse and reference_path.exists():
        candidate = json.loads(reference_path.read_text())
        if candidate.get("case_key") == fingerprint:
            reference = candidate
    if reference is None:
        print("REFERENCE", label, flush=True)
        reference = solve_ccg_reference(instance, scenarios)
        reference["case_key"] = fingerprint
        write_json(reference_path, reference)
    else:
        print("REFERENCE reuse", label, flush=True)
    ours = {}
    if "ours" in methods:
        ours = None
        if reuse and ours_path.exists():
            candidate = json.loads(ours_path.read_text())
            if candidate.get("case_key") == ours_fingerprint:
                ours = candidate
        if ours is None:
            ours = solve_ours(instance, reference, scenarios,
                              seed=instance["seed"] + 29)
            ours["case_key"] = ours_fingerprint
            ours["proposal_configuration"] = proposal_configuration
            write_json(ours_path, ours)
        else:
            print("OURS reuse", label, flush=True)
    ldr = {}
    if "ldr" in methods:
        ldr = None
        if reuse and ldr_path.exists():
            candidate = json.loads(ldr_path.read_text())
            if candidate.get("case_key") == fingerprint:
                ldr = candidate
        if ldr is None:
            print("LDR", label, flush=True)
            ldr = solve_ldr(instance, float(reference["objective"]), scenarios)
            ldr["case_key"] = fingerprint
            write_json(ldr_path, ldr)
        else:
            print("LDR reuse", label, flush=True)
    return {**metadata, "selected_methods": ",".join(methods),
            "reference_status": reference.get("status"),
            "reference_certified": reference.get("certified", False),
            "reference_kind": reference.get("reference_kind"),
            "reference_solver_gap": reference.get("solver_gap"),
            "reference_lower": reference.get("lower"),
            "reference_runtime_sec": reference.get("runtime_to_target_sec"),
            "reference_full_runtime_sec": reference.get("runtime_sec"),
            "reference_reporting_gap": reference.get("reporting_target_gap", TARGET_GAP),
            "reference_solver_target_gap": reference.get("solver_target_gap", REFERENCE_GAP),
            "reference_objective": reference.get("objective"),
            "ours_status": ours.get("status"), "ours_runtime_sec": ours.get("runtime_sec"),
            "ours_certified": ours.get("certified"),
            "ours_total_wall_time_sec": ours.get("total_wall_time_sec"),
            "ours_gap": ours.get("gap"), "ours_iterations": ours.get("iterations"),
            "ours_warmup_runtime_sec": ours.get("continuous_relaxation_warmup", {}).get("runtime_sec"),
            "ours_warmup_iterations": ours.get("continuous_relaxation_warmup", {}).get("iterations"),
            "ours_warmup_cuts": ours.get("continuous_relaxation_warmup", {}).get("cuts"),
            "secondary_lp_failures": ours.get("secondary_lp_failures"),
            "ldr_status": ldr.get("status"), "ldr_runtime_sec": ldr.get("runtime_sec"),
            "ldr_gap": ldr.get("gap")}


def save_experiment_table(name, rows):
    frame = pd.DataFrame(rows)
    path = OUTPUT_ROOT / f"{name}_raw.csv"
    frame.to_csv(path, index=False)
    summary = (frame.groupby([c for c in ("dim", "scenario_count", "gamma") if c in frame and frame[c].notna().any()],
                             dropna=False)
               .agg(n=("rep", "count"),
                    reached=("ours_status", lambda x: int((x.astype(str) == "certified_gap_reached").sum())),
                    certified_references=("reference_certified", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
                     ours_mean_sec=("ours_runtime_sec", "mean"),
                     ours_std_sec=("ours_runtime_sec", "std"),
                     warmup_mean_sec=("ours_warmup_runtime_sec", "mean"),
                     warmup_mean_cuts=("ours_warmup_cuts", "mean"),
                    ours_mean_gap=("ours_gap", "mean"),
                    ccg_mean_sec=("reference_runtime_sec", "mean"),
                    ccg_std_sec=("reference_runtime_sec", "std"),
                    ldr_mean_sec=("ldr_runtime_sec", "mean"),
                    ldr_std_sec=("ldr_runtime_sec", "std"),
                    ldr_mean_gap=("ldr_gap", "mean"))
               .reset_index())
    summary.to_csv(OUTPUT_ROOT / f"{name}_summary.csv", index=False)
    display(summary)
    return frame, summary


# ---------------------------------------------------------------------------
# Reproducible Section 5.2 experiment entry points
# ---------------------------------------------------------------------------

def run_ellipsoid_experiment(
    methods=("ours", "ldr"), dims=(3, 10, 30), reps=REPS, reuse=REUSE
):
    rows = []
    for dim in tuple(dims):
        for rep in tuple(reps):
            instance = make_instance(dim, rep, "ellipsoid")
            rows.append(run_case(instance, reuse=reuse, methods=methods))
            pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "ellipsoid_progress.csv", index=False)
    return save_experiment_table("ellipsoid", rows)


def run_scenario_experiment(
    methods=("ours", "ldr"), dims=(3, 10, 30),
    scenario_sizes=(1_000, 5_000, 10_000, 50_000, 100_000),
    reps=REPS, reuse=REUSE,
):
    rows = []
    for dim in tuple(dims):
        for scenario_count in tuple(scenario_sizes):
            for rep in tuple(reps):
                instance = make_instance(dim, rep, "scenario")
                scenarios = make_scenarios(instance, scenario_count)
                rows.append(run_case(
                    instance, scenarios, reuse=reuse, methods=methods))
                pd.DataFrame(rows).to_csv(
                    OUTPUT_ROOT / "scenario_progress.csv", index=False)
    return save_experiment_table("scenario", rows)


def run_budget_experiment(
    methods=("ours", "ldr"), dims=(10, 30, 50, 70, 100),
    gammas=(0.1, 0.3, 0.5, 0.7, 0.9), reps=REPS, reuse=REUSE,
):
    rows = []
    for dim in tuple(dims):
        for gamma in tuple(gammas):
            for rep in tuple(reps):
                instance = make_instance(dim, rep, "budget", gamma=gamma)
                rows.append(run_case(instance, reuse=reuse, methods=methods))
                pd.DataFrame(rows).to_csv(
                    OUTPUT_ROOT / "budget_progress.csv", index=False)
    return save_experiment_table("budget", rows)
