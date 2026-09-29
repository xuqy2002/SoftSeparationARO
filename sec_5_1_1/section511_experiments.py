"""Implementation of the Section 5.1.1 experiments.
"""
from pathlib import Path
import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "section511_results"
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Instance and finite-scenario generation
# ---------------------------------------------------------------------------

def _ellipsoid_parameters(dim, rng):
    center = np.full(dim, 0.5)
    axes = rng.uniform(0.10, 0.50, dim)
    rotation, triangular = np.linalg.qr(rng.normal(size=(dim, dim)))
    rotation *= np.where(np.diag(triangular) < 0, -1.0, 1.0)
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    return center, axes, rotation


def _ellipsoid_from_unit(instance, unit):
    return instance["center"] + np.asarray(unit) @ instance["transform"].T


def _ellipsoid_to_unit(instance, point):
    return ((np.asarray(point) - instance["center"]) @ instance["rotation"]) / instance["axes"]


def _make_instance(rep):
    dim = 3
    seed = 100000 * dim + 1000 * int(rep)
    rng = np.random.default_rng(seed)
    fixed = rng.integers(300, 400, dim).astype(float)
    capacity = rng.integers(10, 30, dim).astype(float)
    transport = rng.integers(20, 40, (dim, dim)).astype(float)
    base = rng.integers(200, 300, dim).astype(float)
    sensitivity = 0.1 * rng.integers(1, 5, dim) * base
    geometry_rng = np.random.default_rng(np.random.SeedSequence([seed, 20260916]))
    center, axes, rotation = _ellipsoid_parameters(dim, geometry_rng)
    return {
        "dim": dim,
        "rep": int(rep),
        "seed": seed,
        "c": np.r_[fixed / 20.0, capacity],
        "transport": transport,
        "base": base,
        "sensitivity": sensitivity,
        "emergency": 100,
        "y_upper": 20.0,
        "slope": 40.0,
        "center": center,
        "axes": axes,
        "rotation": rotation,
        "transform": rotation * axes,
    }


def _sample_surface(instance, rng, size):
    chunks = []
    remaining = int(size)
    while remaining:
        count = min(max(2 * remaining, 8), 8192)
        unit = rng.normal(size=(count, instance["dim"]))
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        probability = np.min(instance["axes"]) * np.linalg.norm(
            unit / instance["axes"], axis=1
        )
        selected = unit[rng.random(count) <= probability][:remaining]
        chunks.append(_ellipsoid_from_unit(instance, selected))
        remaining -= len(selected)
    return np.concatenate(chunks)


def _make_scenarios(instance, count):
    rng = np.random.default_rng(instance["seed"] + int(count) + 17)
    return _sample_surface(instance, rng, int(count))


def _demand(instance, scenario):
    return instance["base"] + instance["sensitivity"] * np.asarray(scenario)


# ---------------------------------------------------------------------------
# LP recourse, exact finite separation, and CCG reference
# ---------------------------------------------------------------------------

def _new_model(name):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = 1e-4
    model.Params.OptimalityTol = 1e-4
    model.Params.MIPGap = 1e-6
    model.Params.MIPGapAbs = 1e-8
    return model


class _Recourse:
    def __init__(self, instance, select_min_l1=False, deterministic=False):
        self.instance = instance
        self.select_min_l1 = bool(select_min_l1)
        self.deterministic = bool(deterministic)
        dim = instance["dim"]
        self.model = _new_model("section511_recourse")
        if self.deterministic:
            self.model.Params.Method = 1
            self.model.Params.Seed = 0
        self.u = self.model.addVars(dim, lb=0, ub=instance["emergency"], name="u")
        self.v = self.model.addVars(dim, lb=0, ub=instance["emergency"], name="v")
        for i in range(dim):
            for j in range(dim):
                self.model.addConstr(
                    self.v[j] - self.u[i] <= instance["transport"][i, j]
                )
        self.model.update()

    def solve(self, decision, scenario):
        dim = self.instance["dim"]
        scenario_demand = _demand(self.instance, scenario)
        objective = gp.quicksum(
            float(scenario_demand[j]) * self.v[j] for j in range(dim)
        )
        objective -= gp.quicksum(
            float(decision[dim + i]) * self.u[i] for i in range(dim)
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)
        if self.deterministic:
            self.model.reset()
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Recourse LP not optimal: {self.model.Status}")
        value = float(self.model.ObjVal)
        if not self.select_min_l1:
            u = np.array([self.u[i].X for i in range(dim)])
            return value, np.r_[np.zeros(dim), -u]

        face = self.model.addConstr(objective == value, name="optimal_face")
        try:
            self.model.setObjective(gp.quicksum(self.u.values()), GRB.MINIMIZE)
            self.model.reset()
            self.model.optimize()
            if self.model.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Dual selection not optimal: {self.model.Status}")
            selected_value = float(objective.getValue())
            u = np.array([self.u[i].X for i in range(dim)])
            return selected_value, np.r_[np.zeros(dim), -u]
        finally:
            self.model.remove(face)
            self.model.setObjective(objective, GRB.MAXIMIZE)
            self.model.update()

    def close(self):
        self.model.dispose()


class _FiniteSeparation:
    def __init__(self, instance, scenarios):
        self.scenarios = np.asarray(scenarios, dtype=float)
        self.oracle = _Recourse(instance)

    def solve(self, decision):
        worst_value = -math.inf
        worst_scenario = None
        self.oracle.model.reset(1)
        for scenario in self.scenarios:
            value, _ = self.oracle.solve(decision, scenario)
            if value > worst_value:
                worst_value = value
                worst_scenario = scenario.copy()
        return worst_value, worst_scenario

    def close(self):
        self.oracle.close()


def _first_stage_model(instance):
    model = _new_model("section511_ccg_master")
    dim = instance["dim"]
    decision = model.addVars(2 * dim, lb=0, name="x")
    for i in range(dim):
        decision[i].UB = instance["y_upper"]
        model.addConstr(decision[dim + i] <= instance["slope"] * decision[i])
    eta = model.addVar(lb=0, name="eta")
    model.setObjective(
        gp.quicksum(float(instance["c"][i]) * decision[i] for i in range(2 * dim)) + eta,
        GRB.MINIMIZE,
    )
    return model, decision, eta


def _add_scenario(model, decision, eta, instance, scenario, label):
    dim = instance["dim"]
    shipment = model.addVars(dim + 1, dim, lb=0, name=f"r_{label}")
    for i in range(dim):
        model.addConstr(
            gp.quicksum(shipment[i, j] for j in range(dim)) <= decision[dim + i]
        )
    scenario_demand = _demand(instance, scenario)
    for j in range(dim):
        model.addConstr(
            gp.quicksum(shipment[i, j] for i in range(dim + 1))
            >= float(scenario_demand[j])
        )
    model.addConstr(
        eta
        >= gp.quicksum(
            float(instance["transport"][i, j]) * shipment[i, j]
            for i in range(dim)
            for j in range(dim)
        )
        + instance["emergency"] * gp.quicksum(shipment[dim, j] for j in range(dim))
    )


def _relative_bound_gap(lower, upper):
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return math.inf
    return max(0.0, upper - lower) / max(1.0, abs(lower))


def _solve_reference(instance, scenarios):
    master, decision, eta = _first_stage_model(instance)
    separator = _FiniteSeparation(instance, scenarios)
    lower = -math.inf
    upper = math.inf
    try:
        _add_scenario(master, decision, eta, instance, scenarios[0], 0)
        for iteration in range(1, 10_001):
            master.optimize()
            if master.Status != GRB.OPTIMAL:
                raise RuntimeError(f"CCG master not optimal: {master.Status}")
            lower = max(lower, float(master.ObjVal))
            candidate = np.array([decision[i].X for i in range(2 * instance["dim"])])
            recourse_value, worst_scenario = separator.solve(candidate)
            upper = min(upper, float(instance["c"] @ candidate) + recourse_value)
            if _relative_bound_gap(lower, upper) <= 1e-6:
                return {
                    "objective": upper,
                    "lower": lower,
                    "upper": upper,
                    "iterations": iteration,
                }
            _add_scenario(master, decision, eta, instance, worst_scenario, iteration)
        raise RuntimeError("CCG did not reach a 1e-6 relative gap")
    finally:
        separator.close()
        master.dispose()


# ---------------------------------------------------------------------------
# Fixed proposal, projection, step size, averaging, and trajectory
# ---------------------------------------------------------------------------

def _midpoint(instance):
    dim = instance["dim"]
    first = np.full(dim, 0.5 * instance["y_upper"])
    capacity = 0.5 * instance["slope"] * first
    return np.r_[first, capacity]


def _fixed_preconditioner(instance):
    dim = instance["dim"]
    bounds = np.r_[
        np.abs(instance["c"][:dim]),
        np.maximum(
            np.abs(instance["c"][dim:]),
            np.abs(instance["c"][dim:] - instance["emergency"]),
        ),
    ]
    bounds = np.maximum(bounds, 1e-8)
    return bounds / np.exp(np.mean(np.log(bounds)))


def _project_metric(point, instance, diagonal):
    dim = instance["dim"]
    slope = instance["slope"]
    upper = instance["y_upper"]
    first = np.asarray(point[:dim])
    capacity = np.asarray(point[dim:])
    first_weight = diagonal[:dim]
    capacity_weight = diagonal[dim:]
    line_first = np.clip(
        (first_weight * first + slope * capacity_weight * capacity)
        / (first_weight + slope * slope * capacity_weight),
        0,
        upper,
    )
    candidates = np.stack(
        [
            np.column_stack([np.clip(first, 0, upper), np.zeros(dim)]),
            np.column_stack([np.full(dim, upper), np.clip(capacity, 0, slope * upper)]),
            np.column_stack([line_first, slope * line_first]),
        ],
        axis=1,
    )
    pairs = np.column_stack([first, capacity])
    weights = np.column_stack([first_weight, capacity_weight])
    distances = np.sum(weights[:, None, :] * (candidates - pairs[:, None, :]) ** 2, axis=2)
    selected = candidates[np.arange(dim), np.argmin(distances, axis=1)]
    feasible = (
        (first >= 0)
        & (first <= upper)
        & (capacity >= 0)
        & (capacity <= slope * first)
    )
    selected[feasible] = pairs[feasible]
    return np.r_[selected[:, 0], selected[:, 1]]


def _dual_proposal_gradient(instance, recourse_gradient):
    dim = instance["dim"]
    u = -np.asarray(recourse_gradient)[dim:]
    v = np.minimum(
        instance["emergency"],
        np.min(instance["transport"] + u[:, None], axis=0),
    )
    return instance["sensitivity"] * v


class _ScenarioProposal:
    def __init__(self, instance, scenarios, rng, adaptive):
        self.instance = instance
        self.scenarios = np.asarray(scenarios)
        self.rng = rng
        self.adaptive = bool(adaptive)
        self.kappa = 10.0
        self.units = _ellipsoid_to_unit(instance, scenarios) if adaptive else None
        if adaptive and not np.allclose(np.linalg.norm(self.units, axis=1), 1.0, atol=1e-7):
            raise ValueError("Finite scenarios must lie on the ellipsoid surface")
        self.forward_log_probability = None

    def uniform(self):
        return self.scenarios[self.rng.integers(len(self.scenarios))].copy()

    def _center(self, scenario_gradient):
        direction = self.instance["transform"].T @ scenario_gradient
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            direction = np.zeros(self.instance["dim"])
            direction[0] = 1.0
            return direction
        return direction / norm

    def _log_probabilities(self, center):
        logits = self.kappa * (self.units @ center)
        maximum = float(np.max(logits))
        return logits - maximum - np.log(np.exp(logits - maximum).sum())

    def draw(self, scenario_gradient):
        if not self.adaptive:
            return self.uniform()
        log_probabilities = self._log_probabilities(self._center(scenario_gradient))
        index = self.rng.choice(len(log_probabilities), p=np.exp(log_probabilities))
        self.forward_log_probability = float(log_probabilities[index])
        return self.scenarios[index].copy()

    def hastings_correction(self, current, candidate_gradient):
        if not self.adaptive:
            return 0.0
        indices = np.flatnonzero(np.all(self.scenarios == current, axis=1))
        if not len(indices):
            raise ValueError("Current scenario is absent from the finite scenario set")
        reverse = self._log_probabilities(self._center(candidate_gradient))
        return float(reverse[indices[0]] - self.forward_log_probability)


def _step_size(iteration):
    if iteration < 225:
        stage = (iteration - 1) // 15
        local_iteration = (iteration - 1) % 15 + 1
        return (15.0 - stage) / math.sqrt(local_iteration)
    return 1.0 / math.sqrt(iteration - 224)


def _run_trajectory(
    instance, scenarios, reference, temperature, adaptive, iterations, check_every
):
    optimum = float(reference["objective"])
    recourse = _Recourse(instance, select_min_l1=True, deterministic=True)
    evaluator = _FiniteSeparation(instance, scenarios)
    rng = np.random.default_rng(instance["seed"] + len(scenarios) + 29)
    proposal = _ScenarioProposal(instance, scenarios, rng, adaptive)
    decision = _midpoint(instance)
    scenario = proposal.uniform()
    metric = _fixed_preconditioner(instance)
    cumulative_gradient = np.zeros_like(decision)
    dual_averaging_anchor = None
    weighted_sum = np.zeros_like(decision)
    prefix = [weighted_sum.copy()]
    total_weight = 0.0
    rows = []
    algorithm = "Algorithm 1 (adaptive proposal)" if adaptive else "Algorithm 1 (uniform proposal)"
    start = time.perf_counter()
    excluded_time = 0.0
    try:
        for iteration in range(iterations + 1):
            if iteration:
                current_value, current_gradient = recourse.solve(decision, scenario)
                proposal_gradient = _dual_proposal_gradient(instance, current_gradient)
                candidate = proposal.draw(proposal_gradient)
                candidate_value, candidate_gradient = recourse.solve(decision, candidate)
                candidate_proposal_gradient = _dual_proposal_gradient(instance, candidate_gradient)
                correction = proposal.hastings_correction(scenario, candidate_proposal_gradient)
                log_ratio = (candidate_value - current_value) / temperature + correction
                if math.log(max(rng.random(), np.finfo(float).tiny)) <= min(0.0, log_ratio):
                    scenario = candidate
                    current_gradient = candidate_gradient

                alpha = _step_size(iteration)
                gradient = instance["c"] + current_gradient
                if iteration == 225:
                    dual_averaging_anchor = decision.copy()
                    cumulative_gradient.fill(0.0)
                if iteration >= 225:
                    cumulative_gradient += gradient
                    decision = _project_metric(
                        dual_averaging_anchor - alpha * cumulative_gradient / metric,
                        instance,
                        metric,
                    )
                else:
                    decision = _project_metric(
                        decision - alpha * gradient / metric,
                        instance,
                        metric,
                    )

                weight = float(iteration)
                weighted_sum += weight * decision
                total_weight += weight
                prefix.append(weighted_sum.copy())
                half = iteration // 2
                reported = (weighted_sum - prefix[half]) / (
                    total_weight - half * (half + 1) / 2
                )
            else:
                alpha = math.nan
                reported = decision.copy()

            if not (
                iteration == 0
                or iteration == iterations
                or iteration % check_every == 0
            ):
                continue
            algorithm_sec = time.perf_counter() - start - excluded_time
            evaluation_start = time.perf_counter()
            recourse_value, _ = evaluator.solve(reported)
            objective = float(instance["c"] @ reported) + recourse_value
            gap = 100.0 * (objective - optimum) / abs(optimum)
            excluded_time += time.perf_counter() - evaluation_start
            rows.append(
                {
                    "algorithm": algorithm,
                    "iteration": iteration,
                    "algorithm_sec": algorithm_sec,
                    "gap": gap,
                }
            )
            logging_start = time.perf_counter()
            print(
                f"algorithm={algorithm}, iteration={iteration}, "
                f"alg={algorithm_sec:.6f}s, gap={gap:.6f}%",
                flush=True,
            )
            excluded_time += time.perf_counter() - logging_start
        return rows
    finally:
        evaluator.close()
        recourse.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _write_source_and_summary(root, source, group_columns):
    root.mkdir(parents=True, exist_ok=True)
    source = pd.DataFrame(source)
    source_columns = [
        *group_columns,
        *(
            column
            for column in (
                "scenario_size",
                "rep",
                "algorithm",
                "iteration",
                "algorithm_sec",
                "gap",
            )
            if column not in group_columns
        ),
    ]
    missing = [column for column in source_columns if column not in source.columns]
    if missing:
        raise ValueError(f"Source data are missing required columns: {missing}")
    source = source[source_columns].sort_values([*group_columns, "rep", "iteration"])
    source.to_csv(root / "source_data.csv", index=False)
    summary_rows = []
    for keys, frame in source.groupby([*group_columns, "iteration"], sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = frame["gap"].to_numpy()
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        row = dict(zip([*group_columns, "iteration"], keys))
        row.update(
            n=len(frame),
            mean=float(values.mean()),
            q25=float(q25),
            median=float(median),
            q75=float(q75),
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values([*group_columns, "iteration"])
    summary.to_csv(root / "summary.csv", index=False)
    return {"output_dir": root, "data": source, "summary": summary}


def _load_packaged_data(filename, root, group_columns):
    path = DATA / filename
    if not path.is_file():
        return None
    source = pd.read_csv(path)
    if "gap" not in source and "gap_percent" in source:
        source = source.rename(columns={"gap_percent": "gap"})
    if "algorithm" not in source:
        if "proposal" in source:
            source["algorithm"] = source["proposal"].map({
                "uniform": "Algorithm 1 (uniform proposal)",
                "dual_vmf": "Algorithm 1 (adaptive proposal)",
            })
        else:
            source["algorithm"] = "Algorithm 1 (uniform proposal)"
    if "algorithm_sec" not in source:
        source["algorithm_sec"] = math.nan
    return _write_source_and_summary(root, source, group_columns)


def run_scenario_size_experiment():
    """Uniform proposal, w=0.001, 20 reps, and five scenario-set sizes."""
    root = RESULTS / "scenario_size"
    packaged = _load_packaged_data(
        "scenario_size.csv", root, ["scenario_size", "proposal"]
    )
    if packaged is not None:
        return packaged
    source = []
    for scenario_size in (1_000, 5_000, 10_000, 50_000, 100_000):
        for rep in range(1, 21):
            instance = _make_instance(rep)
            scenarios = _make_scenarios(instance, scenario_size)
            reference = _solve_reference(instance, scenarios)
            rows = _run_trajectory(
                instance,
                scenarios,
                reference,
                temperature=0.001,
                adaptive=False,
                iterations=10_000,
                check_every=100,
            )
            source.extend(
                {**row, "scenario_size": scenario_size, "proposal": "uniform", "rep": rep}
                for row in rows
            )
            _write_source_and_summary(root, source, ["scenario_size", "proposal"])
    return _write_source_and_summary(root, source, ["scenario_size", "proposal"])


def run_proposal_comparison_experiment():
    """Uniform versus dual-vMF, I=1,000, w=0.001, and 20 paired reps."""
    root = RESULTS / "proposal_comparison"
    packaged = _load_packaged_data(
        "proposal_comparison.csv", root, ["scenario_size", "proposal"]
    )
    if packaged is not None:
        return packaged
    source = []
    for rep in range(1, 21):
        instance = _make_instance(rep)
        scenarios = _make_scenarios(instance, 1_000)
        reference = _solve_reference(instance, scenarios)
        for proposal, adaptive in (("uniform", False), ("dual_vmf", True)):
            rows = _run_trajectory(
                instance,
                scenarios,
                reference,
                temperature=0.001,
                adaptive=adaptive,
                iterations=10_000,
                check_every=100,
            )
            source.extend(
                {**row, "scenario_size": 1_000, "proposal": proposal, "rep": rep}
                for row in rows
            )
        _write_source_and_summary(root, source, ["scenario_size", "proposal"])
    return _write_source_and_summary(root, source, ["scenario_size", "proposal"])


def run_temperature_sensitivity_experiment():
    """Uniform proposal, I=1,000, five temperatures, and 20 paired reps."""
    root = RESULTS / "temperature_sensitivity"
    packaged = _load_packaged_data(
        "temperature_sensitivity.csv", root, ["temperature"]
    )
    if packaged is not None:
        return packaged
    source = []
    for rep in range(1, 21):
        instance = _make_instance(rep)
        scenarios = _make_scenarios(instance, 1_000)
        reference = _solve_reference(instance, scenarios)
        for temperature in (1.0, 5.0, 10.0, 50.0, 100.0):
            rows = _run_trajectory(
                instance,
                scenarios,
                reference,
                temperature=temperature,
                adaptive=False,
                iterations=5_000,
                check_every=100,
            )
            source.extend(
                {**row, "temperature": temperature, "scenario_size": 1_000, "rep": rep}
                for row in rows
            )
        _write_source_and_summary(root, source, ["temperature"])
    return _write_source_and_summary(root, source, ["temperature"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_context(axis_label_size, tick_label_size, legend_font_size):
    import matplotlib as mpl

    return mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": tick_label_size,
            "axes.labelsize": axis_label_size,
            "xtick.labelsize": tick_label_size,
            "ytick.labelsize": tick_label_size,
            "legend.fontsize": legend_font_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _format_percent_axis(axis, y_limits):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    axis.set_yscale("log")
    axis.set_ylim(*y_limits)
    ticks = 10.0 ** np.arange(
        math.ceil(math.log10(y_limits[0])), math.floor(math.log10(y_limits[1])) + 1
    )
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}%"))
    axis.yaxis.set_minor_formatter(NullFormatter())


def _save_figure(figure, directory, stem, formats, dpi):
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for extension in formats:
        path = directory / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi)
        files.append(path)
    return files


def plot_scenario_size_experiment(
    results,
    *,
    min_iteration=0,
    max_iteration=None,
    y_limits=(0.001, 100.0),
    figsize=(7.2, 4.2),
    band_alpha=0.13,
    line_width=1.6,
    title=None,
    axis_label_size=9,
    tick_label_size=8,
    legend_font_size=8,
    formats=("pdf", "svg", "png", "tiff"),
    dpi=600,
    show=True,
):
    import matplotlib.pyplot as plt

    root = Path(results["output_dir"] if isinstance(results, dict) else results)
    summary = pd.read_csv(root / "summary.csv")
    summary = summary[summary["iteration"] >= min_iteration]
    if max_iteration is not None:
        summary = summary[summary["iteration"] <= max_iteration]
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555")
    linestyles = ("-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)))
    with _plot_context(axis_label_size, tick_label_size, legend_font_size):
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
        for index, scenario_size in enumerate(sorted(summary["scenario_size"].unique())):
            rows = summary[summary["scenario_size"] == scenario_size].sort_values("iteration")
            x = rows["iteration"].to_numpy()
            low = np.maximum(rows["q25"].to_numpy(), y_limits[0])
            median = rows["median"].to_numpy()
            high = rows["q75"].to_numpy()
            axis.fill_between(x, low, high, color=colors[index], alpha=band_alpha, linewidth=0)
            axis.plot(
                x,
                median,
                color=colors[index],
                linestyle=linestyles[index],
                linewidth=line_width,
                label=rf"$I={int(scenario_size):,}$",
            )
        _format_percent_axis(axis, y_limits)
        axis.set(
            xlabel="Iteration",
            ylabel="Optimality gap (%)",
            title=title,
            xlim=(min_iteration, max_iteration or int(summary["iteration"].max())),
        )
        axis.legend()
        files = _save_figure(figure, root / "figures", "scenario_size", formats, dpi)
        if show:
            plt.show()
        else:
            plt.close(figure)
    return {"figure": figure, "files": files, "summary": summary}


def plot_proposal_comparison_experiment(
    results,
    *,
    min_iteration=0,
    max_iteration=None,
    y_limits=(0.001, 100.0),
    figsize=(7.2, 4.2),
    band_alpha=0.13,
    line_width=1.6,
    title=None,
    axis_label_size=9,
    tick_label_size=8,
    legend_font_size=8,
    formats=("pdf", "svg", "png", "tiff"),
    dpi=600,
    show=True,
):
    import matplotlib.pyplot as plt

    root = Path(results["output_dir"] if isinstance(results, dict) else results)
    summary = pd.read_csv(root / "summary.csv")
    summary = summary[summary["iteration"] >= min_iteration]
    if max_iteration is not None:
        summary = summary[summary["iteration"] <= max_iteration]
    styles = {
        "uniform": ("Uniform proposal", "#0072B2", "-"),
        "dual_vmf": ("Adaptive proposal", "#D55E00", "--"),
    }
    with _plot_context(axis_label_size, tick_label_size, legend_font_size):
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
        for proposal in ("uniform", "dual_vmf"):
            rows = summary[summary["proposal"] == proposal].sort_values("iteration")
            label, color, linestyle = styles[proposal]
            x = rows["iteration"].to_numpy()
            low = np.maximum(rows["q25"].to_numpy(), y_limits[0])
            median = rows["median"].to_numpy()
            high = rows["q75"].to_numpy()
            axis.fill_between(x, low, high, color=color, alpha=band_alpha, linewidth=0)
            axis.plot(x, median, color=color, linestyle=linestyle, linewidth=line_width, label=label)
        _format_percent_axis(axis, y_limits)
        axis.set(
            xlabel="Iteration",
            ylabel="Optimality gap (%)",
            title=title,
            xlim=(min_iteration, max_iteration or int(summary["iteration"].max())),
        )
        axis.legend()
        files = _save_figure(figure, root / "figures", "proposal_comparison", formats, dpi)
        if show:
            plt.show()
        else:
            plt.close(figure)
    return {"figure": figure, "files": files, "summary": summary}


def plot_temperature_sensitivity_experiment(
    results,
    *,
    min_iteration=0,
    max_iteration=None,
    y_limits=(0.005, 105.0),
    figsize=(7.2, 4.2),
    show_band=True,
    band_alpha=0.06,
    line_width=1.6,
    title=None,
    axis_label_size=9,
    tick_label_size=8,
    legend_font_size=8,
    formats=("pdf", "svg", "png", "tiff"),
    dpi=600,
    show=True,
):
    import matplotlib.pyplot as plt

    root = Path(results["output_dir"] if isinstance(results, dict) else results)
    summary = pd.read_csv(root / "summary.csv")
    summary = summary[summary["iteration"] >= min_iteration]
    if max_iteration is not None:
        summary = summary[summary["iteration"] <= max_iteration]
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555")
    linestyles = ("-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)))
    with _plot_context(axis_label_size, tick_label_size, legend_font_size):
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
        for index, temperature in enumerate(sorted(summary["temperature"].unique())):
            rows = summary[summary["temperature"] == temperature].sort_values("iteration")
            x = rows["iteration"].to_numpy()
            low = np.maximum(rows["q25"].to_numpy(), y_limits[0])
            median = rows["median"].to_numpy()
            high = rows["q75"].to_numpy()
            if show_band:
                axis.fill_between(x, low, high, color=colors[index], alpha=band_alpha, linewidth=0)
            axis.plot(
                x,
                median,
                color=colors[index],
                linestyle=linestyles[index],
                linewidth=line_width,
                label=rf"$w={temperature:g}$",
            )
        _format_percent_axis(axis, y_limits)
        axis.set(
            xlabel="Iteration",
            ylabel="Optimality gap (%)",
            title=title,
            xlim=(min_iteration, max_iteration or int(summary["iteration"].max())),
        )
        axis.legend()
        files = _save_figure(figure, root / "figures", "temperature_sensitivity", formats, dpi)
        if show:
            plt.show()
        else:
            plt.close(figure)
    return {"figure": figure, "files": files, "summary": summary}
