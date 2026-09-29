"""Exact benchmarks for the Section 5.2 mixed-integer experiments.
"""

import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np


SOLVER_THREADS = 0
MASTER_MIP_GAP = 1e-4
REFERENCE_GAP = 1e-4
TARGET_GAP = 1e-3
TIME_LIMIT = 1800.0
MAX_CCG_ITERATIONS = 1000
ELLIPSOID_COMPLEMENTARITY_MODE = "tight_big_m"


def configure(runtime):
    """Bind shared primitives and current experiment controls."""
    names = (
        "SOLVER_THREADS",
        "MASTER_MIP_GAP",
        "REFERENCE_GAP",
        "TARGET_GAP",
        "TIME_LIMIT",
        "MAX_CCG_ITERATIONS",
        "ELLIPSOID_COMPLEMENTARITY_MODE",
        "ExactRecourseLP",
        "first_stage_value",
        "add_recourse_block",
        "make_master",
        "initial_scenario",
    )
    namespace = globals()
    for name in names:
        namespace[name] = getattr(runtime, name)


def emergency_only_fallback_reference(instance, scenarios=None):
    """A finite robust upper bound with no facilities opened."""
    y_value = np.zeros(instance["dim"], dtype=float)
    z_value = np.zeros(instance["dim"], dtype=float)
    weights = np.asarray(instance["emergency"], dtype=float) * np.asarray(instance["sensitivity"], dtype=float)
    recourse_upper = float(np.asarray(instance["emergency"], dtype=float) @ np.asarray(instance["base"], dtype=float))
    uncertainty = instance["uncertainty"]
    if uncertainty == "scenario":
        if scenarios is None or len(scenarios) == 0:
            raise ValueError("Finite-scenario fallback requires nonempty scenarios")
        support = float(np.max(np.asarray(scenarios, dtype=float) @ weights))
    elif uncertainty == "ellipsoid":
        support = float(weights @ instance["center"] + np.linalg.norm(instance["A"].T @ weights))
    elif uncertainty == "budget":
        budget = min(float(instance["budget"]), float(instance["dim"]))
        ordered = np.sort(np.maximum(weights, 0.0))[::-1]
        whole = int(math.floor(budget))
        support = float(np.sum(ordered[:whole]))
        if budget > whole and whole < len(ordered):
            support += (budget - whole) * float(ordered[whole])
    else:
        raise ValueError("unknown uncertainty set")
    objective = first_stage_value(instance, y_value, z_value) + recourse_upper + support
    return objective, y_value, z_value

class FiniteSeparator:
    def __init__(self, instance, scenarios):
        self.instance = instance
        self.scenarios = np.asarray(scenarios, dtype=float)
        self.recourse = ExactRecourseLP(instance)

    def solve(self, z, deadline=math.inf):
        best_value, best_xi, best_index = -math.inf, None, None
        for index, xi in enumerate(self.scenarios):
            if time.perf_counter() >= deadline:
                return {"value": best_value, "xi": best_xi, "index": best_index,
                        "complete": False, "status": "time_limit"}
            value = self.recourse.solve(z, xi, select_dual=False)["value"]
            if value > best_value:
                best_value, best_xi, best_index = value, xi.copy(), index
        return {"value": float(best_value), "xi": best_xi, "index": best_index,
                "complete": True, "status": "optimal"}

    def close(self):
        self.recourse.close()

class BudgetSeparator:
    """Exact budget dual MILP tightened by the Gabrel LP-(M)."""
    def __init__(self, instance):
        self.instance = instance
        inst, d = instance, instance["dim"]
        self.analytic_u = np.array([
            max(0.0, max(float(inst["emergency"][j]) - float(inst["transport"][i, j])
                         for j in range(d)))
            for i in range(d)
        ])

        # Gabrel LP-(M): solve the recourse dual at full-box demand to obtain
        # tight, instance- and incumbent-specific bounds before each MILP call.
        lp_m = gp.Model("gabrel_LP_M_full_demand")
        lp_m.Params.OutputFlag = 0
        lp_m.Params.Threads = SOLVER_THREADS
        lp_m.Params.Method = 1
        lp_m.Params.FeasibilityTol = 1e-8
        lp_m.Params.OptimalityTol = 1e-8
        self.lp_u = lp_m.addVars(
            d, lb=0.0,
            ub={i: float(self.analytic_u[i]) for i in range(d)}, name="u"
        )
        self.lp_v = lp_m.addVars(
            d, lb=0.0,
            ub={j: float(inst["emergency"][j]) for j in range(d)}, name="v"
        )
        for i in range(d):
            for j in range(d):
                lp_m.addConstr(self.lp_v[j] - self.lp_u[i]
                               <= float(inst["transport"][i, j]))
        lp_m.update()
        self.lp_m = lp_m

        model = gp.Model("budget_exact_dual_milp_gabrel")
        model.Params.OutputFlag = 0
        model.Params.Threads = SOLVER_THREADS
        model.Params.MIPGap = REFERENCE_GAP
        self.u = model.addVars(
            d, lb=0.0,
            ub={i: float(self.analytic_u[i]) for i in range(d)}, name="u"
        )
        self.v = model.addVars(
            d, lb=0.0,
            ub={j: float(inst["emergency"][j]) for j in range(d)}, name="v"
        )
        self.xi = model.addVars(d, vtype=GRB.BINARY, name="xi")
        self.product = model.addVars(d, lb=0.0, name="xi_times_v")
        self.product_gate = {}
        self.product_lower = {}
        for j in range(d):
            initial_m = float(inst["emergency"][j])
            self.product[j].UB = initial_m
            model.addConstr(self.product[j] <= self.v[j])
            self.product_gate[j] = model.addConstr(
                self.product[j] - initial_m * self.xi[j] <= 0
            )
            self.product_lower[j] = model.addConstr(
                self.product[j] - self.v[j] - initial_m * self.xi[j] >= -initial_m
            )
        model.addConstr(gp.quicksum(self.xi[j] for j in range(d))
                        <= int(inst["budget"]))
        for i in range(d):
            for j in range(d):
                model.addConstr(self.v[j] - self.u[i]
                                <= float(inst["transport"][i, j]))
        model.update()
        self.model = model
        self.gabrel_lp_calls = 0
        self.gabrel_lp_total_sec = 0.0

    @staticmethod
    def _set_remaining_time(model, deadline):
        if math.isfinite(deadline):
            model.Params.TimeLimit = max(0.01, deadline - time.perf_counter())

    def solve_lp_m(self, z, deadline):
        started = time.perf_counter()
        inst, d = self.instance, self.instance["dim"]
        full_demand = inst["base"] + inst["sensitivity"]
        self.lp_m.setObjective(
            gp.quicksum(float(full_demand[j]) * self.lp_v[j] for j in range(d))
            - gp.quicksum(float(z[i]) * self.lp_u[i] for i in range(d)),
            GRB.MAXIMIZE,
        )
        self.lp_m.update()
        self.lp_m.reset(1)
        self._set_remaining_time(self.lp_m, deadline)
        self.gabrel_lp_calls += 1
        self.lp_m.optimize()
        self.gabrel_lp_total_sec += time.perf_counter() - started
        if self.lp_m.Status != GRB.OPTIMAL:
            status = "time_limit" if self.lp_m.Status == GRB.TIME_LIMIT else f"status_{self.lp_m.Status}"
            return {"complete": False, "status": "gabrel_lp_" + status}

        raw_v = np.array([self.lp_v[j].X for j in range(d)])
        raw_u = np.array([self.lp_u[i].X for i in range(d)])
        padding = max(1e-7, 10.0 * float(self.lp_m.Params.FeasibilityTol))
        if (not np.all(np.isfinite(raw_v)) or not np.all(np.isfinite(raw_u))
                or np.min(raw_v) < -padding or np.min(raw_u) < -padding
                or np.max(raw_v - inst["emergency"]) > padding
                or np.max(raw_v[None, :] - raw_u[:, None] - inst["transport"]) > padding):
            return {"complete": False, "status": "gabrel_lp_invalid_solution"}

        v_upper = np.minimum(inst["emergency"], np.maximum(0.0, raw_v) + padding)
        # For fixed v, the componentwise-minimal feasible u is canonical and
        # preserves full-demand optimality because z is nonnegative.
        u_upper = np.minimum(
            self.analytic_u,
            np.maximum(0.0, np.max(v_upper[None, :] - inst["transport"], axis=1)) + padding,
        )
        return {
            "complete": True,
            "status": "optimal",
            "v_upper": v_upper,
            "u_upper": u_upper,
            "objective": float(self.lp_m.ObjVal),
            "padding": padding,
        }

    def solve(self, z, deadline=math.inf):
        inst, d = self.instance, self.instance["dim"]
        z = np.asarray(z, dtype=float)
        if z.shape != (d,) or not np.all(np.isfinite(z)) or np.any(z < -1e-9):
            raise ValueError("Budget separation requires finite nonnegative capacities")

        bounds = self.solve_lp_m(np.maximum(z, 0.0), deadline)
        if not bounds["complete"]:
            return {"value": math.nan, "xi": None, "complete": False,
                    "status": bounds["status"], "big_m_source": "gabrel_lp_m"}

        for j in range(d):
            big_m = float(bounds["v_upper"][j])
            self.v[j].UB = big_m
            self.product[j].UB = big_m
            self.model.chgCoeff(self.product_gate[j], self.xi[j], -big_m)
            self.model.chgCoeff(self.product_lower[j], self.xi[j], -big_m)
            self.product_lower[j].RHS = -big_m
        for i in range(d):
            self.u[i].UB = float(bounds["u_upper"][i])

        self.model.setObjective(
            gp.quicksum(float(inst["base"][j]) * self.v[j]
                        + float(inst["sensitivity"][j]) * self.product[j]
                        for j in range(d))
            - gp.quicksum(float(z[i]) * self.u[i] for i in range(d)),
            GRB.MAXIMIZE,
        )
        self.model.update()
        self.model.reset(1)
        self._set_remaining_time(self.model, deadline)
        self.model.optimize()
        complete = self.model.Status == GRB.OPTIMAL
        value = float(self.model.ObjVal) if self.model.SolCount else math.nan
        point = (np.array([self.xi[j].X for j in range(d)])
                 if self.model.SolCount else None)
        status = ("optimal" if complete else "time_limit"
                  if self.model.Status == GRB.TIME_LIMIT else f"status_{self.model.Status}")
        return {
            "value": value,
            "xi": point,
            "complete": complete,
            "status": status,
            "big_m_source": "gabrel_lp_m",
            "gabrel_lp_calls": self.gabrel_lp_calls,
            "gabrel_lp_total_sec": self.gabrel_lp_total_sec,
            "gabrel_lp_objective": bounds["objective"],
            "gabrel_padding": bounds["padding"],
        }

    def close(self):
        self.lp_m.dispose()
        self.model.dispose()

class EllipsoidSeparator:
    """Exact KKT separation with selectable SOS1 or tight-big-M complementarity."""
    def __init__(self, instance):
        self.instance = instance

    def solve(self, z, deadline=math.inf):
        inst, d = self.instance, self.instance["dim"]
        z = np.asarray(z, dtype=float)
        if z.shape != (d,) or not np.all(np.isfinite(z)) or np.any(z < -1e-9):
            raise ValueError("Ellipsoid separation requires finite nonnegative capacities")
        z = np.maximum(z, 0.0)
        mode = ELLIPSOID_COMPLEMENTARITY_MODE
        if mode not in ("sos1", "tight_big_m"):
            raise ValueError("ELLIPSOID_COMPLEMENTARITY_MODE must be 'sos1' or 'tight_big_m'")

        # Coordinate bounds are exact for xi = center + A*s, ||s||_2 <= 1.
        coordinate_radius = np.linalg.norm(inst["A"], axis=1)
        xi_min = inst["center"] - coordinate_radius
        xi_max = inst["center"] + coordinate_radius
        demand_min = inst["base"] + inst["sensitivity"] * xi_min
        demand_max = inst["base"] + inst["sensitivity"] * xi_max
        if np.min(demand_min) < -1e-8:
            raise ValueError("Tight-big-M derivation requires nonnegative ellipsoidal demand")
        demand_min = np.maximum(demand_min, 0.0)
        demand_max = np.maximum(demand_max, demand_min)

        # There exists an optimal canonical dual with these u bounds. Primal
        # bounds preserve an optimal no-oversupply transportation solution.
        u_upper = np.array([
            max(0.0, max(float(inst["emergency"][j]) - float(inst["transport"][i, j])
                         for j in range(d)))
            for i in range(d)
        ])
        total_capacity = float(np.sum(z))

        model = gp.Model("ellipsoid_exact_kkt_" + mode)
        model.Params.OutputFlag = 0
        model.Params.Threads = SOLVER_THREADS
        model.Params.MIPGap = REFERENCE_GAP
        if math.isfinite(deadline):
            model.Params.TimeLimit = max(0.01, deadline - time.perf_counter())
        sphere = model.addVars(d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sphere")
        xi = model.addVars(d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xi")
        flow = model.addVars(d, d, lb=0.0, name="flow")
        emergency = model.addVars(d, lb=0.0, name="emergency")
        u = model.addVars(d, lb=0.0, name="u")
        v = model.addVars(d, lb=0.0, name="v")
        cap_slack = model.addVars(d, lb=0.0, name="cap_slack")
        dem_slack = model.addVars(d, lb=0.0, name="dem_slack")
        rc_flow = model.addVars(d, d, lb=0.0, name="rc_flow")
        rc_emergency = model.addVars(d, lb=0.0, name="rc_emergency")

        model.addQConstr(gp.quicksum(sphere[k] * sphere[k] for k in range(d)) <= 1.0)
        for j in range(d):
            xi[j].LB = float(xi_min[j])
            xi[j].UB = float(xi_max[j])
            model.addConstr(xi[j] == float(inst["center"][j])
                            + gp.quicksum(float(inst["A"][j, k]) * sphere[k]
                                          for k in range(d)))
            v[j].UB = float(inst["emergency"][j])
            emergency[j].UB = float(demand_max[j])
            rc_emergency[j].UB = float(inst["emergency"][j])
            dem_slack[j].UB = max(
                0.0, total_capacity + float(demand_max[j] - demand_min[j])
            )
        for i in range(d):
            u[i].UB = float(u_upper[i])
            cap_slack[i].UB = float(z[i])
            model.addConstr(cap_slack[i] == float(z[i])
                            - gp.quicksum(flow[i, j] for j in range(d)))
        for j in range(d):
            demand = float(inst["base"][j]) + float(inst["sensitivity"][j]) * xi[j]
            model.addConstr(dem_slack[j] == gp.quicksum(flow[i, j] for i in range(d))
                            + emergency[j] - demand)
            model.addConstr(rc_emergency[j] == float(inst["emergency"][j]) - v[j])
        for i in range(d):
            for j in range(d):
                flow[i, j].UB = float(min(z[i], demand_max[j]))
                rc_flow[i, j].UB = float(inst["transport"][i, j] + u_upper[i])
                model.addConstr(rc_flow[i, j]
                                == float(inst["transport"][i, j]) + u[i] - v[j])

        if mode == "sos1":
            for i in range(d):
                model.addSOS(GRB.SOS_TYPE1, [cap_slack[i], u[i]])
            for j in range(d):
                model.addSOS(GRB.SOS_TYPE1, [dem_slack[j], v[j]])
                model.addSOS(GRB.SOS_TYPE1, [emergency[j], rc_emergency[j]])
            for i in range(d):
                for j in range(d):
                    model.addSOS(GRB.SOS_TYPE1, [flow[i, j], rc_flow[i, j]])
        else:
            def add_big_m_pair(left, right, left_m, right_m, name):
                left_m, right_m = float(left_m), float(right_m)
                if (not math.isfinite(left_m) or not math.isfinite(right_m)
                        or left_m < -1e-9 or right_m < -1e-9):
                    raise ValueError(f"Invalid complementarity bound for {name}")
                left_m, right_m = max(0.0, left_m), max(0.0, right_m)
                if left_m <= 1e-12:
                    model.addConstr(left == 0.0, name=name + "_left_fixed")
                    return
                if right_m <= 1e-12:
                    model.addConstr(right == 0.0, name=name + "_right_fixed")
                    return
                switch = model.addVar(vtype=GRB.BINARY, name=name)
                model.addConstr(left <= left_m * switch)
                model.addConstr(right <= right_m * (1 - switch))

            for i in range(d):
                add_big_m_pair(cap_slack[i], u[i], z[i], u_upper[i], f"comp_cap_{i}")
            for j in range(d):
                add_big_m_pair(
                    dem_slack[j], v[j],
                    max(0.0, total_capacity + float(demand_max[j] - demand_min[j])),
                    inst["emergency"][j], f"comp_demand_{j}"
                )
                add_big_m_pair(emergency[j], rc_emergency[j], demand_max[j],
                               inst["emergency"][j], f"comp_emergency_{j}")
            for i in range(d):
                for j in range(d):
                    add_big_m_pair(
                        flow[i, j], rc_flow[i, j],
                        min(z[i], demand_max[j]),
                        inst["transport"][i, j] + u_upper[i],
                        f"comp_flow_{i}_{j}"
                    )

        model.setObjective(
            gp.quicksum(float(inst["transport"][i, j]) * flow[i, j]
                        for i in range(d) for j in range(d))
            + gp.quicksum(float(inst["emergency"][j]) * emergency[j] for j in range(d)),
            GRB.MAXIMIZE,
        )
        model.optimize()
        complete = model.Status == GRB.OPTIMAL
        value = float(model.ObjVal) if model.SolCount else math.nan
        point = np.array([xi[j].X for j in range(d)]) if model.SolCount else None
        status = ("optimal" if complete else "time_limit"
                  if model.Status == GRB.TIME_LIMIT else f"status_{model.Status}")
        binary_count = int(model.NumBinVars)
        node_count = float(model.NodeCount)
        model.dispose()
        return {"value": value, "xi": point, "complete": complete, "status": status,
                "complementarity_mode": mode, "separation_binary_count": binary_count,
                "separation_node_count": node_count}

    def close(self):
        pass

def make_separator(instance, scenarios=None):
    if instance["uncertainty"] == "scenario":
        return FiniteSeparator(instance, scenarios)
    if instance["uncertainty"] == "ellipsoid":
        return EllipsoidSeparator(instance)
    if instance["uncertainty"] == "budget":
        return BudgetSeparator(instance)
    raise ValueError("unknown uncertainty set")

def exact_robust_objective(instance, y, z, scenarios=None, time_limit=TIME_LIMIT):
    separator = make_separator(instance, scenarios)
    try:
        result = separator.solve(z, time.perf_counter() + float(time_limit))
    finally:
        separator.close()
    if not result["complete"]:
        # A separation incumbent is only a lower bound on the worst-case
        # recourse value and therefore cannot certify the reported gap.
        incumbent = result.get("value", math.nan)
        objective = (first_stage_value(instance, y, z) + float(incumbent)
                     if incumbent is not None and math.isfinite(float(incumbent))
                     else math.nan)
        return objective, result
    return first_stage_value(instance, y, z) + float(result["value"]), result

def solve_ccg_reference(instance, scenarios=None, target_gap=REFERENCE_GAP,
                        max_iterations=MAX_CCG_ITERATIONS, time_limit=TIME_LIMIT):
    start = time.perf_counter()
    model, y, z, eta = make_master(instance, "exact_ccg_reference", scenarios)
    model.Params.MIPGap = float(target_gap)
    separator = make_separator(instance, scenarios)
    add_recourse_block(model, instance, y, z, eta, initial_scenario(instance, scenarios), "initial")
    # Complete emergency recourse makes the zero-investment solution feasible
    # and provides a finite robust upper bound if C&CG reaches its time limit.
    best_upper, best_y, best_z = emergency_only_fallback_reference(instance, scenarios)
    reference_kind = "emergency_only_fallback"
    lower, history, status = -math.inf, [], "max_iterations"
    try:
        for iteration in range(1, int(max_iterations) + 1):
            elapsed = time.perf_counter() - start
            if elapsed >= time_limit:
                status = "time_limit"
                break
            model.Params.TimeLimit = max(0.01, time_limit - elapsed)
            model.optimize()
            if model.SolCount == 0:
                status = f"master_status_{model.Status}"
                break
            lower = max(lower, float(model.ObjBound))
            y_value = np.array([round(y[i].X) for i in range(instance["dim"])])
            z_value = np.array([z[i].X for i in range(instance["dim"])])
            separated = separator.solve(z_value, start + time_limit)
            if not separated["complete"]:
                status = "separation_" + separated["status"]
                break
            upper = first_stage_value(instance, y_value, z_value) + float(separated["value"])
            if upper < best_upper:
                best_upper, best_y, best_z = upper, y_value.copy(), z_value.copy()
                reference_kind = "ccg_best_incumbent"
            gap = max(0.0, (best_upper - lower) / max(1.0, abs(best_upper)))
            history.append({"iteration": iteration, "runtime_sec": time.perf_counter() - start,
                            "lower": lower, "upper": best_upper, "gap": gap})
            if gap <= target_gap:
                status = "reached"
                break
            add_recourse_block(model, instance, y, z, eta, separated["xi"], f"cut_{iteration}")
    finally:
        separator.close()
        model.dispose()
    solver_gap = (max(0.0, (best_upper-lower)/max(1.0, abs(best_upper)))
                  if math.isfinite(best_upper) and math.isfinite(lower) else math.inf)
    certified = bool(status == "reached" and solver_gap <= target_gap)
    runtime_to_target_sec = next(
        (float(row["runtime_sec"]) for row in history
         if float(row["gap"]) <= TARGET_GAP),
        math.nan,
    )
    if certified:
        reference_kind = "certified_ccg"
    return {"method": "ccg_reference", "status": status,
            "runtime_sec": time.perf_counter() - start,
            "runtime_to_target_sec": runtime_to_target_sec,
            "reporting_target_gap": TARGET_GAP,
            "solver_target_gap": target_gap, "objective": best_upper,
            "lower": lower, "upper": best_upper,
            "solver_gap": solver_gap, "certified": certified,
            "reference_kind": reference_kind,
            "iterations": len(history), "x": {"y": best_y, "z": best_z}, "history": history}

def add_ellipsoid_support_le(model, constant, coefficients, rhs, instance, name):
    """constant + max_xi coefficients^T xi <= rhs for xi=c+A*u, ||u||<=1."""
    coefficients = list(coefficients)
    d = instance["dim"]
    transformed = model.addVars(d, lb=-GRB.INFINITY, name=f"{name}_At_coeff")
    norm = model.addVar(lb=0.0, name=f"{name}_norm")
    for k in range(d):
        model.addConstr(transformed[k] == gp.quicksum(
            float(instance["A"][j, k]) * coefficients[j] for j in range(d)))
    model.addGenConstrNorm(norm, [transformed[k] for k in range(d)], 2.0)
    mean = constant + gp.quicksum(float(instance["center"][j]) * coefficients[j]
                                  for j in range(d))
    model.addConstr(mean + norm <= rhs)

def add_budget_support_le(model, constant, coefficients, rhs, instance, name):
    """Bertsimas-Sim counterpart for max{a^T xi: 0<=xi<=1, sum xi<=B}."""
    coefficients = list(coefficients)
    d = instance["dim"]
    theta = model.addVar(lb=0.0, name=f"{name}_theta")
    excess = model.addVars(d, lb=0.0, name=f"{name}_excess")
    for j in range(d):
        model.addConstr(theta + excess[j] >= coefficients[j])
    model.addConstr(constant + float(instance["budget"]) * theta
                    + gp.quicksum(excess[j] for j in range(d)) <= rhs)

def make_ldr_model(instance, name, scenarios=None):
    d = instance["dim"]
    model, y, z, eta = make_master(instance, name, scenarios)
    flow0 = model.addVars(d, d, lb=-GRB.INFINITY, name="flow0")
    flow_slope = model.addVars(d, d, d, lb=-GRB.INFINITY, name="flow_slope")
    emergency0 = model.addVars(d, lb=-GRB.INFINITY, name="emergency0")
    emergency_slope = model.addVars(d, d, lb=-GRB.INFINITY, name="emergency_slope")
    return model, y, z, eta, flow0, flow_slope, emergency0, emergency_slope

def add_ldr_scenario_constraints(model, instance, variables, xi, label):
    y, z, eta, flow0, flow_slope, emergency0, emergency_slope = variables
    d = instance["dim"]
    xi = np.asarray(xi, dtype=float)
    flow = {(i, j): flow0[i, j] + gp.quicksum(float(xi[k]) * flow_slope[i, j, k]
                                               for k in range(d))
            for i in range(d) for j in range(d)}
    emergency = {j: emergency0[j] + gp.quicksum(float(xi[k]) * emergency_slope[j, k]
                                                 for k in range(d)) for j in range(d)}
    for i in range(d):
        for j in range(d):
            model.addConstr(flow[i, j] >= 0.0, name=f"{label}_flow_nonnegative_{i}_{j}")
    for j in range(d):
        model.addConstr(emergency[j] >= 0.0, name=f"{label}_emergency_nonnegative_{j}")
    for i in range(d):
        model.addConstr(gp.quicksum(flow[i, j] for j in range(d)) <= z[i],
                        name=f"{label}_capacity_{i}")
    for j in range(d):
        demand = float(instance["base"][j] + instance["sensitivity"][j] * xi[j])
        model.addConstr(gp.quicksum(flow[i, j] for i in range(d)) + emergency[j] >= demand,
                        name=f"{label}_demand_{j}")
    model.addConstr(
        gp.quicksum(float(instance["transport"][i, j]) * flow[i, j]
                    for i in range(d) for j in range(d))
        + gp.quicksum(float(instance["emergency"][j]) * emergency[j] for j in range(d)) <= eta,
        name=f"{label}_cost",
    )

def extract_ldr_policy(instance, variables):
    y, z, eta, flow0, flow_slope, emergency0, emergency_slope = variables
    d = instance["dim"]
    return {
        "y": np.array([round(y[i].X) for i in range(d)]),
        "z": np.array([z[i].X for i in range(d)]),
        "eta": float(eta.X),
        "flow0": np.array([[flow0[i, j].X for j in range(d)] for i in range(d)]),
        "flow_slope": np.array([[[flow_slope[i, j, k].X for k in range(d)]
                                  for j in range(d)] for i in range(d)]),
        "emergency0": np.array([emergency0[j].X for j in range(d)]),
        "emergency_slope": np.array([[emergency_slope[j, k].X for k in range(d)]
                                      for j in range(d)]),
    }

def worst_ldr_scenario_violation(instance, policy, scenarios, batch_size=1000):
    d = instance["dim"]
    worst = {"value": -math.inf, "index": None, "kind": None}
    for start in range(0, len(scenarios), batch_size):
        batch = np.asarray(scenarios[start:start + batch_size], dtype=float)
        flow = policy["flow0"][None, :, :] + np.einsum(
            "bk,ijk->bij", batch, policy["flow_slope"], optimize=True)
        emergency = policy["emergency0"][None, :] + batch @ policy["emergency_slope"].T
        candidates = {
            "flow_nonnegative": -flow.reshape(len(batch), -1).min(axis=1),
            "emergency_nonnegative": -emergency.min(axis=1),
            "capacity": flow.sum(axis=2).max(axis=1) - np.max(policy["z"]),
            "demand": (instance["base"][None, :] + instance["sensitivity"][None, :] * batch
                       - flow.sum(axis=1) - emergency).max(axis=1),
            "cost": (np.einsum("ij,bij->b", instance["transport"], flow, optimize=True)
                     + emergency @ instance["emergency"] - policy["eta"]),
        }
        # Capacity must be checked facility by facility, not against max(z).
        candidates["capacity"] = (flow.sum(axis=2) - policy["z"][None, :]).max(axis=1)
        for kind, values in candidates.items():
            local = int(np.argmax(values))
            if float(values[local]) > worst["value"]:
                worst = {"value": float(values[local]), "index": start + local, "kind": kind}
    return worst

def solve_scenario_ldr(instance, scenarios, reference_value, time_limit=TIME_LIMIT,
                       max_cuts=50, tolerance=1e-6):
    start = time.perf_counter()
    model, y, z, eta, flow0, flow_slope, emergency0, emergency_slope = make_ldr_model(
        instance, "scenario_full_affine_ldr", scenarios)
    variables = (y, z, eta, flow0, flow_slope, emergency0, emergency_slope)
    initial = sorted(set([0, len(scenarios) // 2, len(scenarios) - 1]))
    for number, index in enumerate(initial):
        add_ldr_scenario_constraints(model, instance, variables, scenarios[index], f"initial_{number}")
    status, policy, iterations = "max_cuts", None, 0
    for iteration in range(max_cuts + 1):
        iterations = iteration + 1
        remaining = time_limit - (time.perf_counter() - start)
        if remaining <= 0:
            status = "time_limit"
            break
        model.Params.TimeLimit = max(0.01, remaining)
        model.optimize()
        if model.SolCount == 0:
            status = f"status_{model.Status}"
            break
        policy = extract_ldr_policy(instance, variables)
        violation = worst_ldr_scenario_violation(instance, policy, scenarios)
        if violation["value"] <= tolerance:
            status = "optimal" if model.Status == GRB.OPTIMAL else "time_limit_with_solution"
            break
        add_ldr_scenario_constraints(model, instance, variables,
                                     scenarios[violation["index"]], f"cut_{iteration}")
    runtime = time.perf_counter() - start
    objective = float(model.ObjVal) if model.SolCount else math.nan
    model.dispose()
    gap = (max(0.0, (objective-reference_value)/max(1.0, abs(reference_value)))
           if math.isfinite(objective) else math.nan)
    return {"method": "ldr", "status": status, "runtime_sec": runtime,
            "objective": objective, "gap": gap, "iterations": iterations}

def solve_robust_ldr(instance, reference_value, time_limit=TIME_LIMIT):
    """Full affine LDR for ellipsoid or budget uncertainty."""
    start = time.perf_counter()
    model, y, z, eta, flow0, flow_slope, emergency0, emergency_slope = make_ldr_model(
        instance, f"{instance['uncertainty']}_full_affine_ldr")
    variables = (y, z, eta, flow0, flow_slope, emergency0, emergency_slope)
    support = add_ellipsoid_support_le if instance["uncertainty"] == "ellipsoid" else add_budget_support_le
    d = instance["dim"]
    for i in range(d):
        for j in range(d):
            support(model, -flow0[i, j], [-flow_slope[i, j, k] for k in range(d)],
                    0.0, instance, f"flow_nonnegative_{i}_{j}")
    for j in range(d):
        support(model, -emergency0[j], [-emergency_slope[j, k] for k in range(d)],
                0.0, instance, f"emergency_nonnegative_{j}")
    for i in range(d):
        support(model, gp.quicksum(flow0[i, j] for j in range(d)),
                [gp.quicksum(flow_slope[i, j, k] for j in range(d)) for k in range(d)],
                z[i], instance, f"capacity_{i}")
    for j in range(d):
        coefficients = []
        for k in range(d):
            demand_coefficient = float(instance["sensitivity"][j]) if j == k else 0.0
            coefficients.append(demand_coefficient
                                - gp.quicksum(flow_slope[i, j, k] for i in range(d))
                                - emergency_slope[j, k])
        support(model, float(instance["base"][j])
                - gp.quicksum(flow0[i, j] for i in range(d)) - emergency0[j],
                coefficients, 0.0, instance, f"demand_{j}")
    cost_constant = (gp.quicksum(float(instance["transport"][i, j]) * flow0[i, j]
                                 for i in range(d) for j in range(d))
                     + gp.quicksum(float(instance["emergency"][j]) * emergency0[j]
                                   for j in range(d)))
    cost_coefficients = [
        gp.quicksum(float(instance["transport"][i, j]) * flow_slope[i, j, k]
                    for i in range(d) for j in range(d))
        + gp.quicksum(float(instance["emergency"][j]) * emergency_slope[j, k]
                      for j in range(d)) for k in range(d)]
    support(model, cost_constant, cost_coefficients, eta, instance, "recourse_cost")
    model.Params.TimeLimit = float(time_limit)
    model.optimize()
    runtime = time.perf_counter() - start
    objective = float(model.ObjVal) if model.SolCount else math.nan
    status = ("optimal" if model.Status == GRB.OPTIMAL else "time_limit_with_solution"
              if model.Status == GRB.TIME_LIMIT and model.SolCount else f"status_{model.Status}")
    model.dispose()
    gap = (max(0.0, (objective-reference_value)/max(1.0, abs(reference_value)))
           if math.isfinite(objective) else math.nan)
    return {"method": "ldr", "status": status, "runtime_sec": runtime,
            "objective": objective, "gap": gap, "iterations": 1}

def solve_ldr(instance, reference_value, scenarios=None, time_limit=None):
    limit = TIME_LIMIT if time_limit is None else float(time_limit)
    if instance["uncertainty"] == "scenario":
        return solve_scenario_ldr(
            instance, scenarios, reference_value, time_limit=limit)
    return solve_robust_ldr(instance, reference_value, time_limit=limit)

