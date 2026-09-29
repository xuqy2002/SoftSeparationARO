"""C&CG and LDR benchmark implementations for Section 5.1.2.
"""

import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np




class EllipsoidKKTSeparation:
    def __init__(self, runtime, inst):
        self.runtime = runtime
        self.inst = inst
        d = inst["dim"]
        self.model = runtime.new_model("ellipsoid_kkt_separation")
        if np.any(np.asarray(inst["transport"]) <= 0) or inst["emergency"] <= 0:
            raise ValueError("KKT separation requires positive recourse costs")
        u_upper = np.minimum(
            float(inst["emergency"]),
            np.maximum(
                0.0,
                float(inst["emergency"])-np.min(inst["transport"], axis=1),
            ),
        )
        self.u = self.model.addVars(
            d, lb=0, ub={i: float(u_upper[i]) for i in range(d)}, name="u")
        self.v = self.model.addVars(
            d, lb=0, ub=inst["emergency"], name="v")
        widths = runtime.ellipsoid_coordinate_radii(inst)
        g_lower = np.asarray(inst["center"])-widths
        g_upper = np.asarray(inst["center"])+widths
        self.g = self.model.addVars(
            d,
            lb={j: float(g_lower[j]) for j in range(d)},
            ub={j: float(g_upper[j]) for j in range(d)},
            name="xi",
        )
        demand_upper = np.maximum(
            np.asarray(inst["base"])+np.asarray(inst["sensitivity"])*g_lower,
            np.asarray(inst["base"])+np.asarray(inst["sensitivity"])*g_upper,
        )
        self.demand_upper = np.maximum(0.0, demand_upper)
        self.r = self.model.addVars(d, d, lb=0, name="shipment")
        self.e = self.model.addVars(
            d,
            lb=0,
            ub={j: float(self.demand_upper[j]) for j in range(d)},
            name="emergency_supply",
        )
        active = self.model.addVars(d, d, vtype=GRB.BINARY, name="shipment_active")
        emergency_active = self.model.addVars(
            d, vtype=GRB.BINARY, name="emergency_active")
        capacity_active = self.model.addVars(
            d, vtype=GRB.BINARY, name="capacity_active")
        slack = self.model.addVars(d, lb=0, name="capacity_slack")
        self.capacity_balance = {}
        self.capacity_complement = {}
        self.flow_gate = {}
        for j in range(d):
            self.model.addConstr(
                gp.quicksum(self.r[i, j] for i in range(d))+self.e[j]
                == float(inst["base"][j])+float(inst["sensitivity"][j])*self.g[j]
            )
            self.model.addConstr(
                self.e[j] <= float(self.demand_upper[j])*emergency_active[j])
            self.model.addConstr(
                float(inst["emergency"])-self.v[j]
                <= float(inst["emergency"])*(1-emergency_active[j]))
        for i in range(d):
            self.capacity_balance[i] = self.model.addConstr(
                gp.quicksum(self.r[i, j] for j in range(d))+slack[i] == 0)
            self.model.addConstr(
                self.u[i] <= float(u_upper[i])*capacity_active[i])
            self.capacity_complement[i] = self.model.addConstr(
                slack[i]+capacity_active[i] <= 0)
            for j in range(d):
                self.flow_gate[i, j] = self.model.addConstr(
                    self.r[i, j]-active[i, j] <= 0)
                reduced_cost = float(inst["transport"][i, j])-self.v[j]+self.u[i]
                self.model.addConstr(
                    reduced_cost
                    <= (float(inst["transport"][i, j])+float(u_upper[i]))
                    *(1-active[i, j]))
        self.active = active
        self.capacity_active = capacity_active
        self.model.setObjective(
            gp.quicksum(
                float(inst["transport"][i, j])*self.r[i, j]
                for i in range(d) for j in range(d))
            + float(inst["emergency"])*gp.quicksum(self.e.values()),
            GRB.MAXIMIZE,
        )
        for i in range(d):
            for j in range(d):
                self.model.addConstr(
                    self.v[j]-self.u[i] <= inst["transport"][i, j])
        unit = self.model.addVars(d, lb=-1, ub=1, name="ellipsoid_unit")
        for j in range(d):
            self.model.addConstr(
                self.g[j] == float(inst["center"][j])
                + gp.quicksum(
                    float(inst["transform"][j, k])*unit[k] for k in range(d)))
        self.model.addQConstr(
            gp.quicksum(unit[k]*unit[k] for k in range(d)) <= 1)
        self.model.update()

    def solve(self, x, deadline=math.inf):
        inst, d = self.inst, self.inst["dim"]
        capacities = np.asarray(x[d:], dtype=float)
        if (capacities.shape != (d,)
                or not np.all(np.isfinite(capacities))
                or np.any(capacities < 0)):
            raise ValueError("KKT separation requires finite nonnegative capacities")
        for i in range(d):
            zi = float(capacities[i])
            self.capacity_balance[i].RHS = zi
            self.capacity_complement[i].RHS = zi
            self.model.chgCoeff(
                self.capacity_complement[i], self.capacity_active[i], zi)
            for j in range(d):
                bound = min(zi, float(self.demand_upper[j]))
                self.r[i, j].UB = bound
                self.model.chgCoeff(
                    self.flow_gate[i, j], self.active[i, j], -bound)
        self.model.update()
        self.model.reset(1)
        self.runtime.optimize_before(self.model, deadline)
        value, upper = self.runtime.solver_bounds(self.model)
        xi = (np.array([self.g[j].X for j in range(d)])
              if self.model.SolCount else None)
        return dict(
            value=value,
            upper=upper,
            xi=xi,
            solver_status=int(self.model.Status),
            complete=self.model.Status == GRB.OPTIMAL,
            evaluation_method="kkt_objval",
        )

    def close(self):
        self.model.dispose()


class FiniteSeparation:
    """Enumerate recourse LPs."""

    def __init__(self, runtime, inst, scenarios):
        self.runtime = runtime
        self.inst = inst
        self.scenarios = np.asarray(scenarios, dtype=float)
        if (self.scenarios.ndim != 2
                or self.scenarios.shape[1] != inst["dim"]
                or len(self.scenarios) == 0
                or not np.all(np.isfinite(self.scenarios))):
            raise ValueError("Enumeration requires a nonempty finite scenario matrix")
        self.oracle = runtime.Recourse(inst)
        self.model = self.oracle.model

    def solve(self, x, deadline=math.inf):
        worst, witness, count = -math.inf, None, 0
        self.model.reset(1)
        try:
            for point in self.scenarios:
                value, _ = self.oracle.solve(x, point, deadline)
                if not self.runtime.finite_number(value):
                    raise RuntimeError("Nonfinite recourse value during enumeration")
                count += 1
                if value > worst:
                    worst, witness = value, point.copy()
        except self.runtime.BudgetExpired:
            return dict(
                value=worst if count else math.nan,
                upper=math.inf,
                xi=witness,
                solver_status=GRB.TIME_LIMIT,
                complete=False,
                evaluated_count=count,
                scenario_count=len(self.scenarios),
                evaluation_method="enumeration_incomplete",
            )
        return dict(
            value=worst,
            upper=worst,
            xi=witness,
            solver_status=GRB.OPTIMAL,
            complete=True,
            evaluated_count=count,
            scenario_count=len(self.scenarios),
            evaluation_method="enumeration",
        )

    def close(self):
        self.oracle.close()


class GabrelBudgetDualMILPSeparation:
    """Exact budget vertex/dual MILP using Gabrel LP-(M) bounds."""

    def __init__(self, runtime, inst):
        self.runtime = runtime
        self.inst = inst
        self.model = None
        self.lp_m = None
        d, emergency = inst["dim"], float(inst["emergency"])
        if (np.any(inst["base"] < 0)
                or np.any(inst["sensitivity"] < 0)
                or np.any(inst["transport"] <= 0)
                or emergency <= 0):
            raise ValueError(
                "Budget dual MILP requires nonnegative demands and positive costs")
        _, whole, fraction = runtime.budget_parts(d, inst["gamma"])
        self.fraction = fraction
        self.analytic_u = np.maximum(
            0.0, emergency-np.min(inst["transport"], axis=1))
        try:
            self.model = runtime.new_model("budget_dual_milp_separation")
            self.u = self.model.addVars(
                d,
                lb=0,
                ub={i: float(self.analytic_u[i]) for i in range(d)},
                name="u",
            )
            self.v = self.model.addVars(d, lb=0, ub=emergency, name="v")
            self.g = self.model.addVars(d, vtype=GRB.BINARY, name="xi_one")
            self.product = self.model.addVars(
                d, lb=0, ub=emergency, name="v_xi_one")
            self.product_gate = {}
            self.product_lower = {}
            for i in range(d):
                for j in range(d):
                    self.model.addConstr(
                        self.v[j]-self.u[i] <= float(inst["transport"][i, j]))
            self.fractional_g = None
            self.fractional_product = None
            self.fractional_gate = {}
            self.fractional_lower = {}
            if fraction > 0:
                self.fractional_g = self.model.addVars(
                    d, vtype=GRB.BINARY, name="xi_fraction")
                self.fractional_product = self.model.addVars(
                    d, lb=0, ub=emergency, name="v_xi_fraction")
                self.model.addConstr(gp.quicksum(self.g.values()) == whole)
                self.model.addConstr(
                    gp.quicksum(self.fractional_g.values()) == 1)
                for j in range(d):
                    self.model.addConstr(self.g[j]+self.fractional_g[j] <= 1)
            else:
                self.model.addConstr(
                    gp.quicksum(self.g.values()) <= whole, name="budget")
            products = [
                (self.g, self.product, self.product_gate, self.product_lower)]
            if fraction > 0:
                products.append((
                    self.fractional_g,
                    self.fractional_product,
                    self.fractional_gate,
                    self.fractional_lower,
                ))
            for binary, product, gate, lower in products:
                for j in range(d):
                    self.model.addConstr(product[j] <= self.v[j])
                    gate[j] = self.model.addConstr(
                        product[j]-emergency*binary[j] <= 0)
                    lower[j] = self.model.addConstr(
                        product[j]-self.v[j]-emergency*binary[j] >= -emergency)
            self.model.update()

            self.lp_m = runtime.new_model("gabrel_LP_M_full_demand")
            self.lp_m.Params.Method = 1
            self.lp_m.Params.FeasibilityTol = min(runtime.CFG.solver_tol, 1e-8)
            self.lp_m.Params.OptimalityTol = min(runtime.CFG.solver_tol, 1e-8)
            self.lp_u = self.lp_m.addVars(
                d,
                lb=0,
                ub={i: float(self.analytic_u[i]) for i in range(d)},
                name="u",
            )
            self.lp_v = self.lp_m.addVars(d, lb=0, ub=emergency, name="v")
            for i in range(d):
                for j in range(d):
                    self.lp_m.addConstr(
                        self.lp_v[j]-self.lp_u[i]
                        <= float(inst["transport"][i, j]))
            self.lp_m.update()
            self.gabrel_lp_calls = 0
            self.gabrel_lp_total_sec = 0.0
            self.last_gabrel = {}
        except Exception:
            self.close()
            raise

    def solve_lp_m_gabrel_vmax(self, capacities, deadline):
        start = time.perf_counter()
        d, inst = self.inst["dim"], self.inst
        try:
            full = inst["base"]+inst["sensitivity"]
            self.lp_m.setObjective(
                gp.quicksum(
                    float(full[j])*self.lp_v[j] for j in range(d))
                - gp.quicksum(
                    float(capacities[i])*self.lp_u[i] for i in range(d)),
                GRB.MAXIMIZE,
            )
            self.lp_m.update()
            self.lp_m.reset(1)
            self.gabrel_lp_calls += 1
            self.runtime.optimize_before(self.lp_m, deadline)
            if self.lp_m.Status == GRB.TIME_LIMIT:
                raise self.runtime.BudgetExpired(
                    "Gabrel LP-(M) timed out; no bounds installed")
            if self.lp_m.Status != GRB.OPTIMAL:
                raise RuntimeError(
                    "Gabrel LP-(M) not optimal: "+str(self.lp_m.Status))
            v = np.array([self.lp_v[j].X for j in range(d)])
            u = np.array([self.lp_u[i].X for i in range(d)])
            padding = max(1e-7, 10*float(self.lp_m.Params.FeasibilityTol))
            if (not np.all(np.isfinite(v))
                    or not np.all(np.isfinite(u))
                    or np.min(v) < -padding
                    or np.min(u) < -padding
                    or np.max(v) > inst["emergency"]+padding
                    or np.max(
                        v[None, :]-u[:, None]-inst["transport"]) > padding):
                raise RuntimeError("Gabrel LP-(M) returned an invalid dual solution")
            v_upper = np.minimum(
                float(inst["emergency"]), np.maximum(0.0, v)+padding)
            u_upper = np.minimum(
                self.analytic_u,
                np.maximum(
                    0.0,
                    np.max(v_upper[None, :]-inst["transport"], axis=1),
                )+padding,
            )
            return dict(
                v_upper=v_upper,
                u_upper=u_upper,
                v_raw=v,
                objective=float(self.lp_m.ObjVal),
                padding=padding,
            )
        finally:
            self.gabrel_lp_total_sec += time.perf_counter()-start

    def solve(self, x, deadline=math.inf):
        d, inst = self.inst["dim"], self.inst
        x = np.asarray(x, dtype=float)
        if (x.shape != (2*d,)
                or not np.all(np.isfinite(x))
                or np.any(x[d:] < 0)):
            raise ValueError(
                "Budget dual separation requires a finite x and nonnegative capacities")
        before = self.gabrel_lp_total_sec
        bounds = self.solve_lp_m_gabrel_vmax(x[d:], deadline)
        products = [
            (self.g, self.product, self.product_gate, self.product_lower)]
        if self.fraction > 0:
            products.append((
                self.fractional_g,
                self.fractional_product,
                self.fractional_gate,
                self.fractional_lower,
            ))
        for j in range(d):
            mj = float(bounds["v_upper"][j])
            self.v[j].UB = mj
            for binary, product, gate, lower in products:
                product[j].UB = mj
                self.model.chgCoeff(gate[j], binary[j], -mj)
                self.model.chgCoeff(lower[j], binary[j], -mj)
                lower[j].RHS = -mj
        for i in range(d):
            self.u[i].UB = float(bounds["u_upper"][i])
        objective = (
            gp.quicksum(
                float(inst["base"][j])*self.v[j]
                + float(inst["sensitivity"][j])*self.product[j]
                for j in range(d))
            - gp.quicksum(float(x[d+i])*self.u[i] for i in range(d))
        )
        if self.fraction > 0:
            objective += gp.quicksum(
                self.fraction*float(inst["sensitivity"][j])
                * self.fractional_product[j] for j in range(d))
        self.model.setObjective(objective, GRB.MAXIMIZE)
        self.model.update()
        self.model.reset(1)
        self.runtime.optimize_before(self.model, deadline)
        value, upper = self.runtime.solver_bounds(self.model)
        xi = None
        if self.model.SolCount:
            xi = np.array([self.g[j].X for j in range(d)])
            if self.fraction > 0:
                xi += self.fraction*np.array(
                    [self.fractional_g[j].X for j in range(d)])
        self.last_gabrel = dict(
            big_m_source="gabrel_lp_m",
            gabrel_lp_status=int(self.lp_m.Status),
            gabrel_lp_sec=self.gabrel_lp_total_sec-before,
            gabrel_lp_total_sec=self.gabrel_lp_total_sec,
            gabrel_lp_calls=self.gabrel_lp_calls,
            gabrel_lp_objective=bounds["objective"],
            gabrel_padding=bounds["padding"],
            gabrel_dimension=d,
        )
        return dict(
            value=value,
            upper=upper,
            xi=xi,
            solver_status=int(self.model.Status),
            complete=self.model.Status == GRB.OPTIMAL,
            evaluation_method="budget_dual_milp_objval",
            **self.last_gabrel,
        )

    def close(self):
        if self.lp_m is not None:
            self.lp_m.dispose()
            self.lp_m = None
        if self.model is not None:
            self.model.dispose()
            self.model = None


def separation(runtime, inst, scenarios=None):
    if scenarios is not None:
        return FiniteSeparation(runtime, inst, scenarios)
    if inst.get("uncertainty") == "budget":
        return GabrelBudgetDualMILPSeparation(runtime, inst)
    return EllipsoidKKTSeparation(runtime, inst)


def first_stage_model(runtime, inst, name):
    model = runtime.new_model(name)
    d = inst["dim"]
    x = model.addVars(2*d, lb=0, name="first_stage")
    for i in range(d):
        x[i].UB = inst["y_upper"]
        model.addConstr(x[d+i] <= inst["slope"]*x[i])
    eta = model.addVar(lb=0, name="eta")
    model.setObjective(
        gp.quicksum(float(inst["c"][i])*x[i] for i in range(2*d))+eta,
        GRB.MINIMIZE,
    )
    return model, x, eta


def add_scenario(runtime, model, x, eta, inst, xi, label):
    d = inst["dim"]
    recourse = model.addVars(d+1, d, lb=0, name="r_"+str(label))
    for i in range(d):
        model.addConstr(
            gp.quicksum(recourse[i, j] for j in range(d)) <= x[d+i])
    demands = runtime.demand(inst, xi)
    for j in range(d):
        model.addConstr(
            gp.quicksum(recourse[i, j] for i in range(d+1))
            >= float(demands[j]))
    model.addConstr(
        eta >= gp.quicksum(
            float(inst["transport"][i, j])*recourse[i, j]
            for i in range(d) for j in range(d))
        + inst["emergency"]*gp.quicksum(recourse[d, j] for j in range(d)))


def solve_ccg(runtime, inst, scenarios=None, target=None, time_limit=None,
              max_iter=None, verbose=False, reference=None):
    target = runtime.CFG.gap if target is None else target
    max_iter = runtime.CFG.ccg_max_iter if max_iter is None else max_iter
    reference_value = reference.get("objective") if reference is not None else None
    if reference is not None and not runtime.finite_number(reference_value):
        return dict(status="reference_unavailable", runtime_sec=math.nan, gap=math.nan)

    def ccg_gap(lower, upper):
        if reference is None:
            return runtime.bound_gap(lower, upper)
        if not runtime.finite_number(upper):
            return math.inf
        return max(
            0.0,
            (upper-reference_value)/max(1.0, abs(reference_value)),
        )

    start = time.perf_counter()
    deadline = runtime.deadline_after(time_limit)
    master = separator = None
    lower, upper = -math.inf, math.inf
    best_x = None
    status, iteration = "max_iter", 0
    history = []
    try:
        master, x, eta = first_stage_model(runtime, inst, "ccg_master")
        add_scenario(
            runtime,
            master,
            x,
            eta,
            inst,
            inst["center"] if scenarios is None else scenarios[0],
            0,
        )
        separator = separation(runtime, inst, scenarios)
        for iteration in range(1, max_iter+1):
            runtime.optimize_before(master, deadline)
            _, master_lower = runtime.solver_bounds(master)
            if runtime.finite_number(master_lower):
                lower = max(lower, master_lower)
            if not master.SolCount:
                status = "master_no_solution"
                break
            candidate = np.array(
                [x[i].X for i in range(2*inst["dim"])])
            separated = separator.solve(candidate, deadline)
            if runtime.finite_number(separated["upper"]):
                candidate_upper = float(inst["c"]@candidate)+separated["upper"]
                if candidate_upper < upper:
                    upper, best_x = candidate_upper, candidate.copy()
            history.append(dict(
                iteration=iteration,
                runtime_sec=time.perf_counter()-start,
                lower=lower,
                upper=upper,
            ))
            if verbose:
                runtime.print_progress(
                    "C&CG", iteration, time.perf_counter()-start,
                    ccg_gap(lower, upper))
            if ccg_gap(lower, upper) <= target:
                status = "reached"
                break
            if separated["xi"] is None:
                status = "separation_no_solution"
                break
            if runtime.remaining(deadline) <= 0:
                raise runtime.BudgetExpired()
            add_scenario(
                runtime,
                master,
                x,
                eta,
                inst,
                separated["xi"],
                iteration,
            )
    except runtime.BudgetExpired:
        status = "time_limit"
    finally:
        if separator is not None:
            separator.close()
        if master is not None:
            master.dispose()
    elapsed = time.perf_counter()-start
    if time_limit is not None and elapsed > time_limit and status == "reached":
        status = "time_limit"
    return dict(
        status=status,
        runtime_sec=elapsed,
        wall_time_sec=elapsed,
        excluded_check_time_sec=0.0,
        lower=lower,
        upper=upper,
        objective=upper,
        gap=ccg_gap(lower, upper),
        iterations=iteration,
        x=best_x,
        solver_gap=runtime.bound_gap(lower, upper),
        reference_objective=reference_value,
        ccg_history=history,
        instance_fingerprint=runtime.reference_key(inst, scenarios),
        ccg_formulation=(
            "budget_dual_milp_gabrel_lpm_v3"
            if inst.get("uncertainty") == "budget"
            else "kkt_transport_v1" if scenarios is None
            else "enumeration_lp_v1"),
    )


def extensive_reference(runtime, inst, scenarios):
    start = time.perf_counter()
    deadline = runtime.deadline_after(runtime.CFG.reference_time_limit)
    model = None
    try:
        model, x, eta = first_stage_model(runtime, inst, "finite_reference")
        for index, xi in enumerate(scenarios):
            if runtime.remaining(deadline) <= 0:
                raise runtime.BudgetExpired()
            add_scenario(runtime, model, x, eta, inst, xi, index)
        runtime.optimize_before(model, deadline)
        upper, lower = runtime.solver_bounds(model)
        status = (
            "reached"
            if runtime.bound_gap(lower, upper) <= runtime.CFG.reference_gap
            else "reference_unavailable")
        return dict(
            status=status,
            lower=lower,
            upper=upper,
            objective=upper,
            gap=runtime.bound_gap(lower, upper),
            runtime_sec=time.perf_counter()-start,
            reference_mode="finite_extensive",
        )
    except runtime.BudgetExpired:
        return dict(
            status="reference_unavailable",
            lower=math.nan,
            upper=math.nan,
            runtime_sec=time.perf_counter()-start,
            reference_mode="finite_extensive",
        )
    finally:
        if model is not None:
            model.dispose()


def add_soc_le(runtime, model, intercept, coefficients, rhs, inst):
    if inst.get("uncertainty") == "budget":
        lam = model.addVar(lb=0, name="budget_support_lambda")
        mu = model.addVars(len(coefficients), lb=0, name="budget_support_mu")
        for j, coefficient in enumerate(coefficients):
            model.addConstr(lam+mu[j] >= coefficient)
        model.addConstr(
            intercept+float(inst["budget"])*lam+gp.quicksum(mu.values()) <= rhs)
        return

    scaled_coeff = model.addVars(len(coefficients), lb=-GRB.INFINITY)
    radius = model.addVar(lb=0)
    for k in range(len(coefficients)):
        model.addConstr(
            scaled_coeff[k] == gp.quicksum(
                float(inst["transform"][j, k])*coefficient
                for j, coefficient in enumerate(coefficients)))
    model.addGenConstrNorm(radius, list(scaled_coeff.values()), 2.0)
    center_expr = intercept+gp.quicksum(
        float(inst["center"][j])*coefficient
        for j, coefficient in enumerate(coefficients))
    model.addConstr(center_expr+radius <= rhs)


def _evaluation_details(runtime, result):
    details = runtime.check_details(result)
    details.update({
        key: value for key, value in result.items()
        if key == "big_m_source" or key.startswith("gabrel_")
    })
    return details


def solve_ldr(runtime, inst, scenarios, reference):
    start = time.perf_counter()
    deadline = runtime.deadline_after(runtime.CFG.time_limit)
    model = None
    lower, upper = math.nan, math.nan
    status = "time_limit"
    x_value = None
    d = inst["dim"]
    q = d*d+d
    try:
        model, x, eta = first_stage_model(runtime, inst, "full_affine_ldr")
        model.Params.Threads = 1
        model.Params.FeasibilityTol = 1e-6
        model.Params.OptimalityTol = 1e-6
        model.Params.BarConvTol = 1e-6
        model.Params.BarQCPConvTol = 1e-6
        r0 = model.addVars(q, lb=-GRB.INFINITY, name="intercept")
        affine = model.addVars(q, d, lb=-GRB.INFINITY, name="affine")
        cost = np.r_[
            inst["transport"].ravel(),
            np.full(d, inst["emergency"]),
        ]
        if scenarios is None:
            for k in range(q):
                if runtime.remaining(deadline) <= 0:
                    raise runtime.BudgetExpired()
                add_soc_le(
                    runtime,
                    model,
                    -r0[k],
                    [-affine[k, j] for j in range(d)],
                    0,
                    inst,
                )
            for i in range(d):
                if runtime.remaining(deadline) <= 0:
                    raise runtime.BudgetExpired()
                add_soc_le(
                    runtime,
                    model,
                    gp.quicksum(r0[i*d+j] for j in range(d)),
                    [gp.quicksum(affine[i*d+j, k] for j in range(d))
                     for k in range(d)],
                    x[d+i],
                    inst,
                )
            for j in range(d):
                ids = [i*d+j for i in range(d+1)]
                base = float(inst["base"][j])-gp.quicksum(r0[k] for k in ids)
                coefficients = [
                    (float(inst["sensitivity"][j]) if j == coordinate else 0)
                    - gp.quicksum(affine[k, coordinate] for k in ids)
                    for coordinate in range(d)
                ]
                add_soc_le(runtime, model, base, coefficients, 0, inst)
            emergency_ids = range(d*d, q)
            add_soc_le(
                runtime,
                model,
                gp.quicksum(r0[k] for k in emergency_ids),
                [gp.quicksum(affine[k, j] for k in emergency_ids)
                 for j in range(d)],
                float(np.sum(inst["base"]+inst["sensitivity"])),
                inst,
            )
            add_soc_le(
                runtime,
                model,
                gp.quicksum(float(cost[k])*r0[k] for k in range(q)),
                [gp.quicksum(float(cost[k])*affine[k, j] for k in range(q))
                 for j in range(d)],
                eta,
                inst,
            )
        else:
            for xi in scenarios:
                if runtime.remaining(deadline) <= 0:
                    raise runtime.BudgetExpired()
                expression = [
                    r0[k]+gp.quicksum(
                        float(xi[j])*affine[k, j] for j in range(d))
                    for k in range(q)
                ]
                for value in expression:
                    model.addConstr(value >= 0)
                for i in range(d):
                    model.addConstr(
                        gp.quicksum(expression[i*d+j] for j in range(d))
                        <= x[d+i])
                demands = runtime.demand(inst, xi)
                for j in range(d):
                    model.addConstr(
                        gp.quicksum(expression[i*d+j] for i in range(d+1))
                        >= float(demands[j]))
                model.addConstr(
                    gp.quicksum(expression[k] for k in range(d*d, q))
                    <= float(np.sum(inst["base"]+inst["sensitivity"])))
                model.addConstr(
                    gp.quicksum(float(cost[k])*expression[k] for k in range(q))
                    <= eta)
        runtime.optimize_before(model, deadline)
        upper, lower = runtime.solver_bounds(model)
        status = (
            "optimal" if model.Status == GRB.OPTIMAL
            else "time_limit" if model.Status == GRB.TIME_LIMIT
            else f"solver_{model.Status}")
        if model.SolCount:
            x_value = np.array([x[i].X for i in range(2*d)])
    except runtime.BudgetExpired:
        status = "time_limit"
    finally:
        if model is not None:
            model.dispose()

    elapsed = time.perf_counter()-start
    ref_lower = reference.get("lower")
    policy_objective = upper
    robust_objective, evaluation_upper = math.nan, math.nan
    evaluation_status, evaluation_solver_gap = "no_ldr_solution", math.nan
    evaluation_start = time.perf_counter()
    evaluation_deadline = evaluation_start+300.0
    evaluator = None
    evaluation_details = {}
    evaluation_method = "not_checked"
    if x_value is not None:
        try:
            evaluator = separation(runtime, inst, scenarios)
            evaluator.model.Params.FeasibilityTol = 1e-6
            evaluator.model.Params.OptimalityTol = 1e-6
            evaluator.model.Params.MIPGap = 1e-6
            result = runtime.checked_separation(
                evaluator, inst, scenarios, x_value, evaluation_deadline)
            evaluation_method = result["evaluation_method"]
            evaluation_details = _evaluation_details(runtime, result)
            evaluation_status = str(result["solver_status"])
            evaluation_solver_gap = runtime.bound_gap(
                result["value"], result["upper"])
            if evaluation_method == "enumeration_incomplete":
                evaluation_status = "enumeration_incomplete"
            elif runtime.finite_number(result["value"]):
                robust_objective = float(inst["c"]@x_value)+result["value"]
                if runtime.finite_number(result["upper"]):
                    evaluation_upper = float(inst["c"]@x_value)+result["upper"]
            else:
                evaluation_status = "no_solution_"+evaluation_status
        except gp.GurobiError as exc:
            evaluation_status = "error_"+str(exc)
        finally:
            if evaluator is not None:
                evaluator.close()
    evaluation_sec = time.perf_counter()-evaluation_start
    ref_objective = reference.get("objective")
    approximation_gap = (
        max(
            0.0,
            (robust_objective-ref_objective)/max(1.0, abs(ref_objective)),
        )
        if (runtime.finite_number(robust_objective)
            and runtime.finite_number(ref_objective))
        else math.nan)
    approx_upper = runtime.bound_gap(ref_lower, evaluation_upper)
    result = dict(
        status=status,
        runtime_sec=elapsed,
        wall_time_sec=elapsed,
        excluded_check_time_sec=evaluation_sec,
        objective=robust_objective,
        evaluation_method=evaluation_method,
        evaluation_details=evaluation_details,
        certification_scope=runtime.check_scope(evaluation_method),
        heuristic_reached=(
            evaluation_method == "sampled_10000"
            and approximation_gap <= runtime.CFG.gap),
        ldr_policy_objective=policy_objective,
        robust_evaluation_status=evaluation_status,
        robust_evaluation_solver_gap=evaluation_solver_gap,
        robust_evaluation_sec=evaluation_sec,
        solver_gap=runtime.bound_gap(lower, upper),
        gap=approximation_gap,
        approximation_gap=approximation_gap,
        reference_objective=ref_objective,
        gap_reference="reference.objective",
        ldr_threads=1,
        ldr_tolerance=1e-6,
        ldr_coordinates="raw_xi",
        ldr_cone_formulation=(
            "budget_support_dual_lp"
            if inst.get("uncertainty") == "budget"
            else "GenConstrNorm"),
        approximation_certified=(
            evaluation_method not in ("sampled_10000", "enumeration_incomplete")
            and reference.get("status") == "reached"
            and approx_upper <= runtime.CFG.gap),
    )
    if inst.get("uncertainty") == "budget":
        result.update(
            ldr_uncertainty="budget",
            ldr_coordinates="raw_xi",
        )
    runtime.print_progress("LDR", 1, elapsed, approximation_gap)
    return result


def budget_ccg_timing(reference, target, time_limit):
    """Return the first recorded C&CG upper-bound hit."""
    baseline = reference.get("objective")
    if baseline is None or not math.isfinite(baseline):
        return dict(
            status="reference_unavailable", runtime_sec=math.nan, gap=math.nan)
    history = sorted(
        reference.get("ccg_history") or [],
        key=lambda row: row.get("runtime_sec", math.inf),
    )
    for row in history:
        upper, elapsed = row.get("upper"), row.get("runtime_sec")
        if (upper is None
                or elapsed is None
                or not math.isfinite(upper)
                or not math.isfinite(elapsed)
                or elapsed < 0):
            continue
        gap = max(0.0, (upper-baseline)/max(1.0, abs(baseline)))
        if gap <= target:
            return dict(
                status=(
                    "reached"
                    if time_limit is None or elapsed <= time_limit
                    else "time_limit"),
                runtime_sec=float(elapsed),
                gap=gap,
                objective=upper,
                lower=row.get("lower"),
                upper=upper,
                iterations=row.get("iteration"),
                reference_objective=baseline,
                ccg_reused=True,
                ccg_timing_resolution="iteration_history",
            )
    return dict(
        status="reuse_timing_unavailable",
        runtime_sec=math.nan,
        gap=math.nan,
        reference_objective=baseline,
        ccg_reused=True,
        ccg_timing_resolution="unavailable",
    )
