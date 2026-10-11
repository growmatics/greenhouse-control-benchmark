"""Physics-only economic MPC on the GreenLight model (benchmark arm 3).

Direct-collocation nonlinear MPC on a **state-scaled** GreenLight ODE (scaling is
what makes the stiff 28-state model solvable in an optimizer). This module is
deliberately **isolated**: it depends only on the ODE from ``gl_gym`` (imported
lazily) and is validated on its own (unit tests here, ``examples/validate_mpc.py``
against the published config) BEFORE it is registered as a benchmark arm.

Inverse-crime boundary (contract rows 15-19): it holds only a nominal model,
never the simulator's ground-truth parameters (enforced by ``PhysicsController``).

Status: WIP (validation layer 4). The stable-solver recipe (L-BFGS Hessian, relaxed
IPOPT tolerances, L1 soft-bound penalties, and a 10% input-rate constraint on the
state-scaled collocation model) follows van Laatum et al. [vanlaatum2026rlmpc].
Cold-start hardening (a warm control history, a safe hold-last fallback, and
strictly-positive CO2/vapour/biomass box bounds that avoid the ODE's
photosynthesis/LAI singularities at zero) and a solve-speed pass (a build-once
parametric Opti NLP [andersson2019casadi], control warm-start, capped iterations with
acceptable-point termination [wachter2006ipopt], a salvaged last iterate, an MX->SX
``expand``, and Radau degree 2, benchmarked against degree 3 for <1% plant fidelity)
are in place: per-step solve went from 2-25 s (unbounded tail) to ~1 s median with a
~1.3 s tail and no hold-fallbacks.

The full recipe, the two due-diligence reads, and the source-by-source provenance are
in ``../method/mpc-solver-notes.md``; every source is in the shared ``references.bib``.
Remaining: L5 (reproduce the published numbers). Validated (L1-L5) BEFORE
registration as arm 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from greenlight_bench.config import EconomicConfig
from greenlight_bench.controllers.base import (
    ControlContext,
    NominalParameters,
    PhysicsController,
)
from greenlight_bench.models.outputs import outputs_from_state

# 28-state scale: temps/soil ~1e1, CO2/vapour ~1e3, biomass ~1e5, tCanSum ~1e3, time 1.
STATE_SCALE: NDArray[np.float64] = np.array(
    [1e3, 1e3] + [1e1] * 13 + [1e3, 1e3] + [1e1] * 5 + [1e5, 1e5, 1e5, 1e5, 1e3, 1e0],
    dtype=float,
)

# Fallback economic prices (mirror benchmark.yaml economic block / GLG2 GreenhouseReward).
# The live benchmark value comes from the economic config (benchmark.yaml); see __init__.
DEFAULT_PRICES = {"fruit": 1.6, "heat": 0.09, "elec": 0.3, "co2": 0.3}

# Warm control history (heating + CO2 on): a cold season start needs heating, and
# the rate limit is applied relative to this so the MPC does not start from u=0.
U_INIT: NDArray[np.float64] = np.array([0.5, 0.3, 0.0, 0.0, 0.0, 0.0])

# State indices used by the economic objective / soft bounds.
_I_CO2, _I_TAIR, _I_CFRUIT, _I_TIME = 0, 2, 25, 27
_TEMP_IDX = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21)

# Tolerance for accepting a salvaged (non-clean) IPOPT iterate's controls as in-bounds.
_U_TOL = 1e-6

# Characteristic scales that normalise the economic penalty's climate outputs to O(1)
# (raw CO2 ppm ~1000, RH ~100 %, air temp ~10 degC) so the soft-bound constraints do
# not wreck the KKT conditioning. Penalty value is preserved (scale folds into weight).
_CO2_SCALE, _TEMP_SCALE, _RH_SCALE = 1000.0, 10.0, 100.0

# Van Laatum et al. [vanlaatum2026rlmpc] penalise RH from 2 %-points below its ceiling
# (their RL-SMPC code: P >= w * (rh - (rh_max - 2))) -- a small margin that keeps the MPC
# off the humidity bound. Applied to the RH upper bound only.
_RH_UPPER_MARGIN = 2.0

# Scaled-state indices the GreenLight ODE DIVIDES by: CO2 air/top (0, 1), vapour
# pressure air/top (15, 16), and live leaf/stem/fruit biomass (23, 24, 25). IPOPT
# evaluates the model at infeasible trial points during line search and restoration
# (it has no "stay in the domain" mode), so a bound alone does not prevent the ODE from
# seeing a non-positive divisor and returning NaN -- which is fatal. We therefore extend
# the vector field to be FINITE and smooth everywhere by flooring these states with
# sqrt(z^2 + eps^2) before the ODE: a differentiable surrogate outside the physical box
# that equals the state to O(eps^2) inside it (eps << the box floors, so the feasible
# region is unchanged). Standard interior-point practice for domain-restricted models.
_SINGULAR_IDX = (0, 1, 15, 16, 23, 24, 25)
_DOMAIN_EPS = 1e-3  # scaled units


def _domain_safe(z: Any, nx: int) -> Any:
    """Smoothly floor the ODE's divisor states so the vector field is finite everywhere."""
    import casadi as ca

    return ca.vertcat(
        *[
            ca.sqrt(z[i] * z[i] + _DOMAIN_EPS * _DOMAIN_EPS) if i in _SINGULAR_IDX else z[i]
            for i in range(nx)
        ]
    )


def _state_box(nx: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Loose conditioning box, EXCEPT temperatures, which get a physical-generous bound.

    ``_domain_safe`` makes the divisor states (CO2/vapour/biomass) finite everywhere, so
    those stay loose -- a *tight* physical bound (e.g. a CO2 upper limit that dosing
    legitimately crosses) makes the OCP infeasible, since IPOPT never violates a variable
    bound. But the ODE's temperature-dependent terms (exp/satVp in the aux states) still
    blow up at *extreme* temperatures, and a uniformly-loose box lets an iterate reach
    -500 degC and NaN the model. So temperatures keep a generous physical bound
    [-40, 80] degC (never binds; real greenhouse temp is -20..60), while every other state
    stays loose. The physical climate range is still enforced softly in the penalty.
    """
    lb = np.full(nx, -50.0)
    ub = np.full(nx, 1e4)
    for i in _TEMP_IDX:  # temperatures: [-40, 80] degC (scale 1e1); avoids the ODE NaN
        lb[i], ub[i] = -4.0, 8.0
    lb[_I_TIME], ub[_I_TIME] = -1.0, 1e6
    return lb, ub


def radau_coefficients(degree: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Radau collocation differentiation matrix ``C`` and end-weights ``D`` (pure)."""
    import casadi as ca

    tau = [0.0, *ca.collocation_points(degree, "radau")]
    coll = np.zeros((degree + 1, degree + 1))
    end = np.zeros(degree + 1)
    for j in range(degree + 1):
        poly = np.poly1d([1.0])
        for r in range(degree + 1):
            if r != j:
                poly = poly * np.poly1d([1.0, -tau[r]]) / (tau[j] - tau[r])
        end[j] = poly(1.0)
        dpoly = poly.deriv()
        for r in range(degree + 1):
            coll[j, r] = dpoly(tau[r])
    return coll, end


class PhysicsOnlyMPC(PhysicsController):
    """Direct-collocation economic MPC over the GreenLight ODE.

    Args:
        nominal_params: the controller's model parameters (never the sim's truth).
        horizon: number of control intervals.
        degree: collocation polynomial degree (Radau).
        dt: control interval [s].
        prices: economic coefficients; defaults to the GreenLight reward config.
        temp_bounds: soft air-temperature bounds [degC] (penalized, not hard).
    """

    control_kind = "absolute"  # outputs absolute controls -> harness uses step_raw_control

    def __init__(
        self,
        nominal_params: NominalParameters,
        *,
        horizon: int = 6,
        degree: int = 2,  # Radau degree; 2 tracks the CVODES plant to <1% at ~2x the
        # speed of degree 3 over a 900 s step (degree 1 is too coarse: ~10% on tAir).
        # See the degree benchmark in ../method/mpc-solver-notes.md.
        dt: float = 900.0,
        prices: dict[str, float] | None = None,
        temp_bounds: tuple[float, float] = (15.0, 34.0),
        w_energy: float = 1.0,
        w_temp: float = 50.0,
        w_fruit: float = 0.0,
        delta_u_max: float = 0.1,
        u_init: NDArray[np.float64] | None = None,
        state_scale: NDArray[np.float64] = STATE_SCALE,
        linear_solver: str = "mumps",
        expand: bool = True,
        jit: bool = False,
        objective: str = "energy_band",
        economic: EconomicConfig | None = None,
        climate_bounds: dict[str, tuple[float, float]] | None = None,
        max_iter: int = 150,
        verbose: bool = False,
        terminal_weight: float = 0.0,
    ) -> None:
        super().__init__(nominal_params)
        # Terminal cost weight (rho, van Laatum Eq 18). When > 0 the OCP adds an L1
        # terminal cost rho * sum_i |z_i(N) - z_term_i| in scaled coordinates (STATE_SCALE
        # is the per-state normalisation), pulling the horizon end toward a target state
        # supplied at solve time -- the mechanism by which arm 4 injects the RL rollout's
        # longer-horizon economics into the short-horizon MPC. 0 = plain arm 3.
        self.terminal_weight = terminal_weight
        # Objective mode: "energy_band" is the stabilising placeholder (modest energy
        # cost + a soft air-temperature band); "economic" is the GLG2 GreenhouseReward
        # EPI form (fruit revenue - operating costs - CO2/temp/RH penalties), the arm-3
        # objective that reproduces van Laatum's standalone MPC (L5). The placeholder is
        # kept as the default: it is robust and useful for solver validation.
        if objective not in ("energy_band", "economic"):
            raise ValueError(f"objective must be 'energy_band' or 'economic', got {objective!r}")
        if objective == "economic" and (economic is None or climate_bounds is None):
            raise ValueError("objective='economic' requires economic= and climate_bounds=")
        self.objective = objective
        self.economic = economic
        self.climate_bounds = climate_bounds
        self.max_iter = max_iter  # IPOPT iteration cap (bounds worst-case latency)
        self.verbose = verbose  # IPOPT print_level 5 for solver diagnosis
        self.horizon = horizon
        self.degree = degree
        self.dt = dt
        # IPOPT's KKT linear solver (the per-iteration cost driver). MUMPS ships with
        # every CasADi wheel and is deterministic (reproducible benchmark numbers);
        # SPRAL is open-source and can be slightly faster (Waechter & Biegler 2006
        # [wachter2006ipopt]). HSL (ma27/ma57/ma86) is NOT linked in the pip wheel.
        # Picked by the solver benchmark in ../method/mpc-solver-notes.md.
        self.linear_solver = linear_solver
        # expand: flatten the Opti MX graph to SX for faster function/derivative
        # evaluation. jit: compile the NLP functions to C once (amortized over a
        # multi-thousand-step season). Both are CasADi options [andersson2019casadi].
        self.expand = expand
        self.jit = jit
        # Prices are the single source of truth in benchmark.yaml, reaching here via the economic
        # config for the economic objective (the benchmark path). DEFAULT_PRICES is only a fallback
        # for the energy_band objective / bare construction, never the live benchmark value.
        if prices is not None:
            self.prices = dict(prices)
        elif self.economic is not None:
            self.prices = {
                "fruit": self.economic.fruit_price,
                "heat": self.economic.heating_price,
                "elec": self.economic.elec_price,
                "co2": self.economic.co2_price,
            }
        else:
            self.prices = dict(DEFAULT_PRICES)
        self.temp_bounds = temp_bounds
        # Objective weights. The climate-band penalty (w_temp) must dominate the
        # energy cost (w_energy) so a short-horizon MPC still heats when cold
        # instead of starving the crop. w_fruit is a (myopic) short-horizon crop
        # reward, off by default; the long-horizon value is arm 4's (RL-MPC) job.
        self.w_energy = w_energy
        self.w_temp = w_temp
        self.w_fruit = w_fruit
        self.delta_u_max = delta_u_max
        self._u_init = np.asarray(u_init, dtype=float) if u_init is not None else U_INIT.copy()
        self._u_prev = self._u_init.copy()
        self.scale = np.asarray(state_scale, dtype=float)
        self._params_vec = np.asarray(list(nominal_params.values.values()), dtype=float)
        self._warm: dict | None = None
        self._C, self._D = radau_coefficients(degree)
        # The parametric NLP is built once (lazily, on the first solve) and reused:
        # rebuilding the 28-state symbolic ODE + collocation graph every step is the
        # dominant cost. State, previous control, and the forecast are parameters.
        # This is the standard CasADi Opti pattern for MPC (Andersson et al. 2019
        # [andersson2019casadi]); see ../method/mpc-solver-notes.md.
        self._opti: Any = None  # ca.Opti, built lazily; casadi is stubless
        self._built_nd: int | None = None
        self._p_xterm: Any = None  # terminal-target parameter (arm 4); None if terminal_weight=0
        self._x_nodes: list = []  # interval-end state variables, for prediction readout
        self._last_pred: dict | None = None  # {"x": [...unscaled states...], "u": [...]}
        # Per-solve outcome tally: a deployment-reliability metric. ``clean`` = IPOPT
        # converged; ``salvage`` = non-optimal but its last iterate was a usable approximate
        # optimum; ``hold`` = iterate unusable, repeated the last control (the only true
        # degradation). The benchmark reports solve-success = (clean+salvage)/total per arm.
        self.solve_stats: dict[str, int] = {"clean": 0, "salvage": 0, "hold": 0}

    def reset(self, seed: int = 0) -> None:
        """Clear the warm-start and control history (start of a season).

        The compiled NLP is kept: its structure is season-independent, only the
        warm-start guess and control history reset.
        """
        self._warm = None
        self._u_prev = self._u_init.copy()
        self.solve_stats = {"clean": 0, "salvage": 0, "hold": 0}

    def act(self, ctx: ControlContext) -> NDArray[np.float64]:  # pragma: no cover - integration
        """Controller-protocol adapter: plan from the full state and weather forecast.

        The physics arm ignores ``ctx.obs`` (lossy) and ``ctx.env``; it needs the full
        state and the disturbance forecast, which is what ``solve`` consumes.
        """
        return self.solve(ctx.state, ctx.forecast)

    def model_step(  # pragma: no cover - integration (needs gl_gym)
        self, x: NDArray[np.float64], u: NDArray[np.float64], d: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """One-interval prediction with the MPC's scaled collocation model.

        The controller's own forward model, used by the validation harness to
        measure prediction fidelity against the plant (validation layer 2).
        """
        import casadi as ca
        from gl_gym.environments.models.ode import ODE

        s = self.scale
        z = ca.SX.sym("z", s.size)
        u_sym = ca.SX.sym("u", 6)
        d_sym = ca.SX.sym("d", np.asarray(d).size)
        p_sym = ca.SX.sym("p", self._params_vec.size)
        integrator = ca.integrator(
            "model",
            "collocation",
            {
                "x": z,
                "u": u_sym,
                "p": ca.vertcat(d_sym, p_sym),
                "ode": ODE(ca.DM(s) * z, u_sym, d_sym, p_sym) / ca.DM(s),
            },
            0.0,
            self.dt,
            {
                "number_of_finite_elements": 10,
                "collocation_scheme": "radau",
                "interpolation_order": 3,
            },
        )
        z_next = integrator(
            x0=ca.DM(np.asarray(x, dtype=float) / s),
            u=ca.DM(u),
            p=ca.vertcat(ca.DM(d), ca.DM(self._params_vec)),
        )["xf"]
        return np.asarray(z_next, dtype=float).flatten() * s

    def _economic_stage_cost(  # pragma: no cover - integration (needs casadi Opti)
        self, opti: Any, uk: Any, xn: Any, s: NDArray[np.float64]
    ) -> Any:
        """One-interval economic cost: operating costs + CO2/temp/RH soft penalties.

        Matches GLG2 ``GreenhouseReward``: heating/CO2/lamp costs via the GreenLight
        ``p[]`` conversions, and ``lambda . violations`` on the *unscaled* climate
        outputs (CO2 ppm, air temperature, RH) against ``climate_bounds``. Slacks keep
        the penalty differentiable. Returns the stage cost to be minimised.
        """
        import casadi as ca

        ec = self.economic
        bounds = self.climate_bounds
        assert ec is not None and bounds is not None  # guaranteed by __init__
        p = self._params_vec
        pi = ec.param_indices
        h_idx, c_idx, l_idx = ec.cost_input_indices
        dt_kwh = self.dt / 3600 * 1e-3  # W/aFlr -> kWh/m2 over one interval
        # operating costs over one dt interval (per GLG2 GreenhouseReward conversions)
        heat = uk[h_idx] * p[pi["heat_power"]] / p[pi["floor_area"]] * dt_kwh * ec.heating_price
        elec = uk[l_idx] * p[pi["lamp_power"]] * dt_kwh * ec.elec_price
        co2c = (
            uk[c_idx] * p[pi["co2_capacity"]] / p[pi["floor_area"]] * self.dt * 1e-6 * ec.co2_price
        )
        stage = heat + elec + co2c
        # output soft-bound penalties on the unscaled outputs. Each output is
        # normalised by a characteristic scale so its soft-bound constraint is O(1)
        # (raw ppm ~1000, RH ~100, temp ~10 otherwise wreck the KKT conditioning and
        # blow up IPOPT's duals). The penalty VALUE is unchanged: the slack is now in
        # normalised units, so the effective weight absorbs the scale
        # (weight*char*slack == lambda*raw_violation).
        co2_ppm, temp, rh = outputs_from_state(xn * ca.DM(s))
        lam = ec.penalty_weights  # order [CO2, temp, RH]
        rh_lo, rh_hi = bounds["rh_percent"]
        for y, (lo, hi), w, char in (
            (co2_ppm, bounds["co2_ppm"], lam[0], _CO2_SCALE),
            (temp, bounds["air_temp_c"], lam[1], _TEMP_SCALE),
            (rh, (rh_lo, rh_hi - _RH_UPPER_MARGIN), lam[2], _RH_SCALE),  # van Laatum RH margin
        ):
            s_lo = opti.variable()
            s_hi = opti.variable()
            opti.subject_to(s_lo >= 0)
            opti.subject_to(s_hi >= 0)
            opti.subject_to(y / char >= lo / char - s_lo)
            opti.subject_to(y / char <= hi / char + s_hi)
            stage += (w * char) * (s_lo + s_hi)
        return stage

    def _build(self, nd: int) -> None:  # pragma: no cover - integration (needs gl_gym)
        """Construct the parametric collocation NLP once (reused every step).

        Initial state, previous control, and the per-interval forecast are Opti
        *parameters*, so a step changes only their values, not the symbolic graph.
        The nominal parameters are baked in as constants (they never change).
        """
        import casadi as ca
        from gl_gym.environments.models.ode import ODE

        s = self.scale
        nx, nu = s.size, 6
        lb, ub = _state_box(nx)
        z_sym = ca.SX.sym("z", nx)
        u_sym = ca.SX.sym("u", nu)
        d_sym = ca.SX.sym("d", nd)
        f_scaled = ca.Function(
            "f",
            [z_sym, u_sym, d_sym],
            [
                ODE(ca.DM(s) * _domain_safe(z_sym, nx), u_sym, d_sym, ca.DM(self._params_vec))
                / ca.DM(s)
            ],
        )

        opti = ca.Opti()
        p_x0 = opti.parameter(nx)  # initial state (scaled)
        p_uprev = opti.parameter(nu)  # control applied before this horizon
        p_d = [opti.parameter(nd) for _ in range(self.horizon)]  # forecast per interval
        p_xterm = opti.parameter(nx) if self.terminal_weight > 0.0 else None  # RL terminal target
        cost = 0.0
        controls: list = []
        seed_states: list = []  # every state variable, seeded to x0 on a cold start
        x_nodes: list = []  # interval-end states, for the prediction readout
        xk = p_x0  # stage-0 state is the parameter itself (no extra variable)
        cfruit_start = p_x0[_I_CFRUIT] * s[_I_CFRUIT]
        prev_u = p_uprev
        u_first = None
        for k in range(self.horizon):
            uk = opti.variable(nu)
            opti.subject_to(opti.bounded(0, uk, 1))
            opti.subject_to(opti.bounded(-self.delta_u_max, uk - prev_u, self.delta_u_max))
            prev_u = uk
            controls.append(uk)
            if k == 0:
                u_first = uk
            dk = p_d[k]
            x_end = self._D[0] * xk
            coll = [opti.variable(nx) for _ in range(self.degree)]
            for j in range(1, self.degree + 1):
                opti.subject_to(opti.bounded(ca.DM(lb), coll[j - 1], ca.DM(ub)))
                seed_states.append(coll[j - 1])
                xp = self._C[0, j] * xk
                for r in range(1, self.degree + 1):
                    xp = xp + self._C[r, j] * coll[r - 1]
                opti.subject_to(self.dt * f_scaled(coll[j - 1], uk, dk) == xp)
                x_end = x_end + self._D[j] * coll[j - 1]
            xn = opti.variable(nx)
            opti.subject_to(xn == x_end)
            seed_states.append(xn)
            x_nodes.append(xn)
            if self.objective == "economic":
                cost += self._economic_stage_cost(opti, uk, xn, s)
            else:
                # placeholder: modest energy cost + a smooth soft air-temperature band
                # via slacks (differentiable, unlike fmax(0,.)^2).
                t_air = xn[_I_TAIR] * s[_I_TAIR]
                s_lo = opti.variable()
                s_hi = opti.variable()
                opti.subject_to(s_lo >= 0)
                opti.subject_to(s_hi >= 0)
                opti.subject_to(t_air >= self.temp_bounds[0] - s_lo)
                opti.subject_to(t_air <= self.temp_bounds[1] + s_hi)
                cost += self.w_energy * (uk[0] + uk[1] + uk[4]) + self.w_temp * (s_lo + s_hi)
            xk = xn
        # crop reward over the horizon (telescoping sum of fruit growth). In the
        # economic objective the fruit revenue uses the GreenhouseReward conversion
        # (mg dry matter -> EUR at the per-kgDW price); the placeholder uses w_fruit.
        cfruit_gain = xk[_I_CFRUIT] * s[_I_CFRUIT] - cfruit_start
        if self.objective == "economic":
            assert self.economic is not None  # guaranteed by __init__
            cost -= 1e-6 * self.economic.price_per_kg_dw * cfruit_gain
        else:
            cost -= self.w_fruit * cfruit_gain
        # terminal cost rho * sum_i |z_i(N) - z_term_i| (van Laatum Eq 18), in scaled
        # coordinates so STATE_SCALE is the per-state normalisation. L1 via slacks keeps
        # it differentiable. This pulls the horizon end toward the RL rollout's terminal
        # state (arm 4); time (index _I_TIME) is excluded (not a controllable target).
        if self.terminal_weight > 0.0 and p_xterm is not None:
            for i in range(nx):
                if i == _I_TIME:
                    continue
                s_t = opti.variable()
                opti.subject_to(s_t >= xk[i] - p_xterm[i])
                opti.subject_to(s_t >= p_xterm[i] - xk[i])
                cost += self.terminal_weight * s_t
        opti.minimize(cost)
        solver_opts: dict[str, Any] = {
            "expand": self.expand,  # MX -> SX graph for faster evaluation
        }
        if self.jit:  # compile the NLP functions to C once; amortized over the season
            solver_opts["jit"] = True
            solver_opts["compiler"] = "shell"
            solver_opts["jit_options"] = {"flags": ["-O2"], "verbose": False}
        opti.solver(
            "ipopt",
            {
                **solver_opts,
                "ipopt.print_level": 5 if self.verbose else 0,
                "print_time": 0,
                "ipopt.linear_solver": self.linear_solver,
                # Cap iterations to bound the worst-case per-step latency; pair it with
                # a short acceptable_iter so IPOPT returns a good-enough point promptly
                # instead of grinding to the cap (acceptable-point termination:
                # Waechter & Biegler 2006 [wachter2006ipopt]).
                "ipopt.max_iter": self.max_iter,
                "ipopt.acceptable_iter": 5,
                # Hard wall-clock safety net: a pathological (near-singular) state must
                # abort in seconds and let the salvage/hold fallback take over, never
                # grind for minutes. Healthy solves finish in ~1-5 s, well under this.
                "ipopt.max_cpu_time": 30.0,
                # van Laatum et al. 2026 [vanlaatum2026rlmpc] recipe: a quasi-Newton
                # (L-BFGS) Hessian avoids the exact 2nd-order ODE sensitivity that
                # fails, and relaxed tolerances terminate at an approximately
                # feasible/optimal point, making the stiff model tractable.
                "ipopt.hessian_approximation": "limited-memory",
                "ipopt.tol": 0.1,
                "ipopt.acceptable_tol": 0.1,
                "ipopt.constr_viol_tol": 1e-2,
                "ipopt.acceptable_constr_viol_tol": 0.1,
                # Speed lever not taken here: IPOPT's default linear solver is MUMPS
                # (the only one shipped with the pip CasADi wheel). An HSL build
                # (MA57/MA27) is materially faster on these KKT systems (HSL [hsl];
                # Waechter & Biegler 2006 [wachter2006ipopt]); adopt it if the public
                # artifact ships an HSL-enabled CasADi. See ../method/mpc-solver-notes.md.
            },
        )
        self._opti = opti
        self._p_x0, self._p_uprev, self._p_d = p_x0, p_uprev, p_d
        self._p_xterm = p_xterm
        self._u_first = u_first
        self._controls = controls
        self._seed_states = seed_states
        self._x_nodes = x_nodes
        self._built_nd = nd

    def solve(  # pragma: no cover - integration (needs gl_gym + solver)
        self,
        x: NDArray[np.float64],
        weather: NDArray[np.float64],
        terminal_target: NDArray[np.float64] | None = None,
        state_guess: list[NDArray[np.float64]] | None = None,
    ) -> NDArray[np.float64]:
        """Return the first control ``u0`` in [0,1]^6 for state ``x`` and a weather
        forecast ``weather`` of shape ``(>=horizon, nd)`` (disturbances per interval).

        ``terminal_target`` (an unscaled state) is required when ``terminal_weight > 0``:
        the horizon end is pulled toward it by the terminal cost (arm 4's RL terminal state).

        ``state_guess`` (an optional list of ``horizon`` unscaled states, one per interval
        end) warm-starts the collocation states from a physically consistent forward
        trajectory (arm 4 passes the RL rollout's states). Defaults to seeding every state to
        ``x0`` -- the robust cold-start guess; a *shifted previous-solve* seed backfired on
        the stiff solves, but a fresh forward rollout from the current ``x0`` does not.
        """
        s = self.scale
        nu = 6
        nd = int(weather.shape[1])
        if self._opti is None or self._built_nd != nd:
            self._build(nd)
        opti = self._opti
        z0 = np.asarray(x, dtype=float) / s
        opti.set_value(self._p_x0, z0)
        opti.set_value(self._p_uprev, self._u_prev)
        if self.terminal_weight > 0.0 and self._p_xterm is not None:
            if terminal_target is None:
                raise ValueError("terminal_weight > 0 requires a terminal_target in solve()")
            opti.set_value(self._p_xterm, np.asarray(terminal_target, dtype=float) / s)
        for k in range(self.horizon):
            opti.set_value(self._p_d[k], weather[k])
        # Seed the states fresh from the current x0 every step (they shift too much
        # between steps to reuse) and warm-start only the controls from the previous
        # solve (they persist well). Reusing the full primal, including states, made
        # the stiff cold-region solves fail; controls-only is both faster and robust.
        if state_guess is not None:
            # Warm-start each interval's states (degree collocation points + the interval
            # node) from the forward-rollout state at that interval's end; fall back to x0
            # for any interval past the guess length.
            per = self.degree + 1
            for k in range(self.horizon):
                zk = np.asarray(state_guess[k], dtype=float) / s if k < len(state_guess) else z0
                for v in self._seed_states[k * per : (k + 1) * per]:
                    opti.set_initial(v, zk)
        else:
            for v in self._seed_states:
                opti.set_initial(v, z0)
        warm_u = self._warm["u"] if self._warm is not None else None
        for k, uk in enumerate(self._controls):
            opti.set_initial(uk, warm_u[k] if warm_u is not None else 0.3)
        try:
            sol = opti.solve()
        except RuntimeError:
            # IPOPT stopped non-optimal (max_iter / a stiff-ODE corner). Its last
            # iterate (via CasADi's Opti.debug, Andersson et al. 2019
            # [andersson2019casadi]) is usually a finite, in-bounds *approximate*
            # optimum under the relaxed tolerances, so prefer it (an honest approximate
            # solve) over the crude hold-last-control; hold only if it is unusable.
            u_try = np.asarray(opti.debug.value(self._u_first), dtype=float).reshape(nu)
            if (
                np.all(np.isfinite(u_try))
                and np.all(u_try >= -_U_TOL)
                and np.all(u_try <= 1 + _U_TOL)
            ):
                u0 = np.clip(u_try, 0.0, 1.0)
                self._u_prev = u0
                self._warm = None  # drop the warm start after a non-clean solve
                self.solve_stats["salvage"] += 1
                return u0
            self._warm = None
            self.solve_stats["hold"] += 1
            return np.asarray(self._u_prev, dtype=float).reshape(nu)
        u0 = np.asarray(sol.value(self._u_first), dtype=float).reshape(nu)
        self._u_prev = u0
        self.solve_stats["clean"] += 1
        controls_pred = [np.asarray(sol.value(c), dtype=float).reshape(nu) for c in self._controls]
        self._warm = {"u": controls_pred}
        # cache the predicted (unscaled) state trajectory and controls for diagnosis
        self._last_pred = {
            "x": [np.asarray(sol.value(xn), dtype=float).flatten() * s for xn in self._x_nodes],
            "u": controls_pred,
        }
        return u0
