
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List
import math
import numpy as np
import pandas as pd


@dataclass
class FOPDTModel:
    gain: float
    tau: float
    dead_time: float

    def to_dict(self) -> Dict[str, float]:
        return {"gain": float(self.gain), "tau": float(self.tau), "dead_time": float(self.dead_time)}


@dataclass
class PIDTuning:
    name: str
    method: str
    controller_type: str
    kc: float
    ti: float
    td: float

    @property
    def ki(self) -> float:
        return 0.0 if math.isinf(self.ti) or self.ti <= 0 else self.kc / self.ti

    @property
    def kd(self) -> float:
        return self.kc * self.td

    def to_dict(self) -> Dict[str, float | str]:
        d = asdict(self)
        d["ki"] = float(self.ki)
        d["kd"] = float(self.kd)
        return d

    @classmethod
    def from_parallel_gains(
        cls,
        name: str,
        controller_type: str,
        kc: float,
        ki: float,
        kd: float,
        method: str = "Manual",
    ) -> "PIDTuning":
        controller_type = controller_type.upper()
        kc_mag = abs(float(kc))
        ki_mag = abs(float(ki))
        kd_mag = abs(float(kd))
        ti = math.inf if ki_mag < 1e-12 else kc_mag / ki_mag
        td = 0.0 if kc_mag < 1e-12 else kd_mag / kc_mag
        if controller_type == "PI":
            td = 0.0
        return cls(name=name, method=method, controller_type=controller_type, kc=kc_mag, ti=float(ti), td=float(td))

    @classmethod
    def from_time_constants(
        cls,
        name: str,
        controller_type: str,
        kc: float,
        ti: float,
        td: float,
        method: str = "Manual",
    ) -> "PIDTuning":
        controller_type = controller_type.upper()
        if controller_type == "PI":
            td = 0.0
        return cls(
            name=name,
            method=method,
            controller_type=controller_type,
            kc=abs(float(kc)),
            ti=math.inf if math.isinf(ti) else abs(float(ti)),
            td=abs(float(td)),
        )


@dataclass
class SimulationConfig:
    horizon: float = 3600.0
    dt: float = 1.0
    setpoint: float = 0.02
    disturbance_time: float = 1e9
    disturbance_amplitude: float = 0.0
    mv_min: float = -50.0
    mv_max: float = 50.0
    anti_windup: bool = True
    derivative_filter_n: float = 10.0
    beta: float = 1.0
    noise_std: float = 0.0
    direct_action: bool = True


@dataclass
class SimulationResult:
    data: pd.DataFrame
    metrics: Dict[str, float]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_dead_time(dead_time: float, dt: float) -> int:
    return max(0, int(round(dead_time / max(dt, 1e-9))))


def tuning_imc_simc(model: FOPDTModel, lambda_value: float, controller_type: str = "PID", name: str = "IMC/SIMC") -> PIDTuning:
    k = abs(float(model.gain))
    tau = max(float(model.tau), 1e-9)
    theta = max(float(model.dead_time), 0.0)
    lam = max(float(lambda_value), 1e-9)
    controller_type = controller_type.upper()
    if k < 1e-12:
        raise ValueError("O ganho do processo não pode ser zero.")
    if controller_type == "PI":
        kc = tau / (k * (lam + theta))
        ti = min(tau, 4.0 * (lam + theta))
        td = 0.0
    else:
        kc = (tau + 0.5 * theta) / (k * (lam + 0.5 * theta))
        ti = tau + 0.5 * theta
        td = (tau * theta) / max((2.0 * tau + theta), 1e-9)
    return PIDTuning(name=name, method="IMC/SIMC", controller_type=controller_type, kc=kc, ti=ti, td=td)


def tuning_ziegler_nichols_openloop(model: FOPDTModel, controller_type: str = "PID", name: str = "Ziegler-Nichols") -> PIDTuning:
    k = abs(float(model.gain))
    tau = max(float(model.tau), 1e-9)
    theta = max(float(model.dead_time), 1e-9)
    controller_type = controller_type.upper()
    if controller_type == "PI":
        kc = 0.9 * tau / (k * theta)
        ti = 3.33 * theta
        td = 0.0
    else:
        kc = 1.2 * tau / (k * theta)
        ti = 2.0 * theta
        td = 0.5 * theta
    return PIDTuning(name=name, method="Ziegler-Nichols", controller_type=controller_type, kc=kc, ti=ti, td=td)


def tuning_cohen_coon(model: FOPDTModel, controller_type: str = "PID", name: str = "Cohen-Coon") -> PIDTuning:
    k = abs(float(model.gain))
    tau = max(float(model.tau), 1e-9)
    theta = max(float(model.dead_time), 1e-9)
    r = theta / tau
    controller_type = controller_type.upper()
    if controller_type == "PI":
        kc = (0.9 / k) * (1.0 / r) * (1.0 + r / 12.0)
        ti = theta * ((30.0 + 3.0 * r) / (9.0 + 20.0 * r))
        td = 0.0
    else:
        kc = (1.0 / k) * (1.0 / r) * (4.0 / 3.0 + r / 4.0)
        ti = theta * ((32.0 + 6.0 * r) / (13.0 + 8.0 * r))
        td = theta * (4.0 / (11.0 + 2.0 * r))
    return PIDTuning(name=name, method="Cohen-Coon", controller_type=controller_type, kc=kc, ti=ti, td=td)


def simulate_closed_loop(model: FOPDTModel, tuning: PIDTuning, config: SimulationConfig) -> SimulationResult:
    dt = max(float(config.dt), 1e-9)
    n_steps = int(config.horizon / dt) + 1
    time = np.linspace(0.0, config.horizon, n_steps)

    dead_steps = safe_dead_time(model.dead_time, dt)
    mv_queue = [0.0] * max(dead_steps + 1, 1)

    pv = np.zeros(n_steps)
    pv_meas = np.zeros(n_steps)
    sp = np.zeros(n_steps)
    mv = np.zeros(n_steps)
    p_term = np.zeros(n_steps)
    i_term = np.zeros(n_steps)
    d_term = np.zeros(n_steps)
    disturbance = np.zeros(n_steps)

    if n_steps > int(300 / dt):
        sp[int(300 / dt):] = float(config.setpoint)
    else:
        sp[:] = float(config.setpoint)

    integral = 0.0
    filtered_d = 0.0
    prev_pv_meas = 0.0
    alpha = tuning.td / max(tuning.td + (dt * max(config.derivative_filter_n, 1e-6)), 1e-9) if tuning.td > 0 else 0.0

    for i in range(1, n_steps):
        disturbance[i] = config.disturbance_amplitude if time[i] >= config.disturbance_time else 0.0
        pv_meas[i - 1] = pv[i - 1] + (np.random.normal(0.0, config.noise_std) if config.noise_std > 0 else 0.0)

        error = (sp[i - 1] - pv_meas[i - 1]) if config.direct_action else (pv_meas[i - 1] - sp[i - 1])
        weighted_error = (config.beta * sp[i - 1] - pv_meas[i - 1]) if config.direct_action else (pv_meas[i - 1] - config.beta * sp[i - 1])

        proportional = tuning.kc * weighted_error
        tentative_integral = integral + tuning.ki * error * dt

        raw_derivative = -(pv_meas[i - 1] - prev_pv_meas) / dt
        filtered_d = alpha * filtered_d + (1.0 - alpha) * raw_derivative
        derivative = tuning.kd * filtered_d

        unsat_mv = proportional + tentative_integral + derivative
        sat_mv = clamp(unsat_mv, config.mv_min, config.mv_max)
        if config.anti_windup:
            at_upper = sat_mv >= config.mv_max - 1e-12 and unsat_mv > sat_mv and error > 0
            at_lower = sat_mv <= config.mv_min + 1e-12 and unsat_mv < sat_mv and error < 0
            if not (at_upper or at_lower):
                integral = tentative_integral
        else:
            integral = tentative_integral

        mv[i] = clamp(proportional + integral + derivative, config.mv_min, config.mv_max)
        p_term[i], i_term[i], d_term[i] = proportional, integral, derivative

        mv_queue.append(mv[i])
        delayed_mv = mv_queue.pop(0)

        dpv = (-(pv[i - 1]) + model.gain * delayed_mv + disturbance[i]) / max(model.tau, 1e-9)
        pv[i] = pv[i - 1] + dt * dpv
        prev_pv_meas = pv_meas[i - 1]

    pv_meas[-1] = pv[-1] + (np.random.normal(0.0, config.noise_std) if config.noise_std > 0 else 0.0)
    data = pd.DataFrame({"time": time, "sp": sp, "pv": pv, "pv_meas": pv_meas, "mv": mv, "p_term": p_term, "i_term": i_term, "d_term": d_term, "disturbance": disturbance})
    error = data["sp"] - data["pv"]
    amp = float(data["sp"].iloc[-1] - data["sp"].iloc[0])
    overshoot = 0.0 if abs(amp) < 1e-12 else max(0.0, (float(data["pv"].max()) - float(data["sp"].iloc[-1])) / abs(amp) * 100.0)
    band = 0.02 * max(abs(amp), 1e-9)
    settling = math.nan
    for i in range(len(data)):
        if np.all(np.abs(data["pv"].iloc[i:] - data["sp"].iloc[-1]) <= band):
            settling = float(data["time"].iloc[i])
            break
    metrics = {
        "overshoot_pct": float(overshoot),
        "settling_time": float(settling),
        "IAE": float(np.trapz(np.abs(error), data["time"])),
        "ISE": float(np.trapz(error**2, data["time"])),
        "ITAE": float(np.trapz(data["time"] * np.abs(error), data["time"])),
        "total_variation_mv": float(np.sum(np.abs(np.diff(data["mv"])))),
    }
    return SimulationResult(data=data, metrics=metrics)
