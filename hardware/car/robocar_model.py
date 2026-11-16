"""RoboCar risk model shared by robocar_certify.py and robocar_feasibility.py (Section 6.3).

Same structure as the vehicle-following model of Section 6.1, scaled to the 1/10 vehicle.
State x = [s, v, vl]: gap [m], ego speed [m/s], lead speed [m/s].
  s' = vl - v,   v' = u + d_v (|d_v| <= DELTA_V),   vl' = w_l (|w_l| <= A_LEAD)
Risk:        m(x) = s - S0 - H v - (v^2 - vl^2)/(2 BD),   R(x) = sigma(-m/ELL)
Nominal law: u_nom = clip(KV (vl - v) + KS (s - G0 - TH v), U_MIN, U_MAX)
"""
import json, math

DEFAULTS = dict(
    S0=0.15, H=0.40, ELL=0.25, BD=None,             # BD = BD_FRAC * |U_MIN| if None
    BD_FRAC=0.8, KV=0.8, KS=0.5, G0=0.40, TH=1.0,   # nominal law: gap target G0 + TH v
    VMAX=1.0,
    U_MIN=-2.0, U_MAX=1.5, DELTA_V=0.5, A_LEAD=1.5,
)


class Car:
    def __init__(self, params_file=None, **over):
        p = dict(DEFAULTS)
        if params_file:
            with open(params_file) as f:
                p.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
        p.update(over)
        if p["BD"] is None:
            p["BD"] = p["BD_FRAC"] * abs(p["U_MIN"])
        self.p = p
        for k, v in p.items():
            setattr(self, k, v)

    def margin(self, s, v, vl):
        return s - self.S0 - self.H * v - (v * v - vl * vl) / (2 * self.BD)

    def risk(self, s, v, vl):
        z = -self.margin(s, v, vl) / self.ELL
        z = max(-60.0, min(60.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def margin_of_level(self, b):
        return -self.ELL * math.log(b / (1 - b))

    def gap_on_level(self, b, v, vl):
        """Gap s that puts (s, v, vl) on {R = b}."""
        return self.margin_of_level(b) + self.S0 + self.H * v + (v * v - vl * vl) / (2 * self.BD)
