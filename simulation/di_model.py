"""Double-integrator example with a closed-form (analytic) certificate.

Plant: x1' = x2,  x2' = u + w,   |u| <= U_BAR,  |w| <= DELTA   (matched disturbance)
       f(x) = [x2, 0],  g = [0, 1],  d = [0, w].
Margin (distance to the wall x1 = X_BAR, minus headway TAU*x2 and braking distance at BRK):
       h(x) = X_BAR - x1 - TAU*x2 - x2|x2|/(2*BRK)          (C^1 in x)
Risk:  R(x) = sigma(-h(x)/ELL) in (0,1).
On the level set {R = b}: sigma' = b(1-b) =: s_b and
   grad R = (s_b/ELL) [1, TAU + |x2|/BRK],   a(x) = (s_b/ELL)(TAU + |x2|/BRK) >= s_b TAU/ELL,
   L_f R = (s_b/ELL) x2,                       sup_w grad R . d = |a| DELTA.
Closed-form constants (U = [-U_BAR, U_BAR], requires U_BAR - DELTA >= BRK):
   eta   = s_b TAU (U_BAR - DELTA)/ELL       (input-feasibility margin incl. disturbance, attained at x2 = 0)
   k_bar = (2 U_BAR)^2 / eta                 (projected-law gain bound)
   a_min = s_b TAU / ELL                     (no near-singular boundary points if TAU > 0)
TAU = 0 is the negative case: a(x) = 0 at x2 = 0 on the boundary, eta = 0, k_bar = infinity.
"""
import numpy as np

X_BAR, BRK, ELL = 10.0, 1.0, 0.5
U_BAR, DELTA = 2.0, 0.5
DT = 0.01

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def h(x1, x2, tau):
    return X_BAR - x1 - tau*x2 - x2*np.abs(x2)/(2*BRK)

def risk_grad(x1, x2, tau):
    R = sigmoid(-h(x1, x2, tau)/ELL); sp = R*(1 - R)
    return R, sp/ELL, sp*(tau + np.abs(x2)/BRK)/ELL     # R, dR/dx1, dR/dx2 (= a)

def h_of_level(b):
    return -ELL*np.log(b/(1 - b))

def analytic_constants(b, tau):
    s = b*(1 - b)
    eta = s*tau*(U_BAR - DELTA)/ELL
    return dict(level=b, s_b=s, margin_h=float(h_of_level(b)), a_min=s*tau/ELL, eta=eta,
                k_bar=(2*U_BAR)**2/eta if eta > 0 else float("inf"),
                feasibility_requires=f"U_BAR - DELTA = {U_BAR-DELTA} >= BRK = {BRK}: {U_BAR-DELTA >= BRK}")

def u_nominal(x1, x2, r, kp=1.0, kd=1.5):
    return np.clip(-kp*(x1 - r) - kd*x2, -U_BAR, U_BAR)

def gain(M, k0, k1, mu):
    return k0*np.clip(1.0 - M/mu, 0.0, 1.0) + k1*np.maximum(0.0, -M)

def u_supervisor(x1, x2, r, B, tau, k0, k1=10.0, mu=0.05):
    R, R1, a = risk_grad(x1, x2, tau)
    uu = u_nominal(x1, x2, r) - gain(B - R, k0, k1, mu)*a
    u = np.clip(uu, -U_BAR, U_BAR)
    return u, R, a

def u_cbf_robust(x1, x2, r, B, tau, gamma=5.0):
    R, R1, a = risk_grad(x1, x2, tau)
    LfR = R1*x2 + np.abs(a)*DELTA
    un = u_nominal(x1, x2, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        bnd = (gamma*(B - R) - LfR)/a
    u = np.where(a > 1e-12, np.minimum(un, bnd), un)
    return np.clip(u, -U_BAR, U_BAR), R

def step(x1, x2, u, w, dt=DT):
    return x1 + x2*dt, x2 + (u + w)*dt
