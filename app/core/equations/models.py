
from math import sqrt, exp
from .base import BaseEquation, EquationResult
from ..models.inputs import ProcessConditions
from ..thermo.mixture_props import MixtureProps
from ..utils.si import normalize, audit_text_header, SINorm

R_UNIV = 8.31446261815324  # J/mol-K
BETA_MIN, BETA_MAX = 0.2, 0.75
RE_MIN, RE_MAX = 2e4, 1e7

def cd_iso5167(beta_val: float, Re_D: float, L1_m: float, tap_type: str, M2p: float = 0.0) -> float:
    b = beta_val
    A = 0.0 if tap_type in ("flange","corner") else 1.0
    term_Re = 0.000521 * (1.0e6 * b / Re_D) ** 0.7
    term_Re_beta = (0.0188 + 0.0063 * A) * (b ** 3) * (1.0e6 / Re_D) ** 0.3
    term_L1 = (0.043 + 0.080 * exp(-10.0 * L1_m) - 0.123 * exp(-7.0 * L1_m)) * (1.0 - 0.11 * A) * (b**4) / (1.0 - b**4)
    term_Mach = -0.031 * (M2p - 0.8 * (M2p ** 1.1)) * (b ** 1.3)
    C = 0.5961 + 0.0261 * b**2 - 0.216 * b**8 + term_Re + term_Re_beta + term_L1 + term_Mach
    return C

def validate_ranges(b: float, Re_D: float) -> list[str]:
    msgs = []
    if not (BETA_MIN <= b <= BETA_MAX):
        msgs.append(f"β={b:.3f} fora de [{BETA_MIN:.2f}, {BETA_MAX:.2f}] (ISO 5167).")
    if Re_D > 0 and not (RE_MIN <= Re_D <= RE_MAX):
        msgs.append(f"Re_D={Re_D:.1f} fora de [{RE_MIN:.1e}, {RE_MAX:.1e}] (ISO 5167).")
    return msgs

def iter_asme_aga(si: SINorm, L1_m: float, tap_type: str, eps: float) -> tuple[float, float, float, list]:
    Cd = 0.61
    hist = []
    Ev = 1.0 / sqrt(1.0 - si.beta**4)
    for _ in range(60):
        qm = Ev * Cd * eps * si.A_orif_m2 * sqrt(2.0 * si.delta_p_pa * si.rho1)
        V = qm / (si.rho1 * si.A_pipe_m2)
        Re_D = max(1.0, si.rho1 * V * si.D_m / si.mu)
        Cd_new = cd_iso5167(si.beta, Re_D, L1_m, tap_type, M2p=0.0)
        hist.append((Cd, Re_D))
        if abs(Cd_new - Cd) < 1e-7:
            Cd = Cd_new
            break
        Cd = Cd_new
    qm = Ev * Cd * eps * si.A_orif_m2 * sqrt(2.0 * si.delta_p_pa * si.rho1)
    V = qm / (si.rho1 * si.A_pipe_m2)
    Re_D = max(1.0, si.rho1 * V * si.D_m / si.mu)
    return qm, Cd, Re_D, hist

class AsmeMFC3M(BaseEquation):
    name = "ASME MFC-3M"
    def check_applicability(self, s: ProcessConditions, m: MixtureProps):
        return (s.pback_bar/s.p0_bar >= 0.8), f"P2/P1 = {s.pback_bar/s.p0_bar:.3f} ≥ 0.8"
    def compute(self, s: ProcessConditions, m: MixtureProps):
        ok, reason = self.check_applicability(s, m)
        if not ok: return EquationResult(self.name, False, reason)
        si = normalize(s, m); k = m.k
        P2P1 = si.p2_pa/si.p0_pa
        eps = 1.0 - (0.351 + 0.256*si.beta**4 + 0.93*si.beta**8) * (1.0 - P2P1**(k-1.0))
        qm, Cd, ReD, hist = iter_asme_aga(si, s.L1_m, s.tap_type, eps)
        Ev = 1.0 / sqrt(1.0 - si.beta**4)
        warns = validate_ranges(si.beta, ReD)
        audit = audit_text_header() + f"""
A = π/4·d² = {si.A_orif_m2:.6e} [m²]
ΔP = {si.delta_p_pa:.6e} [Pa]
ρ = {si.rho1:.6e} [kg/m³]
Ev = {Ev:.6f} (adim), ε = {eps:.6f} (adim), Cd = {Cd:.6f} (adim), Re_D = {ReD:.2e}
q_m = Ev·Cd·ε·A·√(2ΔPρ) = {qm:.6e} kg/s
"""
        reason2 = reason + ("" if not warns else " | " + " ".join(warns))
        return EquationResult(self.name, True, reason2, qm, notes="; ".join(warns) if warns else "", extras={
            "beta": si.beta, "Ev": Ev, "eps": eps, "Cd": Cd, "Re_D": ReD, "audit": audit
        })

class AGA31(BaseEquation):
    name = "AGA Report 3.1"
    def check_applicability(self, s: ProcessConditions, m: MixtureProps):
        r = s.pback_bar/s.p0_bar
        return ((r > 0.8) and (r < 1.0)), f"0.8 < P2/P1 = {r:.3f} < 1.0"
    def compute(self, s: ProcessConditions, m: MixtureProps):
        ok, reason = self.check_applicability(s, m)
        if not ok: return EquationResult(self.name, False, reason)
        si = normalize(s, m); k = m.k
        N3 = 1000.0
        x1 = (si.p2_pa - si.p0_pa) / (N3 * si.p0_pa)
        eps = 1.0 - (0.41 + 0.35*si.beta**4) * (x1 / k)
        qm, Cd, ReD, hist = iter_asme_aga(si, s.L1_m, s.tap_type, eps)
        Ev = 1.0 / sqrt(1.0 - si.beta**4)
        warns = validate_ranges(si.beta, ReD)
        audit = audit_text_header() + f"""
A = π/4·d² = {si.A_orif_m2:.6e} [m²]
ΔP = {si.delta_p_pa:.6e} [Pa]
ρ = {si.rho1:.6e} [kg/m³]
Ev = {Ev:.6f} (adim), ε = {eps:.6f} (adim), Cd = {Cd:.6f} (adim), Re_D = {ReD:.2e}
q_m = Ev·Cd·ε·A·√(2ΔPρ) = {qm:.6e} kg/s
"""
        reason2 = reason + ("" if not warns else " | " + " ".join(warns))
        return EquationResult(self.name, True, reason2, qm, notes="; ".join(warns) if warns else "", extras={
            "beta": si.beta, "Ev": Ev, "eps": eps, "Cd": Cd, "Re_D": ReD, "audit": audit
        })

class ANPModel(BaseEquation):
    name = "ANP"
    def check_applicability(self, s: ProcessConditions, m: MixtureProps):
        return True, "Sem condições matemáticas"
    def compute(self, s: ProcessConditions, m: MixtureProps):
        # Normalização SI
        si = normalize(s, m)
        # Equação ANP (2025) revisada
        F1 = 132.52  # Constante empírica (ajustada para q_m em kg/s)
        d_m = si.d_m          # [m]
        rho1 = si.rho1        # [kg/m³]
        P0_bar = s.p0_bar     # [bar]
        qm = F1 * (d_m ** 2) * (rho1 * P0_bar) ** 0.5

        audit = audit_text_header() + f"""
d = {d_m:.6e} [m]
ρ₁ = {rho1:.6e} [kg/m³]
P₀ = {P0_bar:.6e} [bar]
F₁ = {F1:.2f}
q_m = F₁·d²·√(ρ₁·P₀) = {qm:.6e} kg/s
"""
        return EquationResult(
            self.name,
            True,
            "Aplicável por padrão (ANP 2025 — versão simplificada)",
            qm,
            extras={"beta": si.beta, "audit": audit},
        )

class YuHu1(BaseEquation):
    name = "YuHu 1"
    def _epsilon(self, Ma1: float, k: float) -> float: return (2.0 + Ma1**2 * (k - 1.0))
    def _mach(self, mw_kg_per_kmol: float, p_pa: float, k: float, T_K: float, rho: float) -> float:
        M = mw_kg_per_kmol/1000.0
        return sqrt( 3.0 * M * p_pa / (k * R_UNIV * T_K * rho) )
    def check_applicability(self, s: ProcessConditions, m: MixtureProps):
        si = normalize(s, m)
        Ma1 = self._mach(m.MW_mix, si.p0_pa, m.k, si.t0_K, si.rho1)
        eps = self._epsilon(Ma1, m.k)
        cond = si.p2_pa > si.p0_pa * Ma1 * sqrt(2.0*eps/(m.k+1.0))
        return cond, f"P2 > P1*Ma1*√(2ε/(k+1))? {'Sim' if cond else 'Não'} (Ma1={Ma1:.3f}, ε={eps:.3f})"
    def compute(self, s: ProcessConditions, m: MixtureProps):
        ok, reason = self.check_applicability(s, m)
        if not ok: return EquationResult(self.name, False, reason)
        si = normalize(s, m)
        k = m.k; A = si.A_orif_m2
        qm = A * si.p2_pa * sqrt( (m.MW_mix/1000.0 * k) / (R_UNIV * si.t0_K) * (2.0/(k+1.0))**((k+1.0)/(k-1.0)) )
        audit = audit_text_header() + f"""
A = {A:.6e} [m²], P2 = {si.p2_pa:.6e} [Pa], T = {si.t0_K:.2f} K
q_m = A·P2·√( (M·k)/(R·T) · (2/(k+1))^{(k+1)/(k-1)} ) = {qm:.6e} kg/s
"""
        return EquationResult(self.name, True, reason, qm, extras={"beta": si.beta, "audit": audit})

class YuHu2(BaseEquation):
    name = "YuHu 2"
    def _epsilon(self, Ma1: float, k: float) -> float: return (2.0 + Ma1**2 * (k - 1.0))
    def _cpr(self, k: float) -> float: return (2.0/(k+1.0))**(k/(k-1.0))
    def _mach(self, mw_kg_per_kmol: float, p_pa: float, k: float, T_K: float, rho: float) -> float:
        M = mw_kg_per_kmol/1000.0
        return sqrt( 3.0 * M * p_pa / (k * R_UNIV * T_K * rho) )
    def check_applicability(self, s: ProcessConditions, m: MixtureProps):
        si = normalize(s, m)
        Ma1 = self._mach(m.MW_mix, si.p0_pa, m.k, si.t0_K, si.rho1)
        eps = self._epsilon(Ma1, m.k)
        cpr = self._cpr(m.k)
        cond1 = si.p2_pa > si.p0_pa * Ma1 * sqrt(2.0*eps/(m.k+1.0))
        cond2 = (101325.0 / si.p2_pa) >= cpr
        ok = cond1 and cond2
        return ok, f"cond1={cond1}, Pa/P2>=CPR? {cond2} (CPR={cpr:.3f})"
    def compute(self, s: ProcessConditions, m: MixtureProps):
        ok, reason = self.check_applicability(s, m)
        if not ok: return EquationResult(self.name, False, reason)
        si = normalize(s, m)
        k = m.k; A = si.A_orif_m2
        Cd = 0.61
        term = ( (101325.0/si.p2_pa)**(2.0/k) - (101325.0/si.p2_pa)**((k+1.0)/k) )
        qm = A * Cd * si.p2_pa * sqrt( (2.0*(m.MW_mix/1000.0))/(R_UNIV*si.t0_K) * (k/(k-1.0)) * term )
        audit = audit_text_header() + f"""
A = {A:.6e} [m²], P2 = {si.p2_pa:.6e} [Pa], T = {si.t0_K:.2f} K
q_m = A·Cd·P2·√( (2M)/(R·T) · k/(k-1) · [ (Patm/P2)^{2/k} - (Patm/P2)^{(k+1)/k} ] ) = {qm:.6e} kg/s
"""
        return EquationResult(self.name, True, reason, qm, extras={"beta": si.beta, "Cd": Cd, "audit": audit})

class Montiel(BaseEquation):
    name = "Montiel"
    def _epsilon(self, Ma1: float, k: float) -> float: return (2.0 + Ma1**2 * (k - 1.0))
    def _mach(self, mw_kg_per_kmol: float, p_pa: float, k: float, T_K: float, rho: float) -> float:
        M = mw_kg_per_kmol/1000.0
        return sqrt( 3.0 * M * p_pa / (k * R_UNIV * T_K * rho) )
    def check_applicability(self, s: ProcessConditions, m: MixtureProps):
        si = normalize(s, m)
        Ma1 = self._mach(m.MW_mix, si.p0_pa, m.k, si.t0_K, si.rho1)
        eps = self._epsilon(Ma1, m.k)
        cond = si.p2_pa < si.p0_pa * Ma1 * sqrt(2.0*eps/(m.k+1.0))
        return cond, f"P2 < P1*Ma1*√(2ε/(k+1))? {'Sim' if cond else 'Não'} (Ma1={Ma1:.3f}, ε={eps:.3f})"
    def compute(self, s: ProcessConditions, m: MixtureProps):
        ok, reason = self.check_applicability(s, m)
        if not ok: return EquationResult(self.name, False, reason)
        si = normalize(s, m)
        k = m.k; M = m.MW_mix/1000.0; T1 = si.t0_K; A = si.A_orif_m2
        Ma1 = sqrt( 3.0 * M * si.p0_pa / (k * R_UNIV * T1 * si.rho1) )
        qm = A * Ma1 * si.p0_pa * sqrt( (k/(R_UNIV*T1/M)) * (2.0/(k+1.0))**((k+1.0)/(k-1.0)) )
        audit = audit_text_header() + f"""
A = {A:.6e} [m²], P0 = {si.p0_pa:.6e} [Pa], T = {T1:.2f} K
q_m = A·Ma1·P0·√( k/(R_m·T) · (2/(k+1))^{(k+1)/(k-1)} ) = {qm:.6e} kg/s
"""
        return EquationResult(self.name, True, reason, qm, extras={"beta": si.beta, "Ma1": Ma1, "audit": audit})

class Cengel(BaseEquation):
    name = "Cengel"
    def check_applicability(self, s: ProcessConditions, m: MixtureProps):
        si = normalize(s, m)
        crit = (2.0/(m.k+1.0))**(m.k/(m.k-1.0))
        r = si.p2_pa/si.p0_pa
        return (r <= crit), f"pb/p0={r:.3f} <= {crit:.3f} (estrangulado)"
    def compute(self, s: ProcessConditions, m: MixtureProps):
        ok, reason = self.check_applicability(s, m)
        if not ok: return EquationResult(self.name, False, reason)
        si = normalize(s, m)
        k = m.k; A = si.A_orif_m2; P0 = si.p0_pa; T0 = si.t0_K; Cd = 0.61
        Rm = R_UNIV/(m.MW_mix/1000.0)
        qm = A * Cd * P0 * sqrt( k/(Rm*T0) * (2.0/(k+1.0))**((k+1.0)/(k-1.0)) )
        audit = audit_text_header() + f"""
A = {A:.6e} [m²], P0 = {P0:.6e} [Pa], T = {T0:.2f} K
q_m = A·P0·√( k/(R_m·T) · (2/(k+1))^{(k+1)/(k-1)} ) = {qm:.6e} kg/s
"""
        return EquationResult(self.name, True, reason, qm, extras={"beta": si.beta, "audit": audit})

def run_all_models(pc: ProcessConditions, props: MixtureProps):
    models = [AsmeMFC3M(), AGA31(), ANPModel(), YuHu1(), YuHu2(), Montiel(), Cengel()]
    out = []
    for m in models:
        try:
            out.append(m.compute(pc, props))
        except Exception as e:
            out.append(EquationResult(m.name, False, f"Erro: {e}"))
    return out
