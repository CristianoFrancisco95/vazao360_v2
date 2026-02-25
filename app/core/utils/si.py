from dataclasses import dataclass
from ..models.inputs import ProcessConditions
from ..thermo.mixture_props import MixtureProps
R_UNIV = 8.31446261815324  # J/mol-K

@dataclass
class SINorm:
    p0_pa: float
    p2_pa: float
    delta_p_pa: float
    t0_K: float
    d_m: float
    D_m: float
    A_orif_m2: float
    A_pipe_m2: float
    rho1: float      # calculada internamente (Z=1)
    mu: float
    beta: float

def rho_ideal_pa(mw_kg_per_kmol: float, p_pa: float, T_K: float) -> float:
    # Equivalente à forma que você indicou: p * MW / (R*(T)/1000) == p*(MW/1000)/(R*T)
    return (p_pa * (mw_kg_per_kmol/1000.0)) / (R_UNIV * T_K)

def normalize(pc: ProcessConditions, mp: MixtureProps) -> SINorm:
    p0_pa = pc.p0_bar * 1e5
    p2_pa = pc.pback_bar * 1e5
    delta_p_pa = max(0.0, p0_pa - p2_pa)
    t0_K = pc.t0_c + 273.15
    d_m = pc.orifice_d_mm / 1000.0
    D_m = pc.pipe_D_mm / 1000.0
    from math import pi
    A_orif = (pi/4.0) * d_m * d_m
    A_pipe = (pi/4.0) * D_m * D_m
    beta = d_m / D_m if D_m > 0 else 0.0

    # SEMPRE calcular rho1 internamente (Z = 1)
    rho1 = rho_ideal_pa(mp.MW_mix, p0_pa, t0_K)

    return SINorm(p0_pa, p2_pa, delta_p_pa, t0_K, d_m, D_m, A_orif, A_pipe, rho1, pc.mu_pa_s, beta)

def audit_text_header() -> str:
    return (
        "Unidades alvo: q_m[kg/s], d/D[mm]->m internamente, P[Pa], ΔP[Pa], ρ[kg/m³], T[K]\n"
        "Checagem: q_m = A[m²] · √(2ΔP[Pa]·ρ[kg/m³]) = kg/s\n"
    )