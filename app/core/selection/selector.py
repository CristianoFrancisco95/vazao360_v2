
from dataclasses import dataclass
from ..thermo.mixture_props import MixtureProps
from ..models.inputs import ProcessConditions

@dataclass
class Applicability:
    choked: bool
    reason: str

def _critical_pressure_ratio(k: float) -> float:
    return (2/(k+1))**(k/(k-1))

def evaluate_applicability(props: MixtureProps, pc: ProcessConditions) -> Applicability:
    p0 = pc.p0_bar * 1e5
    pb = pc.pback_bar * 1e5
    k = props.k
    crit = _critical_pressure_ratio(k)
    ratio = pb / p0
    choked = ratio <= crit
    reason = f"pb/p0={ratio:.3f} {'<=' if choked else '>'} razão crítica {crit:.3f} para k={k:.3f}"
    return Applicability(choked=choked, reason=reason)
