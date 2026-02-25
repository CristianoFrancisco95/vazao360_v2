
from dataclasses import dataclass
from typing import Optional

@dataclass
class EquationResult:
    name: str
    applicable: bool
    reason: str
    mass_flow_kg_s: Optional[float] = None
    notes: str = ""
    extras: dict | None = None

class BaseEquation:
    name = "Base"
    def check_applicability(self, scenario, props) -> tuple[bool, str]: ...
    def compute(self, scenario, props) -> EquationResult: ...
