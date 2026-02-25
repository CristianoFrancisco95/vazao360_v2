
from pydantic import BaseModel, field_validator
from typing import Dict

COMPONENT_MAP = {
    "C1 - Metano": "CH4", "C2 - Etano": "C2H6", "C3 - Propano": "C3H8", "iC4 - i-Butano": "iC4H10", "nC4 - n-Butano": "nC4H10", "iC5 - i-Pentano": "iC5H12", "nC5 - n-Pentano": "nC5H12", "nC6 - n-Hexano": "nC6H14", "nC7 - n-Heptano": "nC7H16", "nC8 - n-Octano": "nC8H18", "nC9 - n-Nonano": "nC9H20", "nC10+ - n-Decano+": "nC10plus", "CO2 - Dióxido de Carbono": "CO2", "O2 - Oxigênio": "O2", "N2 - Nitrogênio": "N2", "H2O - Água": "H2O", "H2S - Ácido Sulfídrico": "H2S"
}

class Composition(BaseModel):
    fractions: Dict[str, float]

    @field_validator("fractions")
    @classmethod
    def validate_sum(cls, v: Dict[str, float]):
        total = sum(v.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"A soma das frações deve ser 1.0 (+/-0.001). Atual: {total:.5f}")
        for k,val in v.items():
            if val < 0 or val > 1:
                raise ValueError(f"Frações ∈ [0,1]. Problema em {k}={val}")
        return v

    def to_catalog_keys(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for ui_key, x in self.fractions.items():
            key = COMPONENT_MAP.get(ui_key, ui_key)
            out[key] = out.get(key, 0.0) + x
        return out
