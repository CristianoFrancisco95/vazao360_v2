from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, List

class TransientInterval(BaseModel):
    """Um intervalo de tempo com pressão a montante constante (regime transiente)."""
    dt_min: float = Field(gt=0, description="Duração do intervalo (min)")
    p0_bar: float = Field(gt=0, description="Pressão a montante no intervalo (bar abs)")


class ProcessConditions(BaseModel):
    p0_bar: float = Field(gt=0, description="Pressão a montante (absoluta)")
    pback_bar: float = Field(gt=0, description="Pressão a jusante/ambiente (absoluta)")
    t0_c: float = Field(gt=-273.15, description="Temperatura °C")
    orifice_d_mm: float = Field(gt=0, description="Diâmetro do orifício (mm)")
    pipe_D_mm: float = Field(gt=0, description="Diâmetro interno da tubulação (mm)")
    duration_s: Optional[float] = Field(default=None, ge=0, description="Duração do evento (s)")
    ambiente_fechado: bool = False

    # NOVO: densidade CNTP (apenas para cálculo de volume na Aba 03)
    rho_cntp: float = Field(gt=0, description="Densidade em CNTP (kg/m³)")

    mu_pa_s: float = Field(gt=0, description="Viscosidade dinâmica (Pa·s)")

    tap_type: Literal["flange", "corner", "D_D/2"] = "flange"
    L1_m: float = Field(default=0.0254, ge=0, description="Distância a montante do orifício até a tomada (m)")

    # Regime de operação
    regime: Literal["Contínuo", "Transiente"] = "Contínuo"
    # Tabela de intervalos (apenas para regime Transiente)
    transient_table: Optional[List[TransientInterval]] = None

    @field_validator("p0_bar", "pback_bar")
    @classmethod
    def check_pressures(cls, v):
        if v <= 0:
            raise ValueError("Pressões devem ser absolutas e > 0.")
        return v