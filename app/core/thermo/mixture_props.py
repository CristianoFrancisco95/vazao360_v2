
from dataclasses import dataclass
from typing import Dict
import yaml
from ..models.composition import Composition

R_univ = 8.31446261815324  # J/mol-K

@dataclass
class MixtureProps:
    MW_mix: float           # kg/kmol
    Cp_mass: float      # J/kg-K
    Cv_mass: float      # J/kg-K
    k: float            # Cp/Cv

def load_catalog(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_mixture_props(comp: Composition, catalog_path: str, T_K: float) -> MixtureProps:
    cat = load_catalog(catalog_path)
    y = comp.to_catalog_keys()
    MW_mix = sum(y[sp]*cat[sp]["MW"] for sp in y)
    Cp_molar = sum(y[sp]*cat[sp]["Cp"] for sp in y)  # kJ/kmol-K
    Cp_mass = (Cp_molar*1000.0) / MW_mix
    Rbar_mass = R_univ / (MW_mix/1000.0)
    Cv_mass = Cp_mass - Rbar_mass
    k = Cp_mass / Cv_mass if Cv_mass > 0 else 1.4
    return MixtureProps(MW_mix=MW_mix, Cp_mass=Cp_mass, Cv_mass=Cv_mass, k=k)
