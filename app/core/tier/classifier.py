
from dataclasses import dataclass

def classify_general(massa_kg: float) -> str:
    if massa_kg > 500: return "TIER 1"
    elif massa_kg > 50: return "TIER 2"
    else: return "TIER 3"

def classify_enclosed(massa_kg: float) -> str:
    if massa_kg > 50: return "TIER 1"
    elif massa_kg > 25: return "TIER 2"
    else: return "TIER 3"
