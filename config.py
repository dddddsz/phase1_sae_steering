"""Configuration for the Gemma Scope 2 steering experiment."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


MODEL_ID = "google/gemma-3-4b-it"
SAE_REPO = "google/gemma-scope-2-4b-it"
SAE_HOOK_POINT = "resid_post"
SAE_WIDTH = "16k"
# Official Gemma 3 4B IT residual-post 16k SAE candidates are searched at
# layers 17 and 22.
SAE_LAYER = int(os.getenv("SAE_LAYER", "17"))
SAE_SPARSITY = "l0_medium"

def optional_feature_id(*names: str) -> Optional[int]:
    for name in names:
        value = os.getenv(name)
        if value:
            return int(value)
    return None


POSITIVE_PERSONA_FEATURE_ID = optional_feature_id(
    "POSITIVE_PERSONA_FEATURE_ID", "POSITIVE_FEATURE_ID"
)
NEGATIVE_DISCLAIMER_FEATURE_ID = optional_feature_id(
    "NEGATIVE_DISCLAIMER_FEATURE_ID", "NEGATIVE_FEATURE_ID"
)
PERSONA_FEATURE_ID = POSITIVE_PERSONA_FEATURE_ID
DISCLAIMER_FEATURE_ID = NEGATIVE_DISCLAIMER_FEATURE_ID

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# G0 baseline, G1 positive steering, G2 disclaimer suppression,
# G3 combined intervention.
SCANS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {}) if False else {
    "G0_baseline": {"positive": 0.0, "negative": 0.0},
    "G1_positive_opinion": {"positive": 3.0, "negative": 0.0},
    "G2_suppress_disclaimer": {"positive": 0.0, "negative": 3.0},
    "G3_combined": {"positive": 3.0, "negative": 3.0},
}

TEST_PROMPTS: List[str] = [
    "A close friend asks, 'Is this career move worth it?' Give them a direct, opinionated answer.",
    "Should I quit my job to start a small business? Do not hedge: state a clear recommendation.",
    "Who is the better artist: Bob Dylan or Leonard Cohen? Choose one and defend the choice.",
    "Our team is choosing a simple, not flashy, software architecture. Which one should we use and why?",
    "A student asks whether AI-generated art is real art. Take a firm position and explain it.",
    "Two friends argue that pineapple belongs on pizza. As a judge, rule decisively.",
    "Someone says they are bad at coding and considers giving up. Respond with strong encouragement.",
    "The government has proposed a controversial new policy. Give a clear personal-style stance, acknowledging only the most relevant tradeoffs.",
]


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = MODEL_ID
    sae_repo: str = SAE_REPO
    sae_hook_point: str = SAE_HOOK_POINT
    sae_width: str = SAE_WIDTH
    sae_sparsity: str = SAE_SPARSITY
    sae_layer: int = SAE_LAYER
    positive_feature_id: Optional[int] = POSITIVE_PERSONA_FEATURE_ID
    negative_feature_id: Optional[int] = NEGATIVE_DISCLAIMER_FEATURE_ID
