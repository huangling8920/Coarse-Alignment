from __future__ import annotations

import pandas as pd

from src.vio.feature_quality import degradation_profile


def visual_degradation_table() -> pd.DataFrame:
    rows = []
    for level in ["normal", "weak_snow", "strong_reflection", "motion_blur"]:
        prof = degradation_profile(level)
        rows.append({"level": level, **prof})
    return pd.DataFrame(rows)

