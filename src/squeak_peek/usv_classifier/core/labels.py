import numpy as np
import pandas as pd
from .common import bool_values

RESERVED = {"", "NAN", "NA", "N/A", "NONE", "NULL", "<NA>", "UNCERTAIN", "UNREVIEWED", "PROCESSING_ERROR", "ERROR", "UNKNOWN"}


def canonical_label(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.upper() in RESERVED:
        return ""
    if text.upper() == "NOISE":
        return "0"
    if text.upper() == "USV":
        return "USV"
    try:
        number = float(text)
        if not np.isfinite(number) or number < 0:
            return ""
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def supervised_labels(table, label_col="Cluster_Clean", require_reviewed=True):
    """Independent binary truth permits valid USV with an unreadable contour."""
    subtype = table[label_col].map(canonical_label) if label_col in table else pd.Series("",index=table.index)
    binary = pd.Series("", index=table.index, dtype=object)
    binary.loc[subtype.eq("0")] = "NOISE"
    binary.loc[~subtype.isin(["", "0", "USV"])] = "USV"
    binary.loc[subtype.eq("USV")] = "USV"
    if "ExpertBinaryLabel" in table:
        explicit = table.ExpertBinaryLabel.fillna("").astype(str).str.strip().str.upper()
        if not explicit.isin(["", "NOISE", "USV", *RESERVED]).all():
            raise ValueError("ExpertBinaryLabel must be USV, NOISE or empty")
        use = explicit.isin(["USV", "NOISE"])
        conflict = use & binary.ne("") & binary.ne(explicit)
        if conflict.any():
            raise ValueError("Contradictory expert binary and subtype labels")
        binary.loc[use] = explicit.loc[use]
    if require_reviewed:
        reviewed = bool_values(table.ExpertReviewed) if "ExpertReviewed" in table else np.zeros(len(table), bool)
        subtype.loc[~reviewed],binary.loc[~reviewed] = "",""
    subtype.loc[subtype.isin(["", "0", "USV"])] = ""
    return binary.to_numpy(str), subtype.to_numpy(str)
