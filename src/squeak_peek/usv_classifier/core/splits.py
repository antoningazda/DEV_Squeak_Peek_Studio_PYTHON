import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .common import require_columns


def assert_split_integrity(table, group_col="AnimalID", required=("train", "calibration")):
    require_columns(table,[group_col,"Split"])
    if table[group_col].isna().any() or table[group_col].astype(str).str.strip().eq("").any():
        raise ValueError(f"Missing {group_col}; recording groups must be chosen explicitly")
    if table.Split.isna().any() or not table.Split.isin(["train", "calibration", "test"]).all():
        raise ValueError("Split must be train, calibration or test")
    if (table.groupby(group_col).Split.nunique()>1).any():
        raise ValueError("Group leakage across splits")
    for col in ("RecordingID", "WavFile", "WaveformSHA256"):
        if col in table:
            valid = table[col].notna() & table[col].astype(str).ne("")
            if (table.loc[valid].groupby(col).Split.nunique()>1).any():
                raise ValueError(f"{col} crosses split boundary")
    if not set(required)<=set(table.Split):
        raise ValueError(f"Required splits absent: {set(required)-set(table.Split)}")


def freeze_splits(recordings, group_col="AnimalID", test_fraction=.2, calibration_fraction=.2, seed=7):
    """Split before inspecting features/labels; fail later on insufficient class support."""
    t=recordings.copy()
    require_columns(t,[group_col])
    if "Split" in t:
        assert_split_integrity(t,group_col,("train","calibration","test"))
        return t
    if t[group_col].isna().any() or t[group_col].astype(str).str.strip().eq("").any():
        raise ValueError("Missing grouping identifiers")
    if min(test_fraction,calibration_fraction)<=0 or test_fraction+calibration_fraction>=1:
        raise ValueError("Invalid partition fractions")
    groups=np.sort(t[group_col].astype(str).unique())
    if len(groups)<5:raise ValueError("At least five independent groups needed for three-way partition")
    groups=np.random.default_rng(seed).permutation(groups)
    n_test=max(1,int(np.floor(len(groups)*test_fraction+.5)))
    n_cal=max(1,int(np.floor(len(groups)*calibration_fraction+.5)))
    if n_test+n_cal>=len(groups):raise ValueError("Partition leaves no training groups")
    mapping={g:"test" if i<n_test else "calibration" if i<n_test+n_cal else "train" for i,g in enumerate(groups)}
    t["Split"]=t[group_col].astype(str).map(mapping)
    assert_split_integrity(t,group_col,("train","calibration","test"))
    return t


def class_coverage(y_train,y_other,classes=None):
    classes=set(classes if classes is not None else np.r_[y_train,y_other])
    if set(y_train)!=classes or set(y_other)!=classes:
        raise ValueError(f"Class coverage failure: train missing {classes-set(y_train)}, validation/calibration missing {classes-set(y_other)}")


def make_group_folds(groups,labels,k=5,seed=7):
    groups,labels=np.asarray(groups,str),np.asarray(labels,str)
    if len(groups)!=len(labels) or not len(groups) or "" in groups:
        raise ValueError("Invalid groups/labels")
    if k<2 or len(set(groups))<k:raise ValueError("Insufficient groups for requested folds")
    for c in set(labels):
        if len(set(groups[labels==c]))<k:
            raise ValueError(f"Class {c} occurs in fewer than {k} independent groups")
    # SGKF is a heuristic; accept only assignments with explicit class coverage.
    for attempt in range(64):
        splitter=StratifiedGroupKFold(k,shuffle=True,random_state=seed+attempt)
        folds=np.full(len(groups),-1,int)
        valid=True
        for fold,(tr,te) in enumerate(splitter.split(np.zeros(len(labels)),labels,groups)):
            if set(labels[tr])!=set(labels) or set(labels[te])!=set(labels):
                valid=False;break
            folds[te]=fold
        if valid:
            assignment=pd.DataFrame({"Group":groups,"Fold":folds}).drop_duplicates().sort_values("Group")
            return folds,assignment
    raise ValueError("Could not create group folds with all classes; use fewer folds or collect more groups")
