import numpy as np
import pandas as pd
from .common import bool_values, read_table, require_columns
from .io import validate_identity
from .labels import canonical_label


def export_expert_review_queue(table,audit_fraction=.05,max_audit_per_class=50,seed=7):
    t=read_table(table);validate_identity(t);require_columns(t,["ClusterPred","Stage1","IsUncertain"])
    if not 0<=audit_fraction<=1 or max_audit_per_class<0:raise ValueError("Invalid audit settings")
    uncertain=bool_values(t.IsUncertain)|t.ClusterPred.isin(["PROCESSING_ERROR","UNCERTAIN"]).to_numpy()
    selected=list(np.flatnonzero(uncertain));rng=np.random.default_rng(seed)
    if audit_fraction>0:
        labels=np.where(t.Stage1.eq("NOISE"),"NOISE",t.ClusterPred.astype(str))
        for c in sorted(set(labels[~uncertain])):
            idx=np.flatnonzero((labels==c)&~uncertain);n=min(len(idx),max_audit_per_class,max(1,int(np.ceil(audit_fraction*len(idx)))))
            selected.extend(rng.choice(idx,n,replace=False))
    q=t.iloc[sorted(set(selected))].copy()
    is_unc=uncertain[sorted(set(selected))]
    error=q.ClusterPred.eq("PROCESSING_ERROR").to_numpy()
    q["ReviewTier"]=np.where(error,0,np.where(is_unc,1,2));q["ReviewSource"]=np.where(error,"PROCESSING_ERROR",np.where(is_unc,"UNCERTAIN","CONFIDENT_AUDIT"))
    priority=np.zeros(len(q))
    for c in ("RFMaxProbability","RFMargin"):
        if c in q:priority+=1-np.nan_to_num(pd.to_numeric(q[c],errors="coerce").to_numpy(float),nan=0,posinf=0,neginf=0)
    q["ReviewPriority"]=priority
    for c in ("ExpertFinalLabel","ExpertBinaryLabel","ExpertDecision","ExpertNotes"):q[c]=""
    return q.sort_values(["ReviewTier","ReviewPriority","CallID"],ascending=[True,False,True],kind="stable").reset_index(drop=True)


def apply_expert_review(table,review):
    t,r=read_table(table),read_table(review);validate_identity(t);validate_identity(r)
    require_columns(r,["ExpertFinalLabel"])
    index=t.set_index("CallID",drop=False)
    if not set(r.CallID)<=set(index.index):raise ValueError("Unmatched review CallID")
    # Validate every key and source interval before applying any decisions.
    for row in r.to_dict("records"):
        original=index.loc[row["CallID"]]
        if str(original.RecordingID)!=str(row["RecordingID"]):raise ValueError("Review RecordingID mismatch")
        for c in ("Start_s","End_s"):
            if c in row and not np.isclose(float(original[c]),float(row[c]),rtol=0,atol=1e-9):raise ValueError("Review interval mismatch")
    if "FinalLabel" not in index:index["FinalLabel"]=index["ClusterPred"].astype(str) if "ClusterPred" in index else ""
    if "Cluster_Clean" not in index:index["Cluster_Clean"]=""
    index["Cluster_Clean"]=index.Cluster_Clean.fillna("").astype(str)
    if "ExpertReviewed" not in index:index["ExpertReviewed"]=False
    else:index["ExpertReviewed"]=bool_values(index.ExpertReviewed)
    for c in ("ExpertBinaryLabel","ExpertNotes","ExpertDecision"):
        if c not in index:index[c]=""
        else:index[c]=index[c].fillna("").astype(str)
    for row in r.to_dict("records"):
        key=row["CallID"];raw="" if pd.isna(row["ExpertFinalLabel"]) else str(row["ExpertFinalLabel"]).strip()
        binary="" if pd.isna(row.get("ExpertBinaryLabel","")) else str(row.get("ExpertBinaryLabel","")).strip().upper()
        if not raw and not binary:continue
        if binary not in ("","USV","NOISE"):raise ValueError("Invalid expert binary label")
        label=canonical_label(raw)
        if label=="0":inferred="NOISE"
        elif label:inferred="USV"
        else:inferred=""
        if binary and inferred and binary!=inferred:raise ValueError("Expert binary/subtype conflict")
        index.at[key,"FinalLabel"]=raw or binary
        index.at[key,"Cluster_Clean"]=label
        index.at[key,"ExpertBinaryLabel"]=binary or inferred
        index.at[key,"ExpertReviewed"]=True
        for c in ("ExpertNotes","ExpertDecision"):
            if c in row and not pd.isna(row[c]):index.at[key,c]=str(row[c])
    return index.loc[t.CallID].reset_index(drop=True)


def review_clusters(table,mapping,cluster_col="Cluster"):
    """Explicit expert mapping: cluster -> subtype or NOISE; unmapped stays unreviewed."""
    t=read_table(table);require_columns(t,[cluster_col]);validate_identity(t)
    mapping={canonical_label(k):v for k,v in mapping.items()}
    known=set(t[cluster_col].map(canonical_label))
    if not set(mapping)<=known:raise ValueError("Mapping references an absent cluster")
    review=t[["RecordingID","CallID","Start_s","End_s"]].copy()
    review["ExpertFinalLabel"]=t[cluster_col].map(canonical_label).map(mapping).fillna("")
    return apply_expert_review(t,review)


def interactive_review(table):
    t=read_table(table);print(t.groupby("Cluster",dropna=False).size().to_string())
    mapping={}
    for c in sorted(t.Cluster.dropna().astype(str).unique()):
        if not c:continue
        answer=input(f"Cluster {c}: subtype name / NOISE / UNCERTAIN / Enter to skip: ").strip()
        if answer:mapping[c]=answer
    return review_clusters(t,mapping)
