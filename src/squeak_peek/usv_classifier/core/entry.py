"""Entry points of the standalone tool, ported from its two scripts.

train_and_classify() is trenink.py's function and classify() is
klasifikace.py's, kept verbatim apart from package-relative imports (their
argparse wrappers are replaced by the app's Classification tab and CLI).
"""
import json
from pathlib import Path


def classify(recordings, model, output, *, cache=None, device="cpu"):
    """Extract model-compatible features and return all candidate predictions."""
    from .workflows import master_inference
    return master_inference(recordings,model,output,cache,device)


def train_and_classify(output, *, recordings=None, table=None, review=None,
                       classify_recordings=None, group_column="GroupID",
                       feature_set="Compact", epochs=12, trees=200, seed=7,
                       batch_size=64, k=None, cache=None, device="cpu",
                       feature_config=None, image_config=None):
    """Return state/artifact paths; fitting and calibration never use test rows.

    Recordings, features and review accept a CSV path or pandas.DataFrame.
    Relative WAV paths in CSV are resolved against that CSV, and in DataFrames
    against cwd. output must be a new directory. Exceptions propagate to caller.
    """
    import pandas as pd
    from .common import new_directory,read_table,write_csv,write_json,digest
    from .config import FeatureConfig,ImageConfig
    from .io import recording_manifest
    from .splits import freeze_splits,assert_split_integrity
    from .legacy import connected_pair_groups
    from .schema import feature_sets
    from .workflows import prepare_training,master_inference
    from .review import apply_expert_review,export_expert_review_queue
    from .labels import supervised_labels
    from .pipeline import train_pipeline,classify_calls,evaluate_pipeline

    if (recordings is None)==(table is None):
        raise ValueError("Zadejte právě jeden zdroj: recordings nebo table")
    columns=feature_sets()[feature_set]
    for name,value in (("epochs",epochs),("trees",trees),("batch_size",batch_size)):
        if type(value) is not int or value<1:raise ValueError(f"{name} musí být kladné celé číslo")
    if type(seed) is not int or seed<0:raise ValueError("seed musí být nezáporné celé číslo")
    if k is not None and (type(k) is not int or k<2):raise ValueError("k musí být celé číslo alespoň 2")
    feature_config=feature_config or FeatureConfig()
    image_config=image_config or ImageConfig()

    def groups(t):
        if "AnimalMembers" in t:
            if group_column!="GroupID":raise ValueError("AnimalMembers vyžaduje group_column='GroupID'")
            return connected_pair_groups(t)
        return t

    out=new_directory(Path(output).resolve())
    artifacts={}
    if recordings is not None:
        manifest=freeze_splits(groups(recording_manifest(recordings)),group_column,seed=seed)
        prepare_training(manifest,out/"preparation",config=feature_config,columns=columns,
                         k=k,group_col=group_column,seed=seed,cache_dir=cache)
        table=out/"preparation/calls_features.csv"
        artifacts.update(features_table=str(table),review_template=str(out/"preparation/expert_review_template.csv"))
    root=Path.cwd() if isinstance(table,pd.DataFrame) else Path(table).resolve().parent
    calls=read_table(table)
    calls["WavFile"]=[str((root/str(path)).resolve()) for path in calls.WavFile]
    calls=groups(calls)
    assert_split_integrity(calls,group_column,("train","calibration","test"))
    if review is not None:calls=apply_expert_review(calls,review)
    binary,_=supervised_labels(calls)
    if not (binary!="").any():
        if recordings is None:raise ValueError("Chybí expertní štítky pro učení; zadejte review")
        return dict(status="AWAITING_REVIEW",artifacts=artifacts)
    write_csv(calls,out/"reviewed_calls.csv")
    model=train_pipeline(calls,out/"model",feature_config=feature_config,image_config=image_config,
                         columns=columns,group_col=group_column,epochs=epochs,trees=trees,
                         seed=seed,batch_size=batch_size,device=device)
    if "AnnotationSource" in calls and calls.AnnotationSource.fillna("").str.startswith("synthetic_").all():
        manifest=json.loads(model.read_text());manifest.pop("bundle_id")
        manifest["demo_only"]=True;manifest["bundle_id"]=digest(manifest);write_json(model,manifest)
    artifacts["model"]=str(model)
    if classify_recordings is not None:
        predictions=master_inference(classify_recordings,model,out/"classification",cache,device)
    else:
        test=calls[calls.Split.eq("test")].copy()
        predictions=classify_calls(test,model,device,batch_size)
        write_csv(predictions,out/"classification/predictions.csv")
        write_csv(export_expert_review_queue(predictions),out/"classification/expert_review_queue.csv")
        truth,_=supervised_labels(test)
        if (truth!="").any():
            evaluate_pipeline(test,model,out/"evaluation",device)
            artifacts["metrics"]=str(out/"evaluation/metrics.json")
    artifacts.update(predictions=str(out/"classification/predictions.csv"),
                     review_queue=str(out/"classification/expert_review_queue.csv"))
    return dict(status="COMPLETED",artifacts=artifacts,n_calls=len(predictions))
