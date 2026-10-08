"""Python replacements for the MATLAB training/inference master scripts."""
from pathlib import Path
import joblib
from .common import new_directory,read_table,write_csv,write_json
from .config import FeatureConfig
from .io import extract_recordings,recording_manifest
from .clustering import cluster_calls
from .splits import assert_split_integrity


def prepare_training(recordings,out_dir,config=None,columns=None,k=None,k_range=range(2,21),group_col="AnimalID",seed=7,cache_dir=None):
    """Extract all candidates; taxonomy sees fitting groups only; export review queue."""
    manifest=recording_manifest(recordings);assert_split_integrity(manifest,group_col,required=("train","calibration","test"))
    out=new_directory(out_dir);write_csv(manifest,out/"recordings.csv")
    extract_recordings(manifest,out/"calls_features.csv",config or FeatureConfig(),cache_dir)
    calls=read_table(out/"calls_features.csv");training=calls[calls.Split.eq("train")].copy()
    # Extraction diagnostics are durable even if clustering fails.
    result,model,summary=cluster_calls(training,columns,k,k_range,seed)
    write_csv(result,out/"calls_clustered.csv");write_csv(result[result.ClusterStatus.eq("UNCERTAIN")],out/"calls_uncertain.csv")
    write_csv(summary,out/"kmeans_summaries.csv");joblib.dump(model,out/"clustering.joblib")
    review=calls[["RecordingID","CallID","WavFile","Start_s","End_s",group_col,"Split"]].copy()
    labels=result.set_index("CallID").Cluster
    review["SuggestedCluster"]=review.CallID.map(labels).fillna("")
    for col in ("ExpertFinalLabel","ExpertBinaryLabel","ExpertNotes"):review[col]=""
    write_csv(review,out/"expert_review_template.csv")
    write_json(out/"workflow_status.json",dict(State="AWAITING_EXPERT_LABELS",SuggestedClustersFor="train_only",ReviewIdentity=["RecordingID","CallID"],
                                               Next="Fill expert_review_template.csv, then call train_and_classify with table and review in a new output directory"))
    return out


def master_inference(recordings,bundle,out_dir,cache_dir=None,device="cpu"):
    from .pipeline import load_bundle,classify_calls
    from .review import export_expert_review_queue
    loaded=load_bundle(bundle,device);config=FeatureConfig(**loaded[0]["feature_config"])
    out=new_directory(out_dir)
    extract_recordings(recordings,out/"calls_features.csv",config,cache_dir)
    prediction=classify_calls(out/"calls_features.csv",loaded,device)
    write_csv(prediction,out/"predictions.csv");write_csv(export_expert_review_queue(prediction),out/"expert_review_queue.csv")
    write_json(out/"inference_manifest.json",dict(ModelBundleID=loaded[0]["bundle_id"],NCalls=len(prediction)))
    return prediction

