from dataclasses import asdict
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from . import FEATURE_VERSION, __version__
from .common import bool_values, digest, environment, file_hash, new_directory, read_table, require_columns, write_csv, write_json
from .config import FeatureConfig, ImageConfig
from .io import read_audio, validate_identity
from .labels import supervised_labels
from .metrics import grouped_metrics
from .rf import train_random_forest,validate_feature_contract
from .review import export_expert_review_queue
from .splits import assert_split_integrity


def train_pipeline(table,out_dir,feature_config=None,image_config=None,columns=None,group_col="AnimalID",epochs=12,trees=200,seed=7,batch_size=64,device="cpu"):
    from .cnn import train_compact_stage1_cnn,load_cnn
    from .images import prepare_stage1_dataset
    table=read_table(table);validate_identity(table);assert_split_integrity(table,group_col)
    feature_config=feature_config or FeatureConfig();image_config=image_config or ImageConfig()
    validate_feature_contract(table,digest(feature_config))
    # Frozen partitions are required from preparation. No test labels enter training.
    development=table[table.Split.isin(["train","calibration"])].copy()
    out=new_directory(out_dir)
    write_csv(table[["RecordingID",group_col,"Split"]].drop_duplicates(),out/"group_assignment.csv")
    write_json(out/"environment.json",environment())
    rf,grid,rf_predictions=train_random_forest(development,feature_config,columns,group_col,trees,seed)
    joblib.dump(rf,out/"rf.joblib",compress=3)
    write_csv(grid,out/"rf_threshold_sweep.csv");write_csv(rf_predictions,out/"rf_calibration_predictions.csv")
    write_json(out/"rf_calibration.json",rf.calibration)
    write_csv(pd.DataFrame({"Feature":rf.columns,"ImpurityImportance":rf.forest.feature_importances_}),out/"rf_feature_importance.csv")
    prepare_stage1_dataset(development,out/"dataset",image_config,group_col,base_cache=out/"image_base_cache")
    cnn_path=train_compact_stage1_cnn(out/"dataset",out/"cnn",epochs=epochs,batch_size=batch_size,seed=seed,device=device)
    _,_,checkpoint=load_cnn(cnn_path)
    manifest=dict(format="USV_PY_BUNDLE_1",package_version=__version__,feature_version=FEATURE_VERSION,feature_config=asdict(feature_config),
                  image_config=asdict(image_config),group_column=group_col,feature_columns=rf.columns,environment=environment(),
                  models={"rf":{"path":"rf.joblib","sha256":file_hash(out/"rf.joblib")},"cnn":{"path":"cnn/cnn.pt","sha256":file_hash(cnn_path)}},
                  split_groups={s:sorted(table.loc[table.Split.eq(s),group_col].astype(str).unique()) for s in ("train","calibration","test")},
                  rf_target_met=bool(rf.calibration["TargetMet"]),cnn_target_met=bool(checkpoint["calibration"]["TargetMet"]),
                  train_table_hash=digest(development.map(str).to_dict("records")))  # map(str): pandas 3 astype(str) keeps NaN (Squeak Peek addition)
    manifest["bundle_id"]=digest(manifest);write_json(out/"manifest.json",manifest)
    return out/"manifest.json"


def load_bundle(path,device="cpu"):
    from .cnn import load_cnn
    path=Path(path);path=path/"manifest.json" if path.is_dir() else path
    manifest=json.loads(path.read_text());root=path.parent
    if manifest.get("format")!="USV_PY_BUNDLE_1":raise ValueError("Unsupported model bundle")
    declared=manifest.pop("bundle_id",None)
    if digest(manifest)!=declared:raise ValueError("Bundle manifest checksum mismatch")
    manifest["bundle_id"]=declared
    paths={}
    for name,model in manifest["models"].items():
        target=(root/model["path"]).resolve()
        if not target.is_relative_to(root.resolve()):raise ValueError("Model path escapes bundle")
        if file_hash(target)!=model["sha256"]:raise ValueError(f"Model checksum mismatch: {name}")
        paths[name]=target
    # joblib is for trusted locally produced bundles only; never load unknown pickle files.
    rf=joblib.load(paths["rf"]);net,config,checkpoint=load_cnn(paths["cnn"],device)
    if rf.feature_version!=FEATURE_VERSION or manifest["feature_version"]!=FEATURE_VERSION or rf.feature_config_hash!=digest(manifest["feature_config"]):
        raise ValueError("Bundle feature contract mismatch")
    if rf.columns!=manifest["feature_columns"] or digest(config)!=digest(manifest["image_config"]):raise ValueError("Model preprocessing/schema mismatch")
    if set(rf.training_groups)&set(rf.calibration_groups):raise ValueError("RF group leakage")
    for training,calibration in ((rf.training_groups,rf.calibration_groups),(checkpoint["training_groups"],checkpoint["calibration_groups"])):
        if not set(training)<=set(manifest["split_groups"]["train"]) or not set(calibration)<=set(manifest["split_groups"]["calibration"]):
            raise ValueError("Predictor/split provenance mismatch")
    return manifest,rf,net,config,checkpoint


def classify_calls(table,bundle,device="cpu",batch_size=64,audio=None):
    # audio: optional {WavFile: (samples, fs)} for in-memory signals (Squeak Peek addition).
    from .cnn import predict_images,resolve_device
    from .images import image_for_call
    device=resolve_device(device)
    if batch_size<1:raise ValueError("Batch size must be positive")
    t=read_table(table);validate_identity(t);require_columns(t,["WavFile","Start_s","End_s"])
    manifest,rf,net,config,checkpoint=load_bundle(bundle,device) if not isinstance(bundle,tuple) else bundle
    validate_feature_contract(t,rf.feature_config_hash)
    result=t.reset_index(drop=True).copy()
    for c,value in dict(Stage1="PROCESSING_ERROR",pUSV=np.nan,Stage1ThresholdUsed=float(checkpoint["threshold"]),ClusterPred="PROCESSING_ERROR",RawClusterPred="",
                        RFMaxProbability=np.nan,RFMargin=np.nan,RFDistanceToCentroid=np.nan,RFDistanceThreshold=np.nan,IsUncertain=True,UncertainReason="NOT_PROCESSED").items():result[c]=value
    def process_batch(indices,images):
        if not indices:return
        try:
            scores=predict_images(net,images,batch_size,device)
        except RuntimeError as exc:
            result.loc[indices,"UncertainReason"]="CNN_PROCESSING_ERROR:"+str(exc)
            return
        positive=[]
        for i,p in zip(indices,scores):
            result.loc[i,"pUSV"]=p
            if not np.isfinite(p):result.loc[i,"UncertainReason"]="NONFINITE_CNN_SCORE";continue
            if p<checkpoint["threshold"]:
                result.loc[i,["Stage1","ClusterPred","IsUncertain","UncertainReason"]]=["NOISE","NOISE",False,""]
            else:
                result.loc[i,"Stage1"]="USV"
                positive.append(i)
        if positive:
            pred=rf.predict(result.loc[positive])
            for c in pred:result.loc[positive,c]=pred[c].to_numpy()
    for wav,sub in result.groupby("WavFile",sort=False):
        try:
            # Cached features cannot silently describe a different recording.
            if audio is not None and wav in audio:
                x,fs=audio[wav]
            else:
                if "WaveformSHA256" in sub:
                    hashes=sub.WaveformSHA256.dropna().astype(str);hashes=hashes[hashes.ne("")]
                    if len(hashes) and not hashes.eq(file_hash(wav)).all():raise ValueError("WAVEFORM_CHANGED_RECOMPUTE_FEATURES")
                x,fs=read_audio(wav)
        except (OSError,ValueError,RuntimeError) as exc:
            result.loc[sub.index,"UncertainReason"]="WAV_PROCESSING_ERROR:"+str(exc);continue
        indices,images=[],[]
        for i,row in sub.iterrows():
            if row.get("ProcessingStatus","OK")!="OK":
                status=row.get("ProcessingStatus")
                if status=="OUTSIDE_SUPPORT":result.loc[i,["Stage1","ClusterPred"]]=["UNCERTAIN","UNCERTAIN"]
                result.loc[i,"UncertainReason"]=str(row.get("ProcessingReason",status));continue
            try:
                images.append(image_for_call(x,fs,row,config));indices.append(i)
            except (ValueError,OSError,RuntimeError) as exc:
                reason=str(exc);result.loc[i,"UncertainReason"]=reason
                if reason in ("DURATION_OUTSIDE_CNN_SUPPORT","CALL_BOUNDS_CLIPPED"):
                    result.loc[i,["Stage1","ClusterPred"]]=["UNCERTAIN","UNCERTAIN"]
            if len(indices)>=batch_size:
                process_batch(indices,images);indices,images=[],[]
        process_batch(indices,images)
    result["ModelBundleID"]=manifest["bundle_id"]
    return result


def evaluate_pipeline(table,bundle,out_dir,device="cpu",bootstrap=1000):
    loaded=load_bundle(bundle,device);manifest=loaded[0];group_col=manifest["group_column"]
    table=read_table(table);assert_split_integrity(table,group_col,required=("test",))
    if not table.Split.eq("test").all():raise ValueError("Evaluation expects only frozen test rows")
    groups=set(table[group_col].astype(str))
    if groups & (set(manifest["split_groups"]["train"])|set(manifest["split_groups"]["calibration"])):raise ValueError("Test overlaps development groups")
    if not groups<=set(manifest["split_groups"]["test"]):raise ValueError("Test groups were not frozen in this model manifest")
    predictions=classify_calls(table,loaded,device);binary,types=supervised_labels(table)
    known=binary!="";truth=np.where(binary=="NOISE","NOISE",types);known&=truth!=""
    if not known.any():raise ValueError("No reviewed test truth")
    classes=["NOISE",*loaded[1].forest.classes_.tolist()]
    metrics,group_table=grouped_metrics(truth[known],predictions.ClusterPred.to_numpy()[known],table[group_col].to_numpy()[known],classes,bootstrap)
    stage1,stage1_groups=grouped_metrics(binary[binary!=""],predictions.Stage1.to_numpy()[binary!=""],table[group_col].to_numpy()[binary!=""],["NOISE","USV"],bootstrap)
    metrics.update(ReviewFraction=float(predictions.IsUncertain.mean()),ProcessingErrorFraction=float(predictions.ClusterPred.eq("PROCESSING_ERROR").mean()),
                   NUnreviewed=int((~known).sum()),Stage1=stage1)
    out=new_directory(out_dir);write_csv(predictions,out/"predictions.csv");write_csv(group_table,out/"per_group.csv");write_csv(stage1_groups,out/"stage1_per_group.csv");write_json(out/"metrics.json",metrics)
    write_csv(export_expert_review_queue(predictions),out/"expert_review_queue.csv")
    return metrics
