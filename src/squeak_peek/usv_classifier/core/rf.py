from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from . import FEATURE_VERSION
from .common import bool_values, digest, require_columns
from .config import FeatureConfig
from .labels import supervised_labels
from .metrics import classification_metrics
from .schema import feature_schema, matrix, requires_contour
from .splits import assert_split_integrity, class_coverage


def learn_imputation(x):
    valid=np.isfinite(x)
    if not valid.any(axis=0).all():raise ValueError("All-missing required predictor in training data")
    return np.nanmedian(np.where(valid,x,np.nan),axis=0)


def impute(x,median):
    return np.where(np.isfinite(x),x,median)


def make_forest(trees=200,seed=7,n_jobs=1,min_leaf=1):
    return RandomForestClassifier(n_estimators=trees,min_samples_leaf=min_leaf,max_features="sqrt",class_weight="balanced",random_state=seed,n_jobs=n_jobs)


def fit_forest(forest,x,y):
    """Fit with class_weight="balanced" passed as the equivalent explicit sample weights.

    Squeak Peek addition: identical weights and trees, but avoids scikit-learn
    1.9's failure on numeric-looking string labels ("5" -> "classes [5] are not
    in class_weight"). The fitted forest keeps class_weight="balanced".
    """
    from sklearn.utils.class_weight import compute_sample_weight
    if forest.class_weight!="balanced":return forest.fit(x,y)
    forest.set_params(class_weight=None)
    try:forest.fit(x,y,sample_weight=compute_sample_weight("balanced",y))
    finally:forest.set_params(class_weight="balanced")
    return forest


@dataclass
class DistanceModel:
    median: np.ndarray
    scale: np.ndarray
    classes: np.ndarray
    centroids: np.ndarray
    thresholds: np.ndarray
    quantile: float
    cutoff_source: str = "within_class_training_quantile_heuristic_not_validated_OOD_detector"

    @classmethod
    def fit(cls,x,y,classes,quantile=.99):
        med=np.median(x,axis=0);scale=1.4826*np.median(np.abs(x-med),axis=0);scale[scale<np.finfo(float).eps]=1
        z=(x-med)/scale;centers=[];thresholds=[]
        for c in classes:
            center=np.median(z[y==c],axis=0);centers.append(center)
            distance=np.linalg.norm(z[y==c]-center,axis=1)
            # A degenerate training class does not disable rejection with infinity.
            thresholds.append(max(float(np.quantile(distance,quantile)),1e-9))
        return cls(med,scale,np.asarray(classes),np.array(centers),np.array(thresholds),quantile)

    def predict_distance(self,x,prediction):
        mapping={c:i for i,c in enumerate(self.classes)}
        try:indices=np.array([mapping[c] for c in prediction],int)
        except KeyError as exc:raise ValueError("Predicted class absent from distance model") from exc
        z=(x-self.median)/self.scale
        return np.linalg.norm(z-self.centroids[indices],axis=1),self.thresholds[indices]


def score_summary(scores):
    ordered=np.sort(scores,axis=1)
    return ordered[:,-1],ordered[:,-1]-ordered[:,-2] if scores.shape[1]>1 else ordered[:,-1]


def selective_metrics(y,pred,accepted,classes):
    coverage=float(np.mean(accepted));per_class=[np.mean(accepted[y==c]) if (y==c).any() else 0. for c in classes]
    metrics=classification_metrics(y[accepted],pred[accepted],classes)
    return dict(Coverage=coverage,MinClassCoverage=float(min(per_class)),AcceptedAccuracy=metrics["Accuracy"] if accepted.any() else np.nan,
                MacroF1Accepted=metrics["MacroF1"],NAccepted=int(accepted.sum()))


def calibrate_rejection(y,pred,pmax,margin,dist,dist_threshold,classes,eligible=None,target_accuracy=.85,min_coverage=.5,min_class_coverage=.2,prob_grid=None,margin_grid=None):
    eligible=np.ones(len(y),bool) if eligible is None else eligible
    prob_grid=np.arange(.30,.901,.025) if prob_grid is None else prob_grid
    margin_grid=np.arange(0,.501,.025) if margin_grid is None else margin_grid
    rows=[]
    for p in prob_grid:
        for m in margin_grid:
            accepted=eligible&(pmax>=p)&(margin>=m)&(dist<=dist_threshold)
            row=selective_metrics(y,pred,accepted,classes)
            row.update(ProbabilityThreshold=float(p),MarginThreshold=float(m),Objective=row["MacroF1Accepted"]*np.sqrt(row["Coverage"]))
            row["TargetMet"]=bool(row["AcceptedAccuracy"]>=target_accuracy and row["Coverage"]>=min_coverage and row["MinClassCoverage"]>=min_class_coverage)
            rows.append(row)
    grid=pd.DataFrame(rows)
    if grid.empty:raise ValueError("Empty threshold grid")
    good=grid[grid.TargetMet]
    chosen=good.sort_values(["Coverage","Objective"],ascending=False,kind="stable").iloc[0] if len(good) else grid.sort_values("Objective",ascending=False,kind="stable").iloc[0]
    return chosen.to_dict(),grid


@dataclass
class RFModel:
    forest: object
    columns: list
    imputation: np.ndarray
    distance: DistanceModel
    calibration: dict
    feature_config_hash: str
    training_groups: list
    calibration_groups: list
    feature_version: str = FEATURE_VERSION

    def transform(self,table):
        validate_feature_contract(table,self.feature_config_hash)
        return (matrix(table,self.columns)-self.distance.median)/self.distance.scale

    def predict(self,table):
        validate_feature_contract(table,self.feature_config_hash)
        x=matrix(table,self.columns);bad=~np.isfinite(x).all(axis=1);filled=impute(x,self.imputation)
        scores=self.forest.predict_proba(filled);pred=self.forest.classes_[np.argmax(scores,axis=1)]
        p,margin=score_summary(scores);d,threshold=self.distance.predict_distance(filled,pred)
        reasons=[[] for _ in range(len(table))]
        masks={"MISSING_REQUIRED_FEATURE":bad,"LOW_RF_PROBABILITY":p<self.calibration["ProbabilityThreshold"],
               "LOW_RF_MARGIN":margin<self.calibration["MarginThreshold"],"OUTSIDE_TRAINING_DISTANCE":d>threshold}
        if "ProcessingStatus" in table:masks["FEATURE_PROCESSING_FAILED"]=~table.ProcessingStatus.eq("OK").to_numpy()
        if "BoundsClipped" in table:masks["CALL_BOUNDS_CLIPPED"]=bool_values(table.BoundsClipped)
        if requires_contour(self.columns):
            require_columns(table,["ContourValid"]);masks["LOW_CONTOUR_QUALITY"]=~bool_values(table.ContourValid)
        for reason,mask in masks.items():
            for i in np.flatnonzero(mask):reasons[i].append(reason)
        uncertain=np.array([bool(r) for r in reasons])
        return pd.DataFrame(dict(RawClusterPred=pred,ClusterPred=np.where(uncertain,"UNCERTAIN",pred),RFMaxProbability=p,RFMargin=margin,
                                 RFDistanceToCentroid=d,RFDistanceThreshold=threshold,IsUncertain=uncertain,UncertainReason=[";".join(r) for r in reasons]),index=table.index)


def validate_feature_contract(table,config_hash):
    require_columns(table,["FeatureVersion","FeatureConfigHash"])
    if not table.FeatureVersion.eq(FEATURE_VERSION).all() or not table.FeatureConfigHash.eq(config_hash).all():
        raise ValueError("Feature version/preprocessing incompatible with model; recompute from WAV")


def train_random_forest(table,config=None,columns=None,group_col="AnimalID",trees=200,seed=7,n_jobs=1,**threshold_options):
    config=config or FeatureConfig();columns=list(columns or feature_schema()["Compact"])
    assert_split_integrity(table,group_col);validate_feature_contract(table,digest(config))
    _,labels=supervised_labels(table)
    work=table.copy();work["_label"]=labels
    work=work[work.Split.isin(["train","calibration"]) & work._label.ne("")].copy()
    if "ProcessingStatus" in work:work=work[work.ProcessingStatus.eq("OK")]
    if "BoundsClipped" in work:work=work[~bool_values(work.BoundsClipped)]
    x=matrix(work,columns);y=work._label.to_numpy(str);train=work.Split.eq("train").to_numpy();cal=work.Split.eq("calibration").to_numpy()
    eligible=np.isfinite(x).all(axis=1)
    if requires_contour(columns):eligible&=bool_values(work.ContourValid)
    fit=train&eligible
    classes=np.unique(y)
    if len(classes)<2:raise ValueError("At least two reviewed USV subtypes needed")
    class_coverage(y[fit],y[cal],classes)
    median=learn_imputation(x[fit]);filled=impute(x,median)
    forest=make_forest(trees,seed,n_jobs);fit_forest(forest,filled[fit],y[fit])
    distance=DistanceModel.fit(filled[fit],y[fit],forest.classes_)
    scores=forest.predict_proba(filled[cal]);pred=forest.classes_[scores.argmax(axis=1)]
    p,margin=score_summary(scores);d,dt=distance.predict_distance(filled[cal],pred)
    calibration,grid=calibrate_rejection(y[cal],pred,p,margin,d,dt,classes,eligible[cal],**threshold_options)
    calibration.update(NCalibration=int(cal.sum()),NFit=int(fit.sum()),Seed=seed,DistanceCutoffSource=distance.cutoff_source)
    model=RFModel(forest,columns,median,distance,calibration,digest(config),sorted(work.loc[fit,group_col].astype(str).unique()),sorted(work.loc[cal,group_col].astype(str).unique()))
    # Deliberately save the exact fitted predictor whose thresholds were selected.
    predictions=model.predict(work.loc[cal]);predictions["TrueLabel"]=y[cal];predictions["CallID"]=work.loc[cal,"CallID"]
    return model,grid,predictions
