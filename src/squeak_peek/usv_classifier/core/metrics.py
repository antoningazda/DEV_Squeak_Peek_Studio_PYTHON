import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def classification_metrics(truth,prediction,classes):
    truth,prediction=np.asarray(truth,str),np.asarray(prediction,str)
    classes=list(map(str,classes))
    if len(classes)!=len(set(classes)) or not classes or len(truth)!=len(prediction):
        raise ValueError("Invalid class universe or label lengths")
    if not set(truth)<=set(classes):raise ValueError("Truth outside frozen class universe")
    # Rejections and processing errors count as false negatives, not dropped rows.
    support=np.array([(truth==c).sum() for c in classes])
    tp=np.array([((truth==c)&(prediction==c)).sum() for c in classes])
    predicted=np.array([(prediction==c).sum() for c in classes])
    recall=np.divide(tp,support,out=np.zeros(len(classes),float),where=support>0)
    precision=np.divide(tp,predicted,out=np.zeros(len(classes),float),where=predicted>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(len(classes),float),where=(precision+recall)>0)
    all_labels=classes+sorted(set(prediction)-set(classes))
    return dict(Accuracy=float(np.mean(truth==prediction)) if len(truth) else 0.,
                BalancedAccuracy=float(recall.mean()),MacroF1=float(f1.mean()),N=len(truth),
                ClassOrder=classes,Support=support.tolist(),Recall=recall.tolist(),Precision=precision.tolist(),
                F1=f1.tolist(),ConfusionLabels=all_labels,ConfusionMatrix=confusion_matrix(truth,prediction,labels=all_labels).tolist() if len(truth) else [],
                ZeroSupportClasses=[c for c,n in zip(classes,support) if not n])


def grouped_metrics(truth,prediction,groups,classes,bootstrap=1000,seed=7):
    truth,prediction,groups=np.asarray(truth,str),np.asarray(prediction,str),np.asarray(groups,str)
    if not len(truth) or len(groups)!=len(truth):raise ValueError("Empty or inconsistent evaluation data")
    unique=np.unique(groups)
    group_rows=[]
    for g in unique:
        m=classification_metrics(truth[groups==g],prediction[groups==g],classes)
        group_rows.append({"Group":g,**{k:m[k] for k in ("N","Accuracy","BalancedAccuracy","MacroF1")}})
    result=classification_metrics(truth,prediction,classes)
    result["GroupMacroF1"]=float(np.mean([r["MacroF1"] for r in group_rows]))
    rng=np.random.default_rng(seed);scores=[]
    if len(unique)>1:
        indices={g:np.flatnonzero(groups==g) for g in unique}
        for _ in range(bootstrap):
            idx=np.concatenate([indices[g] for g in rng.choice(unique,len(unique),replace=True)])
            scores.append(classification_metrics(truth[idx],prediction[idx],classes)["MacroF1"])
    result["MacroF1GroupBootstrap95CI"]=np.quantile(scores,[.025,.975]).tolist() if scores else None
    result["NGroups"]=len(unique)
    return result,pd.DataFrame(group_rows)


def calibrate_stage1_threshold(truth,scores,grid=None,minimum_precision=.5,criterion="F1"):
    truth,scores=np.asarray(truth,str),np.asarray(scores,float)
    if set(truth)!={"USV","NOISE"} or len(truth)!=len(scores) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError("Calibration needs both binary classes and finite [0,1] scores")
    grid=np.arange(.05,.951,.01) if grid is None else np.asarray(grid,float)
    if grid.size==0 or not np.isfinite(grid).all() or np.any((grid<0)|(grid>1)) or not 0<=minimum_precision<=1:
        raise ValueError("Invalid calibration grid/precision")
    rows=[];actual=truth=="USV"
    for threshold in grid:
        p=scores>=threshold;tp=(p&actual).sum();fp=(p&~actual).sum();fn=(~p&actual).sum();tn=(~p&~actual).sum()
        precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);specificity=tn/max(tn+fp,1)
        rows.append(dict(Threshold=threshold,Precision=precision,Recall=recall,F1=2*precision*recall/max(precision+recall,np.finfo(float).eps),
                         Specificity=specificity,BalancedAccuracy=(recall+specificity)/2,FP=fp,FN=fn))
    table=pd.DataFrame(rows);eligible=table.Precision>=minimum_precision
    if criterion not in ("F1","Recall","BalancedAccuracy"):raise ValueError("Unknown calibration criterion")
    feasible=bool(eligible.any());candidates=table.loc[eligible] if feasible else table
    best=candidates.loc[candidates[criterion].idxmax()].to_dict()
    return {**best,"TargetMet":feasible,"MinimumPrecision":minimum_precision,"NCalibration":len(truth),"Criterion":criterion},table
