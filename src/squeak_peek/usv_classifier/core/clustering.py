from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .common import bool_values
from .schema import feature_schema, feature_group_weights, matrix, requires_contour


@dataclass
class ClusterModel:
    columns: list
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    centroids: np.ndarray
    feature_config_hash: str
    feature_version: str

    def transform(self,table):
        if not table.FeatureConfigHash.eq(self.feature_config_hash).all() or not table.FeatureVersion.eq(self.feature_version).all():
            raise ValueError("Clustering feature configuration differs")
        return (matrix(table,self.columns)-self.mean)/self.std*self.weights

    def predict(self,table):
        z=self.transform(table)
        if not np.isfinite(z).all():raise ValueError("Nonfinite clustering features")
        return np.argmin(((z[:,None,:]-self.centroids)**2).sum(axis=2),axis=1)+1


def cluster_calls(table,columns=None,k=None,k_range=range(2,21),seed=7,balance_groups=True,silhouette_max=5000):
    columns=list(columns or feature_schema()["Compact"])
    if "Split" in table and not table.Split.eq("train").all():raise ValueError("Taxonomy construction accepts training partition only")
    x=matrix(table,columns);valid=np.isfinite(x).all(axis=1)
    if requires_contour(columns):valid&=bool_values(table.ContourValid)
    if "ProcessingStatus" in table:valid&=table.ProcessingStatus.eq("OK").to_numpy()
    if "BoundsClipped" in table:valid&=~bool_values(table.BoundsClipped)
    if valid.sum()<3:raise ValueError("Fewer than three eligible clustering calls")
    x=x[valid];mean=x.mean(axis=0);std=x.std(axis=0,ddof=1);std[std==0]=1
    weights=feature_group_weights(columns)[0] if balance_groups else np.ones(len(columns))
    z=(x-mean)/std*weights
    candidates=[k] if k is not None else list(k_range)
    if any(int(v)!=v or v<2 for v in candidates):raise ValueError("K must be integer >=2")
    diagnostics=[];models={}
    for value in candidates:
        if value>=len(z):
            if k is not None:raise ValueError("Fixed K must be smaller than eligible call count")
            continue
        model=KMeans(n_clusters=value,n_init=10 if k is not None else 5,max_iter=500,random_state=seed).fit(z)
        if len(np.unique(model.labels_))<value:raise ValueError("Too few distinct feature vectors for K")
        sil=np.nan
        if len(np.unique(model.labels_))>1:
            try:sil=silhouette_score(z,model.labels_,sample_size=min(len(z),silhouette_max),random_state=seed)
            except ValueError:pass
        diagnostics.append(dict(K=value,WSS=model.inertia_,SilhouetteMean=sil));models[value]=model
    if not diagnostics:raise ValueError("No valid K fits available")
    summary=pd.DataFrame(diagnostics)
    if k is None:
        if len(summary)<3:k=int(summary.K.iloc[0])
        else:
            xx=(summary.K-summary.K.iloc[0])/(summary.K.iloc[-1]-summary.K.iloc[0])
            yy=(summary.WSS-summary.WSS.iloc[-1])/(summary.WSS.iloc[0]-summary.WSS.iloc[-1]+np.finfo(float).eps)
            k=int(summary.loc[np.abs(1-xx-yy).idxmax(),"K"])
        final=KMeans(n_clusters=k,n_init=10,max_iter=500,random_state=seed).fit(z)
    else:final=models[k]
    hashes=table.FeatureConfigHash.unique();versions=table.FeatureVersion.unique()
    if len(hashes)!=1 or len(versions)!=1:raise ValueError("Mixed feature configurations/versions")
    model=ClusterModel(columns,mean,std,weights,final.cluster_centers_,str(hashes[0]),str(versions[0]))
    result=table.copy();result["Cluster"]="";result["ClusterStatus"]="UNCERTAIN"
    result.loc[valid,"Cluster"]=(final.labels_+1).astype(str);result.loc[valid,"ClusterStatus"]="CLUSTERED"
    result["Cluster_Clean"]="";result["ExpertReviewed"]=False;result["ExpertBinaryLabel"]=""
    return result,model,summary
