"""Group recordings sharing animal participants."""
import pandas as pd
from .common import require_columns


def connected_pair_groups(recordings,members_col="AnimalMembers"):
    """Semicolon-separated known participants; shared animals bind whole recordings."""
    require_columns(recordings,["RecordingID",members_col]);parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    members=[]
    for value in recordings[members_col]:
        if pd.isna(value):raise ValueError("Missing animal participants")
        animals=[a.strip() for a in str(value).split(";") if a.strip()]
        if not animals:raise ValueError("Empty animal participants")
        members.append(animals)
        for a in animals:parent[find(a)]=find(animals[0])
    components={}
    for a in parent:components.setdefault(find(a),[]).append(a)
    names={root:"pairgroup_"+"_".join(sorted(animals)) for root,animals in components.items()}
    result=recordings.copy();result["GroupID"]=[names[find(a[0])] for a in members]
    return result
