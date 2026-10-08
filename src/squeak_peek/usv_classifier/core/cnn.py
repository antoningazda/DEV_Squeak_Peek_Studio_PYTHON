"""Compact native-resolution CNN; calibration never controls weight fitting."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .common import new_directory, read_table, write_csv, write_json
from .config import ImageConfig
from .metrics import calibrate_stage1_threshold, classification_metrics
from .splits import class_coverage


class CompactCNN(nn.Module):
    def __init__(self,channels=1):
        super().__init__()
        layers=[];previous=channels
        for i,width in enumerate((16,32,64,128)):
            layers.extend([nn.Conv2d(previous,width,3,padding=1),nn.BatchNorm2d(width,eps=1e-5,momentum=.1),nn.ReLU()])
            if i<3:layers.append(nn.MaxPool2d(2))
            previous=width
        layers.extend([nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.25),nn.Linear(128,2)])
        self.layers=nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)


class ImageDataset(Dataset):
    def __init__(self,table,root,config):
        self.table=table.reset_index(drop=True);self.root=Path(root);self.config=config

    def __len__(self):return len(self.table)

    def __getitem__(self,index):
        row=self.table.iloc[index]
        with Image.open(self.root/row.ImagePath) as image:
            a=np.asarray(image.convert("L" if self.config.channels==1 else "RGB"),dtype=np.float32).copy()
        if a.shape[:2]!=tuple(self.config.size):raise ValueError("Dataset/model image sizes differ")
        if a.ndim==2:a=a[:,:,None]
        return torch.from_numpy(a.transpose(2,0,1)),int(row.Label=="USV")


def resolve_device(device):
    if device=="auto":return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu","cuda","mps"):raise ValueError("Unknown device")
    return device


def predict_images(net,images,batch_size=64,device="cpu"):
    if not images:return np.empty(0,float)
    net.eval();scores=[]
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            a=np.stack(images[start:start+batch_size]).astype(np.float32)
            if a.ndim==3:a=a[:,:,:,None]
            x=torch.from_numpy(a.transpose(0,3,1,2)).to(device)
            scores.extend(torch.softmax(net(x),dim=1)[:,1].cpu().numpy())
    return np.array(scores)


def score_dataset(net,loader,device):
    net.eval();scores=[];truth=[]
    with torch.inference_mode():
        for x,y in loader:
            scores.extend(torch.softmax(net(x.to(device)),dim=1)[:,1].cpu().numpy())
            truth.extend(y.numpy())
    return np.where(np.array(truth)==1,"USV","NOISE"),np.array(scores)


def train_compact_stage1_cnn(dataset_dir,out_dir,epochs=12,batch_size=64,learning_rate=1e-3,seed=7,device="cpu",threads=1):
    if epochs<1 or batch_size<1 or learning_rate<=0:raise ValueError("Invalid CNN training parameters")
    root=Path(dataset_dir);manifest=read_table(root/"manifest.csv");config=ImageConfig(**json.loads((root/"settings.json").read_text()))
    train=manifest[manifest.Split.eq("train")];cal=manifest[manifest.Split.eq("calibration")]
    if set(train.Group)&set(cal.Group):raise ValueError("CNN group leakage")
    if manifest.CallID.duplicated().any():raise ValueError("Duplicate CNN dataset calls")
    class_coverage(train.Label,cal.Label,["NOISE","USV"])
    out=new_directory(out_dir)
    device=resolve_device(device);torch.set_num_threads(threads)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(ImageDataset(train,root,config),batch_size=batch_size,shuffle=True,num_workers=0,generator=generator)
    cal_loader=DataLoader(ImageDataset(cal,root,config),batch_size=batch_size,shuffle=False,num_workers=0)
    net=CompactCNN(config.channels).to(device)
    counts=np.array([(train.Label==c).sum() for c in ("NOISE","USV")])
    weight=torch.tensor(counts.sum()/(2*counts),dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weight);optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
    history=[];started=time.perf_counter()
    # Epoch count is fixed in advance. No calibration-based early stopping.
    for epoch in range(epochs):
        net.train();loss_sum=0.;n=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
            loss=criterion(net(x),y)
            if not torch.isfinite(loss):raise ValueError("Nonfinite CNN training loss")
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(y);n+=len(y)
        history.append(dict(Epoch=epoch+1,TrainingLoss=loss_sum/n))
    truth,scores=score_dataset(net,cal_loader,device)
    calibration,sweep=calibrate_stage1_threshold(truth,scores)
    checkpoint=dict(format="USV_PY_CNN_1",state_dict={k:v.cpu() for k,v in net.state_dict().items()},image_config=asdict(config),
                    class_order=["NOISE","USV"],input_scale="uint8_0_255_as_float32",threshold=float(calibration["Threshold"]),
                    calibration={k:(v.item() if isinstance(v,np.generic) else v) for k,v in calibration.items()},
                    training_groups=sorted(train.Group.astype(str).unique()),calibration_groups=sorted(cal.Group.astype(str).unique()),
                    training=dict(epochs=epochs,batch_size=batch_size,learning_rate=learning_rate,seed=seed,n_train=len(train),n_calibration=len(cal),seconds=time.perf_counter()-started))
    torch.save(checkpoint,out/"cnn.pt")
    write_csv(pd.DataFrame(history),out/"training_history.csv");write_csv(sweep,out/"threshold_sweep.csv")
    predictions=cal.copy();predictions["pUSV"]=scores;predictions["Prediction"]=np.where(scores>=checkpoint["threshold"],"USV","NOISE")
    write_csv(predictions,out/"calibration_predictions.csv")
    write_json(out/"calibration.json",calibration)
    return out/"cnn.pt"


def load_cnn(path,device="cpu"):
    device=resolve_device(device)
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint.get("format")!="USV_PY_CNN_1" or checkpoint.get("class_order")!=["NOISE","USV"]:
        raise ValueError("Unsupported CNN format/class order")
    if not np.isfinite(checkpoint["threshold"]) or not 0<=checkpoint["threshold"]<=1:
        raise ValueError("Invalid frozen CNN threshold")
    if set(checkpoint["training_groups"])&set(checkpoint["calibration_groups"]):raise ValueError("CNN provenance contains group leakage")
    config=ImageConfig(**checkpoint["image_config"]);net=CompactCNN(config.channels)
    net.load_state_dict(checkpoint["state_dict"],strict=True);net.to(device);net.eval()
    return net,config,checkpoint
