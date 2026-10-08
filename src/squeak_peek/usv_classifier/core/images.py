from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .common import digest, file_hash, new_directory, write_csv, write_json, bool_values
from .config import ImageConfig
from .features import raw_stft, round_positive
from .io import read_audio, sample_bounds, validate_identity
from .labels import supervised_labels
from .splits import assert_split_integrity, class_coverage


def cnn_base_spectrogram(segment,fs,config=None):
    o=config or ImageConfig()
    x=np.asarray(segment,float)
    if x.ndim==2:x=x.mean(axis=1)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError("Invalid CNN waveform")
    x=x-x.mean()
    win=max(16,round_positive(o.win_ms/1000*fs));hop=max(1,round_positive(o.hop_ms/1000*fs))
    # MATLAB noverlap=max(0,win-hop) implies an effective hop <= window.
    hop=min(hop,win)
    nfft=max(o.nfft,1<<(win-1).bit_length())
    s,f,_=raw_stft(x,fs,win,hop,nfft,window="hamming")
    band=(f>=o.freq_range_hz[0])&(f<=min(o.freq_range_hz[1],.98*fs/2))
    if not band.any():raise ValueError("No CNN frequency bins; sampling rate too low")
    p=10*np.log10(np.abs(s[band])**2+1e-12)
    lo,hi=o.db_range;p=(np.clip(p,lo,hi)-lo)/(hi-lo)
    p=p**(1/o.gamma)
    return 1-p if o.invert else p


def resize_spectrogram(base,config=None):
    o=config or ImageConfig()
    # Explicit Pillow bilinear antialiasing; versioned, not claimed bitwise MATLAB parity.
    resized=np.asarray(Image.fromarray(np.asarray(base,np.float32)).resize((o.size[1],o.size[0]),Image.Resampling.BILINEAR))
    image=np.clip(np.floor(255*resized+.5),0,255).astype(np.uint8)
    return np.repeat(image[:,:,None],3,axis=2) if o.channels==3 else image


def make_cnn_spectrogram(segment,fs,config=None):
    return resize_spectrogram(cnn_base_spectrogram(segment,fs,config),config)


def image_for_call(x,fs,row,config):
    duration=float(row["End_s"])-float(row["Start_s"])
    if not config.min_call_s<=duration<=config.max_call_s:
        raise ValueError("DURATION_OUTSIDE_CNN_SUPPORT")
    if float(row["Start_s"])<0 or float(row["End_s"])>len(x)/fs:
        raise ValueError("CALL_BOUNDS_CLIPPED")
    a,b=sample_bounds(row["Start_s"]-config.pad_s,row["End_s"]+config.pad_s,fs,len(x))
    return make_cnn_spectrogram(x[a:b],fs,config)


def prepare_stage1_dataset(table,out_dir,config=None,group_col="AnimalID",base_cache=None):
    o=config or ImageConfig();validate_identity(table);assert_split_integrity(table,group_col)
    binary,_=supervised_labels(table)
    work=table.copy();work["Label"]=binary
    # Test cohort is never rasterized for model fitting or threshold selection.
    work=work[work.Split.isin(["train","calibration"]) & work.Label.ne("")]
    out=new_directory(out_dir)
    cache=Path(base_cache) if base_cache else None
    if cache:cache.mkdir(parents=True,exist_ok=True)
    rows=[];excluded=[]
    for wav,sub in work.groupby("WavFile",sort=False):
        try:
            x,fs=read_audio(wav);wave_hash=file_hash(wav)
            if "WaveformSHA256" in sub:
                recorded=sub.WaveformSHA256.dropna().astype(str);recorded=recorded[recorded.ne("")]
                if len(recorded) and not recorded.eq(wave_hash).all():
                    raise ValueError("WAVEFORM_CHANGED_RECOMPUTE_FEATURES")
        except (OSError,ValueError,RuntimeError) as exc:
            excluded.extend({"CallID":r.CallID,"Reason":str(exc)} for r in sub.itertuples());continue
        for row in sub.to_dict("records"):
            try:
                duration=row["End_s"]-row["Start_s"]
                if not o.min_call_s<=duration<=o.max_call_s:raise ValueError("DURATION_OUTSIDE_CNN_SUPPORT")
                if row["Start_s"]<0 or row["End_s"]>len(x)/fs:raise ValueError("CALL_BOUNDS_CLIPPED")
                from dataclasses import asdict
                settings=asdict(o);settings.pop("size");settings.pop("channels")
                key=digest({"image_version":"PY_IMAGE_1","waveform":wave_hash,"start":row["Start_s"],"end":row["End_s"],"settings":settings})
                cached=cache/(key+".npy") if cache else None
                if cached and cached.exists():base=np.load(cached,allow_pickle=False)
                else:
                    a,b=sample_bounds(row["Start_s"]-o.pad_s,row["End_s"]+o.pad_s,fs,len(x))
                    base=cnn_base_spectrogram(x[a:b],fs,o)
                    if cached:np.save(cached,base,allow_pickle=False)
                image=resize_spectrogram(base,o)
                relative=Path(row["Split"])/row["Label"]/(digest(str(row["CallID"]))+".png")
                target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(image).save(target)
                rows.append({"CallID":row["CallID"],"RecordingID":row["RecordingID"],"Group":row[group_col],"Split":row["Split"],
                             "Label":row["Label"],"WavFile":wav,"Start_s":row["Start_s"],"End_s":row["End_s"],"ImagePath":str(relative)})
            except (ValueError,OSError,RuntimeError) as exc:
                excluded.append({"CallID":row["CallID"],"Reason":str(exc)})
    manifest=pd.DataFrame(rows,columns=["CallID","RecordingID","Group","Split","Label","WavFile","Start_s","End_s","ImagePath"])
    write_csv(manifest,out/"manifest.csv");write_csv(pd.DataFrame(excluded,columns=["CallID","Reason"]),out/"excluded.csv")
    write_json(out/"settings.json",o)
    class_coverage(manifest.loc[manifest.Split.eq("train"),"Label"],manifest.loc[manifest.Split.eq("calibration"),"Label"],["NOISE","USV"])
    return manifest
