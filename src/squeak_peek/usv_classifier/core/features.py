from dataclasses import replace
import json
import numpy as np
import pandas as pd
from scipy import signal
from . import FEATURE_VERSION
from .common import digest
from .config import FeatureConfig
from .io import sample_bounds
from .schema import feature_schema
from .tracker import contour_descriptors, empty_quality, track_dominant_frequency

EPS = np.finfo(float).eps


def round_positive(x):
    return int(np.floor(x+.5))


def raw_stft(x, fs, win, hop, nfft, window="hann"):
    """Unnormalized one-sided FFT, full frames only, no padding/detrending."""
    if len(x)<win or nfft<win or min(win,hop)<1:
        raise ValueError("Invalid STFT dimensions or segment shorter than window")
    frames = np.lib.stride_tricks.sliding_window_view(x,win)[::hop]
    w = signal.windows.hann(win,sym=False) if window=="hann" else signal.windows.hamming(win,sym=True)
    s = np.fft.rfft(frames*w, n=nfft, axis=1).T
    return s,np.fft.rfftfreq(nfft,1/fs),(np.arange(len(frames))*hop+win/2)/fs


def preprocess_waveform(x, fs, config):
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all() or not np.isfinite(fs) or fs<=0:
        raise ValueError("Invalid waveform or sample rate")
    lo,hi = config.freq_range_hz[0], min(config.freq_range_hz[1],.98*fs/2)
    if hi<=lo:
        raise ValueError("Sampling rate cannot support requested ultrasonic band")
    sos = signal.butter(config.filter_order, [lo,hi], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos,x-x.mean(),padtype="odd")
    filtered = signal.filtfilt([1,-config.preemph],[1],filtered,padtype="odd")
    provenance = dict(method="butterworth_sosfiltfilt_then_filtfilt_preemphasis", sos=sos.tolist(),
                      preemphasis_b=[1,-config.preemph], fs=float(fs), band_hz=[lo,hi])
    return filtered,provenance


def template(row, wav, config):
    r = dict.fromkeys(feature_schema()["OutputNumeric"],np.nan)
    r.update(empty_quality())
    r.update({k:row.get(k,"") for k in ("RecordingID", "CallID", "SourceRow", "Start_s", "End_s")})
    r.update(WavFile=str(wav), Duration_s=np.nan, FeatureVersion=FEATURE_VERSION,
             FeatureConfigHash=digest(config), ProcessingStatus="OK", ProcessingReason="",
             OriginalStart_s=row["Start_s"],OriginalEnd_s=row["End_s"],UsedStart_s=np.nan,UsedEnd_s=np.nan,
             BoundsClipped=False, SampleRate_Hz=np.nan, EffectiveWinLen=np.nan, EffectiveHopLen=np.nan,
             EffectiveNFFT=np.nan, PreprocessingJSON="")
    return r


def error_rows(segments,wav,reason,config):
    rows=[]
    for source in segments.to_dict("records"):
        r=template(source,wav,config)
        r.update(ProcessingStatus="PROCESSING_ERROR",ProcessingReason=reason)
        rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=template({"Start_s":0,"End_s":0},wav,config))


def extract_features(x,fs,segments,wav_path="",config=None):
    o=config or FeatureConfig()
    xp,provenance=preprocess_waveform(x,fs,o)
    provenance_json=json.dumps(provenance,separators=(",",":"))
    global_noise=np.median(np.abs(xp))/.67448975
    if global_noise<=0:
        global_noise=np.sqrt(np.mean(xp*xp))
    rows=[]
    for source in segments.to_dict("records"):
        r=template(source,wav_path,o)
        r.update(SampleRate_Hz=fs,PreprocessingJSON=provenance_json)
        try:
            start,end=float(source["Start_s"]),float(source["End_s"])
            a,b=sample_bounds(start,end,fs,len(xp))
            r.update(UsedStart_s=a/fs,UsedEnd_s=b/fs,BoundsClipped=start<0 or end>len(xp)/fs)
            duration=(b-a)/fs
            r.update(Duration_s=duration,Duration_ms=duration*1000)
            if not o.min_call_s<=duration<=o.max_call_s:
                r.update(ProcessingStatus="OUTSIDE_SUPPORT",ProcessingReason="DURATION_OUTSIDE_FEATURE_SUPPORT")
                rows.append(r);continue
            seg=xp[a:b];length=len(seg)
            minimum=max(32,round_positive(o.min_window_ms/1000*fs))
            win=max(2,min(o.win_len,length,max(minimum,int(o.short_call_window_fraction*length))))
            hop=max(1,min(o.hop_len,win//4))
            target=max(256,min(o.nfft,max(win,256))) if o.nfft_policy=="matlab_v03" else max(win,o.nfft)
            nfft=1 << (int(target)-1).bit_length()
            r.update(EffectiveWinLen=win,EffectiveHopLen=hop,EffectiveNFFT=nfft)
            s,f,t=raw_stft(seg,fs,win,hop,nfft)
            band=(f>=o.freq_range_hz[0])&(f<=min(o.freq_range_hz[1],.98*fs/2))
            f=f[band]; p=np.abs(s[band])**2+EPS
            if not len(f):raise ValueError("No frequency bins in band")
            pf=p/p.sum(axis=0)
            cent=(f[:,None]*pf).sum(axis=0)
            spread=np.sqrt((((f[:,None]-cent)**2)*pf).sum(axis=0))
            flat=np.exp(np.log(p).mean(axis=0))/p.mean(axis=0)
            ent=-(pf*np.log(pf+EPS)).sum(axis=0)/np.log(len(f)) if len(f)>1 else np.zeros(p.shape[1])
            raw_peak=f[p.argmax(axis=0)]
            dt=np.median(np.diff(t)) if len(t)>1 else hop/fs
            tc=replace(o.tracker,smooth_frames=max(1,round_positive(o.smooth_ms/1000/dt)))
            track,q=track_dominant_frequency(p,f,t,tc)
            d=contour_descriptors(track,t,duration,o.min_direction_step_hz,q["ContourValid"])
            nb=round_positive(o.background_window_s*fs)
            neighbors=[xp[max(0,a-nb):a],xp[b:min(len(xp),b+nb)]]
            bg=[np.sqrt(np.mean(z*z)) for z in neighbors if len(z)]
            bg=[z for z in bg if np.isfinite(z) and z>0]
            noise=np.median(bg) if bg else global_noise
            rms=np.sqrt(np.mean(seg*seg)); signs=np.where(seg<0,-1,1)
            r.update(RMS=rms,ZCR=np.count_nonzero(np.diff(signs))/length,SpecCentroid_Hz=cent.mean(),
                     SpecSpread_Hz=spread.mean(),SpecFlatness=flat.mean(),SpecEntropy=ent.mean(),SNR_dB=20*np.log10((rms+EPS)/(max(noise,EPS)+EPS)))
            mapping={"DomFreqMean_Hz":"Mean","DomFreqMedian_Hz":"Median","DomFreqStd_Hz":"Std","DomFreqRobustRange_Hz":"RobustRange",
                     "DomFreqLinearSlope_HzPerS":"LinearSlope","DomFreqTotalVariation_HzPerS":"TotalVariationRate",
                     "DirectionChanges":"DirectionChanges","DirectionChangeRate_per100ms":"DirectionChangeRate",
                     "LinearityResidual_Hz":"LinearityResidual","MaxJump_Hz":"MaxJump","F0Mean_Hz":"Mean","F0Std_Hz":"Std",
                     "F0Range_Hz":"Range","F0MeanAbsDiff_Hz":"MeanAbsDiff","F0DirectionChanges":"DirectionChanges",
                     "F0LineFitResidual_Hz":"LinearityResidual","FMRate_HzPerS":"TotalVariationRate","FMSlopeMean":"MeanSlope",
                     "FMSlopeStd":"SlopeStd","FMInflections":"DirectionChanges"}
            r.update({k:d[v] for k,v in mapping.items()});r.update(q)
            r.update(PeakFreq_Hz=raw_peak.mean(),PeakFreqStd_Hz=raw_peak.std(ddof=1) if len(raw_peak)>1 else 0.,Bandwidth_Hz=2*spread.mean())
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            r.update(ProcessingStatus="PROCESSING_ERROR",ProcessingReason=str(exc))
        rows.append(r)
    if not rows:return error_rows(segments,wav_path,"",o)
    result=pd.DataFrame(rows)
    # ICI is undefined for first call, retained for reporting only.
    order=np.argsort(result.UsedStart_s.to_numpy(),kind="stable")
    previous=None
    for i in order:
        if not np.isfinite(result.loc[i,"UsedStart_s"]):continue
        if previous is not None:result.loc[i,"ICI_ms"]=1000*(result.loc[i,"UsedStart_s"]-previous)
        previous=result.loc[i,"UsedEnd_s"]
    return result
