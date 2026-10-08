"""Dominant spectral ridge tracking; derivatives never bridge long gaps."""
from dataclasses import replace
import warnings
import numpy as np
from scipy.signal import find_peaks
from .config import TrackerConfig

EPS = np.finfo(float).eps


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def moving_median_shrink(x, width, axis=0):
    x = np.asarray(x, float)
    width = max(1, int(width))
    before, after = width//2, (width-1)//2
    if np.isfinite(x).all():
        # Interior has no padding or missing data: avoid NumPy's masked-array
        # nanmedian sorting machinery for every frequency/frame/window entry.
        values=np.moveaxis(x,axis,0);n=len(values);result=np.empty_like(values)
        if width<=n:
            windows=np.lib.stride_tricks.sliding_window_view(values,width,axis=0)
            result[before:n-after]=np.median(windows,axis=-1)
            edges=list(range(before))+list(range(n-after,n))
        else:edges=range(n)
        for i in edges:result[i]=np.median(values[max(0,i-before):min(n,i+after+1)],axis=0)
        return np.moveaxis(result,0,axis)
    pads = [(0,0)] * x.ndim
    pads[axis] = (before, after)
    padded = np.pad(x, pads, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(windows, axis=-1)


def interpolate_short_gaps(track, max_gap):
    out = np.asarray(track, float).copy()
    count = 0
    for a, b in runs(~np.isfinite(out)):
        if b-a <= max_gap and a > 0 and b < len(out):
            out[a:b] = np.linspace(out[a-1], out[b], b-a+2)[1:-1]
            count += b-a
    return out, count


def empty_quality():
    return dict(ContourValid=False, ContourStatus="EMPTY_OR_INVALID_SPECTRUM", TrackCoverage=0.0,
                MedianPeakProminence_dB=np.nan, TrackJumpP95_Hz=np.nan, FractionInterpolated=0.0,
                NumTrackGaps=0, LongestTrackGap_ms=np.nan)


def track_dominant_frequency(power, frequencies, times, config=None, return_debug=False):
    o = config or TrackerConfig()
    p, f, t = np.asarray(power, float), np.asarray(frequencies, float), np.asarray(times, float)
    if p.ndim != 2 or p.shape != (len(f), len(t)):
        raise ValueError("Power must have shape [frequency, time]")
    if not np.isfinite(f).all() or not np.isfinite(t).all() or np.any(np.diff(f) <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("Frequency/time axes must be finite and strictly increasing")
    n_f, n_t = p.shape
    p = np.where(np.isfinite(p) & (p >= 0), p, 0)
    if n_t == 0 or n_f < 2 or p.max() <= 0:
        answer = (np.full(n_t, np.nan), empty_quality())
        return (*answer, {}) if return_debug else answer
    db = 10*np.log10(p+EPS)
    rel = db-db.max()
    width = max(3, int(o.baseline_bins)) | 1
    width = min(width, n_f - ((n_f+1) % 2))
    width = min(3, n_f) if width < 3 else width
    baseline = moving_median_shrink(db, width, axis=0)
    enhanced = db-baseline
    m = o.max_candidates
    cf, cp, ce = np.full((m,n_t), np.nan), np.full((m,n_t), np.nan), np.full((m,n_t), -np.inf)
    for k in range(n_t):
        if rel[:,k].max() < o.min_frame_level_db:
            continue
        peaks, props = find_peaks(enhanced[:,k], prominence=o.min_prominence_db)
        proms = props["prominences"]
        if not len(peaks):
            j = int(enhanced[:,k].argmax())
            if enhanced[j,k] >= 0.5*o.min_prominence_db:
                peaks, proms = np.array([j]), np.array([max(enhanced[j,k], 0)])
        valid = rel[peaks,k] >= o.min_frame_level_db
        peaks, proms = peaks[valid], proms[valid]
        emit = proms + .08*rel[peaks,k]
        order = np.argsort(-emit, kind="stable")[:m]
        cf[:len(order),k], cp[:len(order),k], ce[:len(order),k] = f[peaks[order]], proms[order], emit[order]
    dp, back = np.full((m+1,n_t), -np.inf), np.zeros((m+1,n_t), int)
    dp[:m,0], dp[m,0] = ce[:,0], -o.gap_emission_penalty
    for k in range(1,n_t):
        # Vectorized candidate-to-candidate transitions, plus the explicit gap.
        jump = np.abs(cf[:,k,None]-cf[None,:,k-1])
        penalty = o.jump_penalty_per_khz*jump/1000 + o.large_jump_penalty_per_khz*np.maximum(jump-o.preferred_max_jump_hz,0)/1000
        tr = np.full((m+1,m+1), -np.inf)
        tr[:m,:m] = np.where(np.isfinite(penalty), dp[None,:m,k-1]-penalty, -np.inf)
        tr[:m,m] = dp[m,k-1]-o.gap_transition_penalty
        tr[m,:] = dp[:,k-1]-o.gap_transition_penalty
        tr[m,m] = dp[m,k-1]-o.gap_continue_penalty
        back[:,k] = np.argmax(tr, axis=1)
        dp[:,k] = tr[np.arange(m+1), back[:,k]] + np.r_[ce[:,k], -o.gap_emission_penalty]
    states = np.zeros(n_t, int)
    states[-1] = dp[:,-1].argmax()
    for k in range(n_t-1,0,-1):
        states[k-1] = back[states[k], k]
    raw, prominence = np.full(n_t,np.nan), np.full(n_t,np.nan)
    valid = states < m
    raw[valid], prominence[valid] = cf[states[valid],np.flatnonzero(valid)], cp[states[valid],np.flatnonzero(valid)]
    coverage = float(np.isfinite(raw).mean())
    filled, count = interpolate_short_gaps(raw, o.max_interp_gap_frames)
    smooth = filled.copy()
    # Smooth each remaining valid run independently, preserving all gap masks.
    for a,b in runs(np.isfinite(filled)):
        smooth[a:b] = moving_median_shrink(filled[a:b], o.smooth_frames | 1)
    gaps = runs(~np.isfinite(raw))
    dt = np.median(np.diff(t)) if len(t)>1 else np.nan
    med_prom = float(np.nanmedian(prominence)) if np.isfinite(prominence).any() else np.nan
    jumps = np.abs(np.diff(smooth))
    jumps = jumps[np.isfinite(jumps)]
    enough = np.isfinite(smooth).sum() >= o.min_frames
    is_valid = coverage >= o.min_coverage and enough and np.isfinite(med_prom) and med_prom >= o.min_median_prominence_db
    status = "VALID" if is_valid else "LOW_COVERAGE" if coverage < o.min_coverage else "TOO_FEW_TRACK_POINTS" if not enough else "LOW_PEAK_PROMINENCE"
    q = dict(ContourValid=bool(is_valid), ContourStatus=status, TrackCoverage=coverage,
             MedianPeakProminence_dB=med_prom, TrackJumpP95_Hz=float(np.quantile(jumps,.95)) if len(jumps) else np.nan,
             FractionInterpolated=count/n_t, NumTrackGaps=len(gaps),
             LongestTrackGap_ms=max((b-a for a,b in gaps),default=0)*dt*1000)
    answer = (smooth if is_valid else np.full(n_t,np.nan), q)
    debug = dict(PdbRel=rel, LocalBaseline=baseline, Enhanced=enhanced, CandidateFrequency_Hz=cf,
                 CandidateProminence_dB=cp, RawTrack_Hz=raw, InterpolatedTrack_Hz=filled, StatePath=states)
    return (*answer,debug) if return_debug else answer


def contour_descriptors(track, times, duration, min_step=500, is_valid=True):
    names = ("Mean", "Median", "Std", "Range", "RobustRange", "LinearSlope", "TotalVariationRate", "DirectionChanges", "DirectionChangeRate", "LinearityResidual", "MaxJump", "MeanAbsDiff", "MeanSlope", "SlopeStd")
    d = dict.fromkeys(names, np.nan)
    f,t = np.asarray(track,float), np.asarray(times,float)
    valid = np.isfinite(f) & np.isfinite(t)
    if not is_valid or valid.sum()<2:
        return d
    fv,tv = f[valid],t[valid]
    fit = np.polyfit(tv,fv,1)
    d.update(Mean=fv.mean(), Median=np.median(fv), Std=fv.std(ddof=1), Range=np.ptp(fv),
             RobustRange=np.quantile(fv,.95)-np.quantile(fv,.05), LinearSlope=fit[0],
             LinearityResidual=np.abs(fv-np.polyval(fit,tv)).mean())
    differences, deltas, reversals = [], [], 0
    for a,b in runs(valid):
        df,dt = np.diff(f[a:b]),np.diff(t[a:b])
        ok = dt>0
        df,dt = df[ok],dt[ok]
        differences.extend(df); deltas.extend(dt)
        signs = np.sign(df[np.abs(df)>min_step])
        reversals += int(np.count_nonzero(np.diff(signs)))
    if differences:
        df,dt = np.array(differences),np.array(deltas)
        slopes = df/dt
        d.update(TotalVariationRate=np.abs(slopes).mean(), MeanAbsDiff=np.abs(df).mean(),
                 MeanSlope=slopes.mean(), SlopeStd=slopes.std(ddof=1) if len(slopes)>1 else 0.0,
                 MaxJump=np.abs(df).max(), DirectionChanges=reversals, DirectionChangeRate=reversals/max(duration,EPS)*.1)
    return d
