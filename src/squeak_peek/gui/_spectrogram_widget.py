"""Waveform + spectrogram display with draggable label boundaries and right-click label creation.

Three public additions to the SpectrogramWidget API:

1. COLORMAPS: 15 selectable colormaps (parula, turbo, hsv, hot, cool, spring, summer,
   autumn, winter, gray, bone, copper, pink, jet, invgray). Call set_colormap(name) to
   change, or pass colormap_name kwarg to display(). Persists across display() calls.

2. LABEL COLORS: 7 selectable colors (red, green, blue, cyan, magenta, yellow, white)
   for both detected and reference label overlays. Pass detected_color_name and/or
   reference_color_name kwargs to display() to override theme defaults. None/omitted
   falls back to current theme colors.

3. BOUNDARY DRAG & RIGHT-CLICK: Two new signals—
   - boundary_dragged(str, float): emits ("start" or "end", new_time) when dragging
     movable start/end lines (enable_boundary_drag() draws them, disable_boundary_drag()
     removes them).
   - spectrogram_right_clicked(float): emits time value on right-click in either plot.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from scipy.ndimage import median_filter

from squeak_peek.audio.filters import band_restrict, compute_stft
from squeak_peek.labels.model import Label

from . import _theme as theme

# Reversed grayscale: low power → light gray, high power → dark  (matches MATLAB)
_GRAY_CM = pg.ColorMap(
    pos=np.array([0.0, 1.0]),
    color=np.array([[210, 210, 210, 255], [15, 15, 15, 255]], dtype=np.uint8),
)

# Pitch trace: fixed bright orange, distinct from the 7 selectable label
# colors and both light/dark theme palettes, drawn on top of the spectrogram.
_PEN_PITCH = pg.mkPen("#FFA500", width=1.5)

def _resample_lut_to_256(raw: np.ndarray) -> np.ndarray:
    """Resample an (N, 3) RGB lookup table to exactly (256, 3) via per-channel
    linear interpolation over its normalized position axis. N < 256 upsamples
    (the standard, lossless way to get a smooth 256-entry LUT from a coarser
    sampling of the same underlying curve); N == 256 returns it unchanged."""
    n = raw.shape[0]
    if n == 256:
        return raw.astype(np.float32)
    src_pos = np.linspace(0.0, 1.0, n)
    dst_pos = np.linspace(0.0, 1.0, 256)
    return np.stack(
        [np.interp(dst_pos, src_pos, raw[:, c]) for c in range(3)], axis=1
    ).astype(np.float32)


# ── PARULA COLORMAP (raw samples, normalized 0–1, MATLAB's standard curve) ──────
_PARULA_RAW_LUT = np.array([
    [0.2081, 0.1761, 0.5921], [0.2116, 0.1884, 0.6196], [0.2150, 0.2003, 0.6471],
    [0.2185, 0.2121, 0.6745], [0.2219, 0.2238, 0.7020], [0.2254, 0.2355, 0.7294],
    [0.2288, 0.2470, 0.7569], [0.2323, 0.2586, 0.7843], [0.2357, 0.2701, 0.8039],
    [0.2392, 0.2815, 0.8196], [0.2426, 0.2929, 0.8353], [0.2461, 0.3042, 0.8510],
    [0.2495, 0.3155, 0.8667], [0.2529, 0.3268, 0.8824], [0.2564, 0.3382, 0.8980],
    [0.2598, 0.3496, 0.9137], [0.2633, 0.3609, 0.9294], [0.2667, 0.3722, 0.9451],
    [0.2702, 0.3835, 0.9569], [0.2736, 0.3948, 0.9686], [0.2771, 0.4062, 0.9804],
    [0.2805, 0.4175, 0.9922], [0.2840, 0.4288, 1.0000], [0.2874, 0.4400, 0.9922],
    [0.2909, 0.4512, 0.9804], [0.2943, 0.4623, 0.9725], [0.2978, 0.4734, 0.9686],
    [0.3012, 0.4845, 0.9647], [0.3047, 0.4957, 0.9608], [0.3081, 0.5068, 0.9569],
    [0.3116, 0.5179, 0.9529], [0.3150, 0.5291, 0.9490], [0.3185, 0.5402, 0.9451],
    [0.3219, 0.5514, 0.9412], [0.3254, 0.5625, 0.9373], [0.3288, 0.5736, 0.9333],
    [0.3323, 0.5848, 0.9294], [0.3357, 0.5959, 0.9255], [0.3392, 0.6071, 0.9216],
    [0.3426, 0.6182, 0.9176], [0.3461, 0.6294, 0.9137], [0.3495, 0.6405, 0.9098],
    [0.3530, 0.6516, 0.9059], [0.3564, 0.6628, 0.9020], [0.3598, 0.6739, 0.8980],
    [0.3633, 0.6851, 0.8941], [0.3667, 0.6962, 0.8902], [0.3702, 0.7074, 0.8863],
    [0.3736, 0.7185, 0.8824], [0.3771, 0.7297, 0.8784], [0.3805, 0.7408, 0.8745],
    [0.3840, 0.7520, 0.8706], [0.3874, 0.7631, 0.8667], [0.3909, 0.7743, 0.8627],
    [0.3943, 0.7854, 0.8588], [0.3978, 0.7966, 0.8549], [0.4012, 0.8077, 0.8510],
    [0.4047, 0.8188, 0.8471], [0.4081, 0.8300, 0.8431], [0.4116, 0.8411, 0.8392],
    [0.4150, 0.8523, 0.8353], [0.4185, 0.8634, 0.8314], [0.4219, 0.8746, 0.8275],
    [0.4254, 0.8857, 0.8235], [0.4288, 0.8969, 0.8196], [0.4323, 0.9080, 0.8157],
    [0.4357, 0.9192, 0.8118], [0.4392, 0.9300, 0.8039], [0.4426, 0.9397, 0.7961],
    [0.4461, 0.9495, 0.7882], [0.4495, 0.9592, 0.7804], [0.4530, 0.9690, 0.7725],
    [0.4564, 0.9787, 0.7647], [0.4599, 0.9884, 0.7569], [0.4633, 0.9941, 0.7529],
    [0.4668, 0.9957, 0.7490], [0.4702, 0.9973, 0.7451], [0.4737, 0.9988, 0.7412],
    [0.4771, 1.0000, 0.7373], [0.4806, 0.9988, 0.7333], [0.4840, 0.9973, 0.7294],
    [0.4875, 0.9957, 0.7255], [0.4909, 0.9941, 0.7216], [0.4944, 0.9925, 0.7176],
    [0.4978, 0.9910, 0.7137], [0.5013, 0.9894, 0.7098], [0.5047, 0.9878, 0.7059],
    [0.5082, 0.9863, 0.7020], [0.5116, 0.9847, 0.6980], [0.5151, 0.9831, 0.6941],
    [0.5185, 0.9816, 0.6902], [0.5220, 0.9800, 0.6863], [0.5254, 0.9784, 0.6824],
    [0.5289, 0.9769, 0.6784], [0.5323, 0.9753, 0.6745], [0.5358, 0.9737, 0.6706],
    [0.5392, 0.9722, 0.6667], [0.5427, 0.9706, 0.6627], [0.5461, 0.9690, 0.6588],
    [0.5496, 0.9675, 0.6549], [0.5531, 0.9659, 0.6510], [0.5565, 0.9643, 0.6471],
    [0.5600, 0.9627, 0.6431], [0.5634, 0.9612, 0.6392], [0.5669, 0.9596, 0.6353],
    [0.5703, 0.9580, 0.6314], [0.5738, 0.9565, 0.6275], [0.5772, 0.9549, 0.6235],
    [0.5807, 0.9534, 0.6196], [0.5841, 0.9518, 0.6157], [0.5876, 0.9502, 0.6118],
    [0.5910, 0.9486, 0.6078], [0.5945, 0.9471, 0.6039], [0.5979, 0.9455, 0.6000],
    [0.6014, 0.9439, 0.5961], [0.6048, 0.9424, 0.5922], [0.6083, 0.9408, 0.5882],
    [0.6117, 0.9392, 0.5843], [0.6152, 0.9377, 0.5804], [0.6186, 0.9361, 0.5765],
    [0.6221, 0.9345, 0.5725], [0.6255, 0.9329, 0.5686], [0.6290, 0.9314, 0.5647],
    [0.6324, 0.9298, 0.5608], [0.6359, 0.9282, 0.5569], [0.6393, 0.9267, 0.5529],
    [0.6428, 0.9251, 0.5490], [0.6462, 0.9235, 0.5451], [0.6497, 0.9220, 0.5412],
    [0.6531, 0.9204, 0.5373], [0.6566, 0.9188, 0.5333], [0.6600, 0.9173, 0.5294],
    [0.6635, 0.9157, 0.5255], [0.6669, 0.9141, 0.5216], [0.6704, 0.9125, 0.5176],
    [0.6738, 0.9110, 0.5137], [0.6773, 0.9094, 0.5098], [0.6807, 0.9078, 0.5059],
    [0.6842, 0.9063, 0.5020], [0.6876, 0.9047, 0.4980], [0.6911, 0.9031, 0.4941],
    [0.6945, 0.9016, 0.4902], [0.6980, 0.9000, 0.4863], [0.7014, 0.8984, 0.4824],
    [0.7049, 0.8969, 0.4784], [0.7083, 0.8953, 0.4745], [0.7118, 0.8937, 0.4706],
    [0.7152, 0.8922, 0.4667], [0.7187, 0.8906, 0.4627], [0.7221, 0.8890, 0.4588],
    [0.7256, 0.8875, 0.4549], [0.7290, 0.8859, 0.4510], [0.7325, 0.8843, 0.4471],
    [0.7359, 0.8828, 0.4431], [0.7394, 0.8812, 0.4392], [0.7428, 0.8796, 0.4353],
    [0.7463, 0.8780, 0.4314], [0.7497, 0.8765, 0.4275], [0.7532, 0.8749, 0.4235],
    [0.7566, 0.8733, 0.4196], [0.7601, 0.8718, 0.4157], [0.7635, 0.8702, 0.4118],
    [0.7670, 0.8686, 0.4078], [0.7704, 0.8671, 0.4039], [0.7739, 0.8655, 0.4000],
    [0.7773, 0.8639, 0.3961], [0.7808, 0.8624, 0.3922], [0.7842, 0.8608, 0.3882],
    [0.7877, 0.8592, 0.3843], [0.7911, 0.8577, 0.3804], [0.7946, 0.8561, 0.3765],
    [0.7980, 0.8545, 0.3725], [0.8015, 0.8529, 0.3686], [0.8049, 0.8514, 0.3647],
    [0.8084, 0.8498, 0.3608], [0.8118, 0.8482, 0.3569], [0.8153, 0.8467, 0.3529],
    [0.8187, 0.8451, 0.3490], [0.8222, 0.8435, 0.3451], [0.8256, 0.8420, 0.3412],
    [0.8291, 0.8404, 0.3373], [0.8325, 0.8388, 0.3333], [0.8360, 0.8373, 0.3294],
    [0.8394, 0.8357, 0.3255], [0.8429, 0.8341, 0.3216], [0.8463, 0.8326, 0.3176],
    [0.8498, 0.8310, 0.3137], [0.8532, 0.8294, 0.3098], [0.8567, 0.8279, 0.3059],
    [0.8601, 0.8263, 0.3020], [0.8636, 0.8247, 0.2980], [0.8670, 0.8231, 0.2941],
    [0.8705, 0.8216, 0.2902], [0.8739, 0.8200, 0.2863], [0.8774, 0.8184, 0.2823],
    [0.8808, 0.8169, 0.2784], [0.8843, 0.8153, 0.2745], [0.8877, 0.8137, 0.2706],
    [0.8912, 0.8122, 0.2667], [0.8946, 0.8106, 0.2627], [0.8981, 0.8090, 0.2588],
    [0.9015, 0.8075, 0.2549], [0.9050, 0.8059, 0.2510], [0.9084, 0.8043, 0.2470],
    [0.9119, 0.8027, 0.2431], [0.9153, 0.8012, 0.2392], [0.9188, 0.7996, 0.2353],
    [0.9222, 0.7980, 0.2314], [0.9257, 0.7965, 0.2274], [0.9291, 0.7949, 0.2235],
], dtype=np.float32)
_PARULA_LUT = _resample_lut_to_256(_PARULA_RAW_LUT)

# ── TURBO COLORMAP (raw samples, normalized 0–1, Google's published curve) ──────
_TURBO_RAW_LUT = np.array([
    [0.18995, 0.07176, 0.23217], [0.19483, 0.08339, 0.26149], [0.19956, 0.09498, 0.29024],
    [0.20415, 0.10652, 0.31844], [0.20860, 0.11802, 0.34607], [0.21291, 0.12947, 0.37314],
    [0.21708, 0.14087, 0.39964], [0.22111, 0.15223, 0.42558], [0.22500, 0.16354, 0.45096],
    [0.22873, 0.17481, 0.47578], [0.23231, 0.18603, 0.50004], [0.23571, 0.19720, 0.52373],
    [0.23895, 0.20833, 0.54686], [0.24202, 0.21941, 0.56942], [0.24492, 0.23044, 0.59142],
    [0.24765, 0.24143, 0.61286], [0.25020, 0.25237, 0.63374], [0.25257, 0.26325, 0.65406],
    [0.25477, 0.27410, 0.67381], [0.25678, 0.28490, 0.69300], [0.25862, 0.29565, 0.71162],
    [0.26027, 0.30637, 0.72968], [0.26175, 0.31706, 0.74718], [0.26305, 0.32771, 0.76412],
    [0.26428, 0.33832, 0.78050], [0.26533, 0.34888, 0.79631], [0.26621, 0.35942, 0.81156],
    [0.26692, 0.36992, 0.82624], [0.26747, 0.38038, 0.84037], [0.26785, 0.39081, 0.85393],
    [0.26805, 0.40122, 0.86692], [0.26810, 0.41159, 0.87936], [0.26798, 0.42195, 0.89123],
    [0.26771, 0.43228, 0.90254], [0.26727, 0.44259, 0.91328], [0.26667, 0.45288, 0.92347],
    [0.26592, 0.46314, 0.93309], [0.26500, 0.47340, 0.94214], [0.26383, 0.48364, 0.95064],
    [0.26254, 0.49388, 0.95857], [0.26108, 0.50412, 0.96594], [0.25946, 0.51436, 0.97275],
    [0.25768, 0.52461, 0.97899], [0.25573, 0.53487, 0.98461], [0.25360, 0.54513, 0.98981],
    [0.25131, 0.55540, 0.99427], [0.24880, 0.56567, 0.99823], [0.24616, 0.57595, 0.100177],
    [0.24334, 0.58623, 1.00000], [0.24037, 0.59652, 0.99788], [0.23724, 0.60682, 0.99542],
    [0.23391, 0.61712, 0.99251], [0.23042, 0.62744, 0.98910], [0.22697, 0.63776, 0.98522],
    [0.22338, 0.64810, 0.98088], [0.21974, 0.65846, 0.97598], [0.21594, 0.66883, 0.97050],
    [0.21211, 0.67920, 0.96450], [0.20813, 0.68958, 0.95786], [0.20410, 0.69997, 0.95057],
    [0.19993, 0.71038, 0.94260], [0.19571, 0.72080, 0.93391], [0.19135, 0.73121, 0.92447],
    [0.18694, 0.74162, 0.91425], [0.18240, 0.75203, 0.90319], [0.17781, 0.76244, 0.89124],
    [0.17310, 0.77285, 0.87835], [0.16834, 0.78326, 0.86450], [0.16345, 0.79366, 0.84966],
    [0.15849, 0.80405, 0.83375], [0.15337, 0.81444, 0.81676], [0.14821, 0.82481, 0.79860],
    [0.14289, 0.83518, 0.77921], [0.13758, 0.84551, 0.75853], [0.13212, 0.85583, 0.73639],
    [0.12667, 0.86612, 0.71275], [0.12109, 0.87638, 0.68757], [0.11561, 0.88660, 0.66057],
    [0.11000, 0.89677, 0.63188], [0.10447, 0.90690, 0.60111], [0.09878, 0.91698, 0.56823],
    [0.09318, 0.92700, 0.53329], [0.08735, 0.93697, 0.49644], [0.08174, 0.94688, 0.45726],
    [0.07590, 0.95673, 0.41577], [0.07046, 0.96651, 0.37186], [0.06456, 0.97622, 0.32567],
    [0.05923, 0.98585, 0.27712], [0.05324, 0.99540, 0.22629], [0.04830, 1.00000, 0.17328],
    [0.04203, 0.99418, 0.12841], [0.03640, 0.98683, 0.09191], [0.03038, 0.97798, 0.06508],
    [0.02545, 0.96764, 0.04789], [0.02009, 0.95577, 0.04036], [0.01599, 0.94240, 0.04266],
    [0.01231, 0.92754, 0.05455], [0.00876, 0.91117, 0.07624], [0.00549, 0.89331, 0.10777],
    [0.00235, 0.87395, 0.14914], [0.00000, 0.85306, 0.20040], [0.00000, 0.83061, 0.26160],
    [0.00000, 0.80658, 0.32276], [0.00000, 0.78092, 0.38391], [0.00000, 0.75359, 0.44505],
    [0.00000, 0.72456, 0.50620], [0.00000, 0.69378, 0.56734], [0.00000, 0.66121, 0.62849],
    [0.00000, 0.62678, 0.68964], [0.00000, 0.59046, 0.75079], [0.00000, 0.55219, 0.81194],
    [0.00000, 0.51191, 0.87309], [0.00000, 0.46952, 0.93423], [0.00000, 0.42485, 0.99489],
    [0.00000, 0.37782, 1.00000], [0.00000, 0.32841, 0.98508], [0.00000, 0.27659, 0.96015],
    [0.00000, 0.22242, 0.93521], [0.00000, 0.16590, 0.91028], [0.00000, 0.10708, 0.88535],
], dtype=np.float32)
_TURBO_LUT = _resample_lut_to_256(_TURBO_RAW_LUT)


def _get_color_for_name(color_name: str | None) -> tuple[float, float, float, float] | None:
    """Convert color name to RGBA tuple (0-1 range). Returns None if color_name is None."""
    if color_name is None:
        return None
    color_map = {
        "red": (1.0, 0.0, 0.0, 1.0),
        "green": (0.0, 1.0, 0.0, 1.0),
        "blue": (0.0, 0.0, 1.0, 1.0),
        "cyan": (0.0, 1.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0, 1.0),
        "yellow": (1.0, 1.0, 0.0, 1.0),
        "white": (1.0, 1.0, 1.0, 1.0),
    }
    return color_map.get(color_name.lower())


def _build_colormap(name: str) -> pg.ColorMap:
    """Build a pyqtgraph ColorMap from a MATLAB/matplotlib-compatible name.

    Supports 15 names: parula, turbo, hsv, hot, cool, spring, summer, autumn,
    winter, gray, bone, copper, pink, jet, invgray.

    parula and turbo use hardcoded lookup tables. Others use matplotlib colormaps
    (matplotlib must be available).
    """
    name = name.lower()

    # Hardcoded LUTs for perceptual colormaps
    if name == "parula":
        # Convert from 0-1 to 0-255 for pg.ColorMap
        lut_255 = ((_PARULA_LUT * 255).astype(np.uint8))
        return pg.ColorMap(pos=np.linspace(0, 1, len(lut_255)), color=lut_255)

    if name == "turbo":
        lut_255 = ((_TURBO_LUT * 255).astype(np.uint8))
        return pg.ColorMap(pos=np.linspace(0, 1, len(lut_255)), color=lut_255)

    # Special case: invgray is gray reversed
    if name == "invgray":
        try:
            import matplotlib
        except ImportError:
            raise ImportError("matplotlib required for standard colormaps") from None
        cmap = matplotlib.colormaps["gray"]
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        lut = lut[::-1]  # reverse
        return pg.ColorMap(pos=np.linspace(0, 1, len(lut)), color=lut)

    # Standard matplotlib colormaps. matplotlib.colormaps is a plain registry
    # lookup (not matplotlib.cm.get_cmap, removed in matplotlib >= 3.9) and
    # doesn't touch pyplot or backend resolution, so it's safe to call from a
    # process that already has a Qt event loop running.
    try:
        import matplotlib
    except ImportError:
        raise ImportError("matplotlib required for standard colormaps") from None
    try:
        cmap = matplotlib.colormaps[name]
    except KeyError:
        raise ValueError(f"Unknown colormap: {name}") from None

    # Convert matplotlib colormap to pyqtgraph (256-point LUT, RGB only)
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return pg.ColorMap(pos=np.linspace(0, 1, len(lut)), color=lut)


class SpectrogramWidget(QWidget):
    """Waveform (top) + spectrogram (bottom) with optional label overlays.

    Layout mirrors the MATLAB app: waveform on top, spectrogram below,
    both sharing the same time axis.
    """

    # Signals for interactive features
    boundary_dragged = pyqtSignal(str, float)  # ("start" or "end", new_time)
    spectrogram_right_clicked = pyqtSignal(float)  # time value

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        # ── Waveform plot — row 0 (TOP) ───────────────────────────────────
        self._wave_plot: pg.PlotItem = self._glw.addPlot(row=0, col=0)
        self._wave_plot.showGrid(x=True, y=False, alpha=0.3)
        self._wave_plot.getAxis("bottom").setStyle(showValues=False)
        self._wave_curve = self._wave_plot.plot()

        # ── Spectrogram plot — row 1 (BOTTOM) ────────────────────────────
        self._spec_plot: pg.PlotItem = self._glw.addPlot(row=1, col=0)
        self._spec_plot.showGrid(x=False, y=True, alpha=0.2)

        self._img = pg.ImageItem()
        self._img.setColorMap(_GRAY_CM)
        self._spec_plot.addItem(self._img)

        # Share x-axis so both plots pan/zoom together
        self._spec_plot.setXLink(self._wave_plot)

        # Height ratio 1:4  (waveform is narrow, spectrogram is tall)
        self._glw.ci.layout.setRowStretchFactor(0, 1)
        self._glw.ci.layout.setRowStretchFactor(1, 4)

        self._label_items: list = []   # items added to spec or wave plot

        # Boundary drag lines (for label edit mode)
        self._boundary_lines: dict[str, pg.InfiniteLine] = {}  # "start" → line, "end" → line
        self._current_colormap_name: str = "parula"
        self._current_colormap: pg.ColorMap = _build_colormap("parula")

        # Connect mouse click handlers for right-click label creation
        self._spec_plot.scene().sigMouseClicked.connect(self._on_plot_mouse_click)
        self._wave_plot.scene().sigMouseClicked.connect(self._on_plot_mouse_click)

        # Resolved here (after QApplication exists) so colors match the
        # current mode; call again via refresh_theme() if it changes live.
        self.refresh_theme()

    def refresh_theme(self) -> None:
        """(Re-)apply theme-dependent colors. Existing label overlays keep
        their old pens until the next display() call redraws them."""
        self._pen_det = pg.mkPen(theme.DETECTED_COLOR, width=2.0)
        self._pen_ref = pg.mkPen(theme.REFERENCE_COLOR, width=2.0)
        self._col_det = pg.mkColor(theme.DETECTED_COLOR).getRgb()
        self._col_ref = pg.mkColor(theme.REFERENCE_COLOR).getRgb()

        self._glw.setBackground(theme.SURFACE)
        label_style = {"color": theme.TEXT_SECONDARY, "font-size": "11px"}
        self._wave_plot.setLabel("left", "Amplitude", **label_style)
        self._spec_plot.setLabel("left", "Frequency (kHz)", **label_style)
        self._spec_plot.setLabel("bottom", "Time (s)", **label_style)
        for plot in (self._wave_plot, self._spec_plot):
            for axis in ("left", "bottom"):
                plot.getAxis(axis).setPen(theme.BORDER_HOVER)
                plot.getAxis(axis).setTextPen(theme.TEXT_SECONDARY)
        self._wave_curve.setPen(pg.mkPen(theme.CURVE, width=1))

    # ── Public API ────────────────────────────────────────────────────────

    def display(
        self,
        samples: np.ndarray,
        fs: int,
        t_start: float,
        t_end: float,
        fmin_hz: float = 40_000.0,
        fmax_hz: float = 120_000.0,
        nperseg: int = 1024,
        noverlap: int = 512,
        detected_labels: list[Label] | None = None,
        reference_labels: list[Label] | None = None,
        show_detected: bool = True,
        show_reference: bool = True,
        show_pitch: bool = True,
        colormap_name: str | None = None,
        detected_color_name: str | None = None,
        reference_color_name: str | None = None,
    ) -> None:
        # ── Handle colormap change ────────────────────────────────────────
        if colormap_name is not None:
            self.set_colormap(colormap_name)

        # ── Handle label color overrides ───────────────────────────────────
        pen_det = self._pen_det
        col_det = self._col_det
        pen_ref = self._pen_ref
        col_ref = self._col_ref

        if detected_color_name is not None:
            rgba = _get_color_for_name(detected_color_name)
            if rgba:
                pen_det = pg.mkPen(rgba, width=2.0)
                col_det = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), int(rgba[3]*255))

        if reference_color_name is not None:
            rgba = _get_color_for_name(reference_color_name)
            if rgba:
                pen_ref = pg.mkPen(rgba, width=2.0)
                col_ref = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), int(rgba[3]*255))

        i0 = max(0, int(t_start * fs))
        i1 = min(len(samples), int(t_end * fs))
        chunk = samples[i0:i1]
        if len(chunk) < nperseg:
            return

        # ── Spectrogram ───────────────────────────────────────────────────
        noverlap = min(noverlap, nperseg - 1)
        overlap_factor = noverlap / nperseg
        f, t_rel, Sxx_db = compute_stft(chunk, fs, nperseg, overlap_factor)
        f_sub, Sxx_sub = band_restrict(f, Sxx_db, fmin_hz, fmax_hz)
        f_khz = f_sub / 1_000.0
        t_abs = t_rel + t_start  # absolute time axis

        # ImageItem column-major: shape (T, F)
        img_data = Sxx_sub.T.astype(np.float32)
        vmin = float(np.percentile(img_data, 2))
        vmax = float(np.percentile(img_data, 99.5))

        self._img.setImage(img_data, autoLevels=False, levels=[vmin, vmax])
        self._img.setColorMap(self._current_colormap)  # Apply current colormap
        self._img.setRect(
            QRectF(
                float(t_abs[0]),
                float(f_khz[0]),
                float(t_abs[-1] - t_abs[0]),
                float(f_khz[-1] - f_khz[0]),
            )
        )
        self._spec_plot.setRange(
            xRange=[t_start, t_end],
            yRange=[fmin_hz / 1_000.0, fmax_hz / 1_000.0],
            padding=0.0,
        )

        # ── Waveform ──────────────────────────────────────────────────────
        n = len(chunk)
        t_wave = np.linspace(t_start, t_start + n / fs, n, endpoint=False)
        step = max(1, n // 50_000)
        self._wave_curve.setData(t_wave[::step], chunk[::step])
        self._wave_plot.setRange(xRange=[t_start, t_end], padding=0.0)

        # ── Label overlays ────────────────────────────────────────────────
        self._clear_labels()
        fmax_khz = fmax_hz / 1_000.0
        fmin_khz = fmin_hz / 1_000.0

        labels_for_pitch: list[Label] = []
        if show_detected and detected_labels:
            self._draw_labels(detected_labels, t_start, t_end, pen_det, col_det, fmin_khz, fmax_khz)
            labels_for_pitch.extend(detected_labels)
        if show_reference and reference_labels:
            self._draw_labels(reference_labels, t_start, t_end, pen_ref, col_ref, fmin_khz, fmax_khz)
            labels_for_pitch.extend(reference_labels)

        if show_pitch and labels_for_pitch:
            self._draw_pitch_trace(labels_for_pitch, f_khz, Sxx_sub, t_abs, t_start, t_end)

    def clear(self) -> None:
        self._img.clear()
        self._wave_curve.setData([], [])
        self._clear_labels()

    def set_colormap(self, name: str) -> None:
        """Set the spectrogram colormap by name.

        Supported names: parula (default), turbo, hsv, hot, cool, spring, summer,
        autumn, winter, gray, bone, copper, pink, jet, invgray.

        The colormap persists across display() calls until changed again.
        """
        self._current_colormap_name = name
        self._current_colormap = _build_colormap(name)
        self._img.setColorMap(self._current_colormap)

    def enable_boundary_drag(
        self,
        start_time: float,
        end_time: float,
        fmin_khz: float,
        fmax_khz: float,
    ) -> None:
        """Draw two draggable vertical lines at start_time and end_time.

        The lines span the full label frequency range [fmin_khz, fmax_khz] and
        are movable=True. Emits boundary_dragged signal on drag completion.

        Calling this again replaces any previous drag lines (idempotent).
        """
        # Clean up any existing lines
        self.disable_boundary_drag()

        # Create start line
        start_line = pg.InfiniteLine(
            pos=start_time,
            angle=90,
            pen=pg.mkPen(theme.DETECTED_COLOR, width=2.0, style=Qt.PenStyle.DashLine),
            movable=True,
            name="start",
        )
        start_line.sigPositionChangeFinished.connect(
            lambda: self._on_boundary_line_moved("start", start_line)
        )
        self._spec_plot.addItem(start_line)
        self._boundary_lines["start"] = start_line

        # Create end line
        end_line = pg.InfiniteLine(
            pos=end_time,
            angle=90,
            pen=pg.mkPen(theme.REFERENCE_COLOR, width=2.0, style=Qt.PenStyle.DashDotLine),
            movable=True,
            name="end",
        )
        end_line.sigPositionChangeFinished.connect(
            lambda: self._on_boundary_line_moved("end", end_line)
        )
        self._spec_plot.addItem(end_line)
        self._boundary_lines["end"] = end_line

    def disable_boundary_drag(self) -> None:
        """Remove any draggable boundary lines."""
        for key, line in list(self._boundary_lines.items()):
            self._spec_plot.removeItem(line)
            self._boundary_lines.pop(key)

    # ── Private ───────────────────────────────────────────────────────────

    def _clear_labels(self) -> None:
        for plot, item in self._label_items:
            plot.removeItem(item)
        self._label_items.clear()

    def _draw_labels(
        self,
        labels: list[Label],
        t_start: float,
        t_end: float,
        pen: object,
        color: tuple,
        fmin_khz: float,
        fmax_khz: float,
    ) -> None:
        """Draw start/end lines + text for each label visible in the segment."""
        text_y = fmax_khz - (fmax_khz - fmin_khz) * 0.04  # just inside top edge

        for lbl in labels:
            if lbl.end_time < t_start or lbl.start_time > t_end:
                continue

            # Vertical lines at start and end on spectrogram
            for t in (lbl.start_time, lbl.end_time):
                line = pg.InfiniteLine(pos=t, angle=90, pen=pen, movable=False)
                self._spec_plot.addItem(line)
                self._label_items.append((self._spec_plot, line))

            # Horizontal tick at the top connecting start → end
            top_line = pg.PlotDataItem(
                [lbl.start_time, lbl.end_time],
                [fmax_khz * 0.995, fmax_khz * 0.995],
                pen=pen,
            )
            self._spec_plot.addItem(top_line)
            self._label_items.append((self._spec_plot, top_line))

            # Text label
            label_text = lbl.label if lbl.label else ""
            if label_text:
                txt = pg.TextItem(text=label_text, color=color, anchor=(0.0, 1.0))
                txt.setPos(lbl.start_time, text_y)
                self._spec_plot.addItem(txt)
                self._label_items.append((self._spec_plot, txt))

            # Thin lines on waveform plot too
            for t in (lbl.start_time, lbl.end_time):
                wline = pg.InfiniteLine(pos=t, angle=90, pen=pen, movable=False)
                self._wave_plot.addItem(wline)
                self._label_items.append((self._wave_plot, wline))

    def _draw_pitch_trace(
        self,
        labels: list[Label],
        f_khz: np.ndarray,
        Sxx_sub: np.ndarray,
        t_abs: np.ndarray,
        t_start: float,
        t_end: float,
    ) -> None:
        """Overlay the dominant-frequency contour for each visible call.

        Traces the peak-power frequency bin per STFT column (the same
        "DomFreq" measure used for ML features) within each label's time
        span, so the line follows the pitch only where a squeak is playing.
        """
        peak_idx = np.argmax(Sxx_sub, axis=0)
        pitch_khz = f_khz[peak_idx]

        for lbl in labels:
            if lbl.end_time < t_start or lbl.start_time > t_end:
                continue

            mask = (t_abs >= lbl.start_time) & (t_abs <= lbl.end_time)
            if np.count_nonzero(mask) < 2:
                continue

            trace_t = t_abs[mask]
            trace_f = pitch_khz[mask]
            if len(trace_f) >= 3:
                trace_f = median_filter(trace_f, size=3, mode="nearest")

            curve = pg.PlotDataItem(trace_t, trace_f, pen=_PEN_PITCH)
            self._spec_plot.addItem(curve)
            self._label_items.append((self._spec_plot, curve))

    def _on_boundary_line_moved(self, line_type: str, line: pg.InfiniteLine) -> None:
        """Handle boundary line drag completion."""
        new_time = line.value()
        self.boundary_dragged.emit(line_type, new_time)

    def _on_plot_mouse_click(self, event: object) -> None:
        """Handle right-click on spectrogram or waveform plots."""
        # event is a MouseClickEvent from pyqtgraph
        # Check if it's a right-click
        if event.button() != Qt.MouseButton.RightButton:
            return

        # Map scene position to view coordinates of the spectrogram plot
        # (both plots share x-axis, so time coordinate is valid for both)
        try:
            # Get the scene position and map to the spectrogram viewbox
            scene_pos = event.scenePos()
            vb = self._spec_plot.getViewBox()
            data_pos = vb.mapSceneToView(scene_pos)
            clicked_time = data_pos.x()
            self.spectrogram_right_clicked.emit(clicked_time)
        except Exception:
            # Silently ignore if mapping fails
            pass
