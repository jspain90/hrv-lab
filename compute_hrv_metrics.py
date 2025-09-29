"""
HRV Frequency-Domain Analysis (Elite HRV–style)
- Reads .txt or .csv (auto-delimiter or --sep)
- Bands: VLF 0.003–0.04, LF 0.04–0.15, HF 0.15–0.40 Hz
- TP = LF + HF (excludes VLF)
- Prints metrics to stdout
- Optional: --save_plot <png>; --out_csv <csv>
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline 
from typing import Tuple, Dict, Optional
import sys

PIPELINE_VERSION = "hrv/1.0.0"

def highpass_rr(rr_u_ms, fs_interp: float, fc_hz: float = 0.003, order: int = 2):
    # zero-phase high-pass to avoid phase distortions
    nyq = 0.5 * fs_interp
    wn = fc_hz / nyq
    b, a = butter(order, wn, btype="high", analog=False)
    return filtfilt(b, a, rr_u_ms)

def interpolate_tachogram(rr_s: np.ndarray, fs_interp: float = 4.0, method: str = "cubic") -> Tuple[np.ndarray, np.ndarray]:
    t_beats = np.cumsum(rr_s); t_beats -= t_beats[0]
    t_uniform = np.arange(0, t_beats[-1], 1.0/fs_interp)
    if method == "cubic":
        try:
            cs = CubicSpline(t_beats, rr_s, bc_type="natural")
            rr_uniform = cs(t_uniform)
        except Exception:
            rr_uniform = np.interp(t_uniform, t_beats, rr_s)
    else:
        rr_uniform = np.interp(t_uniform, t_beats, rr_s)
    return t_uniform, rr_uniform

def welch_psd(x: np.ndarray, fs: float, window: str = "hamming",
              nperseg: int | None = None, nfft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(x))
    if n < 8:
        # too short for Welch; make a trivial 2-bin PSD to avoid crashes
        f = np.linspace(0, fs/2, num=2)
        Pxx = np.full_like(f, np.nan, dtype=float)
        return f, Pxx

    # choose a sensible segment length given the actual data
    target = 1024 if (nperseg is None) else int(nperseg)
    seg = min(max(256, n // 2), target)  # at least 256, at most half the data, and not above target
    if seg > n:
        seg = n  # ultimate guard

    # pick an nfft that’s >= seg
    if nfft is None:
        nfft_eff = max(2048, 1 << (seg - 1).bit_length())  # next power of two
    else:
        nfft_eff = max(int(nfft), seg)

    noverlap = seg // 2
    if noverlap >= seg:
        noverlap = seg - 1  # SciPy requirement

    f, Pxx = signal.welch(
        x, fs=fs, window=window, nperseg=seg, noverlap=noverlap, nfft=nfft_eff,
        detrend=False, scaling="density"
    )
    return f, Pxx

def mad_based_outlier_mask(x: np.ndarray, thresh: float = 4.0) -> np.ndarray:
    m = np.median(x)
    mad = np.median(np.abs(x - m)) + 1e-12
    z = 0.6745 * (x - m) / mad
    return np.abs(z) < thresh

def replace_outliers_with_interp(rr: np.ndarray) -> np.ndarray:
    keep = mad_based_outlier_mask(rr, thresh=4.0)
    rr_clean = rr.copy()
    if not np.all(keep):
        idx = np.arange(len(rr))
        rr_clean[~keep] = np.interp(idx[~keep], idx[keep], rr[keep])
    return rr_clean

def smoothness_priors_detrend(x: np.ndarray, lam: float) -> np.ndarray:
    N = len(x)
    I = np.eye(N)
    D = np.zeros((N-2, N))
    for i in range(N-2):
        D[i, i]   = 1.0
        D[i, i+1] = -2.0
        D[i, i+2] = 1.0
    DT_D = D.T @ D
    A = I + (lam**2) * DT_D
    z = np.linalg.solve(A, x)
    detrended = x - z
    return detrended

def band_power(f: np.ndarray, Pxx: np.ndarray, f_low: float, f_high: float) -> float:
    band = (f >= f_low) & (f <= f_high + 1e-12)
    if not np.any(band):
        return 0.0
    return float(np.trapezoid(Pxx[band], f[band]))

def peak_frequency(f: np.ndarray, Pxx: np.ndarray, f_low: float, f_high: float) -> float:
    band = (f >= f_low) & (f < f_high)
    if not np.any(band):
        return float("nan")
    idx = np.argmax(Pxx[band])
    return float(f[band][idx])

def time_domain_metrics(rr_ms: np.ndarray) -> Dict[str, float]:
    rr_s = rr_ms / 1000.0
    hr = 60.0 / np.mean(rr_s)
    sdnn = np.std(rr_ms, ddof=1)
    diff = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff**2))
    pnn50 = 100.0 * np.mean(np.abs(diff) > 50.0)
    return {
        "MEAN_RR_MS": float(np.mean(rr_ms)),
        "HR_BPM": float(hr),
        "SDNN_MS": float(sdnn),
        "RMSSD_MS": float(rmssd),
        "PNN50_PCT": float(pnn50),
    }

def analyze_rr(rr_ms: np.ndarray,
               fs_interp: float = 4.0,
               lam: float = 300.0,
               detrend: str = "smoothness_priors",
               interp_method: str = "cubic",
               welch_window: str = "hamming",
               welch_nperseg: int = None,
               welch_nfft: int = None,
               hp_fc: float = 0.003,
               hp_order: int = 2):
    td = time_domain_metrics(rr_ms)
    rr_ms_clean = replace_outliers_with_interp(rr_ms)
    rr_s = rr_ms_clean / 1000.0
    _, rr_u = interpolate_tachogram(rr_s, fs_interp=fs_interp, method=interp_method)
    rr_u_ms = rr_u * 1000.0
    if detrend == "smoothness_priors":
        rr_u_ms_dt = smoothness_priors_detrend(rr_u_ms - np.mean(rr_u_ms), lam=lam)
    elif detrend == "linear":
        rr_u_ms_dt = signal.detrend(rr_u_ms, type='linear')
    elif detrend == "highpass":
        rr_u_ms_dt = highpass_rr(rr_u_ms - np.mean(rr_u_ms),
                                fs_interp=fs_interp,
                                fc_hz=hp_fc,
                                order=hp_order)
    else:
        rr_u_ms_dt = rr_u_ms - np.mean(rr_u_ms)
    f, Pxx = welch_psd(rr_u_ms_dt, fs=fs_interp, window=welch_window, nperseg=welch_nperseg, nfft=welch_nfft)

    VLF_LOW, VLF_HIGH = 0.003, 0.04
    LF_LOW,  LF_HIGH  = 0.04, 0.15
    HF_LOW,  HF_HIGH  = 0.15, 0.40

    vlf = band_power(f, Pxx, VLF_LOW, VLF_HIGH)
    lf  = band_power(f, Pxx, LF_LOW,  LF_HIGH)
    hf  = band_power(f, Pxx, HF_LOW,  HF_HIGH)
    tp  = lf + hf

    lf_hf = lf / hf if hf > 0 else float("inf")
    vlf_peak = peak_frequency(f, Pxx, VLF_LOW, VLF_HIGH)
    lf_peak = peak_frequency(f, Pxx, LF_LOW, LF_HIGH)
    hf_peak = peak_frequency(f, Pxx, HF_LOW, HF_HIGH)
    
    denom = band_power(f, Pxx, 0.0, 0.15)
    slow_fraction = band_power(f, Pxx, 0.0, 0.06) / denom if denom > 0 else float("nan")
    duration_sec = float(np.sum(rr_ms) / 1000.0)

    metrics = {
        **td,
        "duration_sec": duration_sec,
        "TotalPower_MS2": float(tp),
        "VLF_MS2": float(vlf),
        "LF_MS2": float(lf),
        "HF_MS2": float(hf),
        "LF_HF_Ratio": float(lf_hf),
        "VLF_PEAK_HZ": float(vlf_peak),
        "LF_PEAK_HZ": float(lf_peak),
        "HF_PEAK_HZ": float(hf_peak),
        "BREATH_RATE_BRPM": float(hf_peak * 60.0) if hf_peak == hf_peak else float("nan"),
        "SLOW_FRACTION_0_00_0_06_OVER_0_00_0_15": float(slow_fraction),
        # configuration / provenance
        "FS_INTERP_HZ": float(fs_interp),
        "DETREND": detrend,
        "LAMBDA": float(lam),
        "WINDOW": welch_window,
        "NPERSEG": int(welch_nperseg) if welch_nperseg is not None else None,
        "NFFT": int(welch_nfft) if welch_nfft is not None else None,
        "INTERP": interp_method,
        "PIPELINE_VERSION": PIPELINE_VERSION,
    }
    if detrend == "highpass":
        metrics["HP_FC_HZ"] = float(hp_fc)
        metrics["HP_ORDER"] = int(hp_order)
    return metrics, (f, Pxx)

def plot_psd(f: np.ndarray, Pxx: np.ndarray, out_png: Optional[str] = None) -> None:
    if out_png:
        plt.figure(figsize=(8, 4.5))
        plt.plot(f, Pxx)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (ms²/Hz)")
        plt.title("HRV Power Spectral Density (Welch)")
        plt.xlim(0.0, 0.5)
        plt.grid(True, alpha=0.3)
        plt.savefig(out_png, bbox_inches="tight", dpi=150)

def pick_rr_column(df: pd.DataFrame) -> str:
    cleaned = {c.lower().replace(" ", "").replace("(", "").replace(")", ""): c for c in df.columns}
    for key in ["rr", "rrms", "rr_ms", "ibims", "ibi", "rrinterval", "rrintervalms"]:
        if key in cleaned:
            return cleaned[key]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("No numeric column found for RR intervals. Use --col to specify.")
    return num_cols[0]

def read_rr_table(path: str, col: str | None, units: str, sep: str | None = None) -> np.ndarray:
    path = str(path)
    # Always read as headerless; many Elite exports are single-column TXT with no header
    if sep is None:
        # Let pandas split on any whitespace/newlines
        sep = r"\s+"

    df = pd.read_csv(path, engine="python", sep=sep, header=None)
    # If the caller named a column, try to use it; else pick the first numeric col
    if col is not None and col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            # try coercing first column to numeric
            series = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        else:
            series = df[num_cols[0]]

    rr = series.dropna().to_numpy(dtype=float)
    # Units heuristic fallback, but prefer explicit --units
    if units == "ms":
        rr_ms = rr
    elif units == "s":
        rr_ms = rr * 1000.0
    else:
        # heuristic: if median < 5 it's seconds
        med = float(np.nanmedian(rr)) if rr.size else 0.0
        rr_ms = rr * (1000.0 if med < 5.0 else 1.0)

    # Basic sanity: drop non-positive intervals
    rr_ms = rr_ms[rr_ms > 0]
    return rr_ms

def main():
    parser = argparse.ArgumentParser(description="HRV frequency-domain analysis (Elite HRV–style; TXT/CSV friendly).")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--col", type=str, default=None)
    parser.add_argument("--units", type=str, choices=["ms","s"], default="ms")
    parser.add_argument("--fs", type=float, default=4.0)
    parser.add_argument("--lambda", dest="lam", type=float, default=300.0) 
    parser.add_argument("--sep", type=str, default=None)
    parser.add_argument("--save_plot", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)

    #Tuning options
    parser.add_argument("--elite_preset", action="store_true", help="Use Elite-like PSD settings.")
    parser.add_argument("--window", type=str, default="hamming", help="Welch window (hamming|hann|blackman|...); default varies.")
    parser.add_argument("--nperseg", type=int, default=1024, help="Welch nperseg; default 512 for ≥512 samples.")
    parser.add_argument("--nfft", type=int, default=4096, help="Welch nfft; default 4096 for ≥1024 samples.")
    parser.add_argument("--interp", type=str, choices=["linear","cubic"], default="cubic", help="Interpolation method for RR tachogram.")
    parser.add_argument("--detrend", choices=["smoothness_priors","linear","highpass","none"], default="linear")
    parser.add_argument("--hp_fc", type=float, default=0.003, help="High-pass cutoff in Hz")
    parser.add_argument("--hp_order", type=int, default=2, help="Butterworth order")

    #Convert to JSON
    parser.add_argument("--json", action="store_true",
                        help="Emit a single JSON object with metrics to stdout.")
    args = parser.parse_args()

    # derive PSD parameters
    welch_window = args.window if args.window else ("hamming" if args.elite_preset else "hann")
    welch_nperseg = args.nperseg if args.nperseg is not None else (512 if args.elite_preset else None)
    welch_nfft    = args.nfft    if args.nfft    is not None else (4096 if args.elite_preset else None)
    interp_method = args.interp

    if args.input is None:
        # Demo data (5 min synthetic)
        np.random.seed(0)
        duration_s = 300
        hr_bpm = 65.0
        mean_rr = 60.0 / hr_bpm
        t = np.cumsum(np.random.normal(loc=mean_rr, scale=0.02, size=int(duration_s / mean_rr)))
        t -= t[0]
        rr_s = mean_rr + 0.03*np.sin(2*np.pi*0.17*t) + 0.015*np.sin(2*np.pi*0.09*t) + np.random.normal(0, 0.005, size=t.shape)
        rr_ms = rr_s * 1000.0
    else:
        rr_ms = read_rr_table(args.input, args.col, args.units, sep=args.sep)

    metrics, (f, Pxx) = analyze_rr(
        rr_ms,
        fs_interp=args.fs,
        lam=args.lam,
        detrend=args.detrend,
        interp_method=interp_method,
        welch_window=welch_window,
        welch_nperseg=welch_nperseg,
        welch_nfft=welch_nfft,
        hp_fc=args.hp_fc,
        hp_order=args.hp_order
    )

    # Print metrics
    if args.json:
        import json
        # Ensure only JSON hits stdout
        sys.stdout.write(json.dumps(metrics, ensure_ascii=False))
        sys.stdout.flush()
    else:
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # Optional outputs
    if args.out_csv is not None:
        pd.DataFrame([metrics]).to_csv(args.out_csv, index=False)
    if args.save_plot is not None:
        plot_psd(f, Pxx, out_png=args.save_plot)

if __name__ == "__main__":
    main()
