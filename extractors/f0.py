import os
import glob
import argparse
from pathlib import Path

import numpy as np
import librosa
import pyworld
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from tqdm import tqdm


# -----------------------------
# Core F0 extraction
# -----------------------------
def extract_f0_world(y, sr, method="dio", frame_period=10.0, f0_floor=50.0, f0_ceil=800.0):
    """
    WORLD F0 (dio/harvest) + stonemask refinement
    y: float np.ndarray (T,)
    """
    y64 = y.astype(np.float64)

    if method.lower() == "harvest":
        _f0, t = pyworld.harvest(
            y64,
            sr,
            frame_period=frame_period,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )
    else:
        _f0, t = pyworld.dio(
            y64,
            sr,
            frame_period=frame_period,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )

    f0 = pyworld.stonemask(y64, _f0, t, sr)
    return t, f0


# -----------------------------
# Post-processing utils (kept same logic)
# -----------------------------
def savgol_smooth(x, window_size=25, polyorder=2):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if window_size > len(x):
        raise ValueError("window_size must be <= length of input")
    return savgol_filter(x, window_length=window_size, polyorder=polyorder)


def interpolate_short_zeros(f0, max_gap=15):
    f0 = f0.copy()
    length = len(f0)

    valid_indices = np.where(f0 != 0)[0]
    if len(valid_indices) == 0:
        return f0

    valid_start = valid_indices[0]
    valid_end = valid_indices[-1]

    i = valid_start
    while i <= valid_end:
        if f0[i] == 0:
            start = i
            while i <= valid_end and f0[i] == 0:
                i += 1
            end = i
            gap_len = end - start

            if gap_len <= max_gap:
                left = f0[start - 1]
                right = f0[end] if end < length else f0[start - 1]
                f0[start:end] = np.linspace(left, right, gap_len + 2)[1:-1]
        else:
            i += 1
    return f0


def remove_outliers_iqr(f0):
    f0_nonzero = f0[f0 > 0]
    if len(f0_nonzero) == 0:
        return f0

    q15 = np.percentile(f0_nonzero, 15)
    q85 = np.percentile(f0_nonzero, 85)

    lower_bound = q15 / 1.5
    upper_bound = q85 * 1.5

    return np.where((f0 >= lower_bound) & (f0 <= upper_bound), f0, 0)


def remove_isolated_f0_outliers(f0, threshold_hz=30):
    f0 = f0.copy()
    length = len(f0)

    for i in range(1, length - 1):
        if f0[i] == 0:
            continue
        if f0[i - 1] == 0 and f0[i + 1] == 0:
            prev_nonzero = next((f0[j] for j in range(i - 1, -1, -1) if f0[j] > 0), None)
            next_nonzero = next((f0[j] for j in range(i + 1, length) if f0[j] > 0), None)

            nearest_f0 = None
            if prev_nonzero is not None and next_nonzero is not None:
                nearest_f0 = prev_nonzero if abs(prev_nonzero - f0[i]) < abs(next_nonzero - f0[i]) else next_nonzero
            elif prev_nonzero is not None:
                nearest_f0 = prev_nonzero
            elif next_nonzero is not None:
                nearest_f0 = next_nonzero

            if nearest_f0 is not None and abs(f0[i] - nearest_f0) > threshold_hz:
                f0[i] = 0
    return f0


def remove_large_diff_neighbors(f0, threshold_hz=50):
    f0 = f0.copy()
    length = len(f0)

    for i in range(1, length - 1):
        if f0[i] == 0:
            continue
        neighbors = []
        if f0[i - 1] > 0:
            neighbors.append(f0[i - 1])
        if f0[i + 1] > 0:
            neighbors.append(f0[i + 1])

        for neighbor_f0 in neighbors:
            if abs(f0[i] - neighbor_f0) >= threshold_hz:
                f0[i] = 0
                break
    return f0


def normalize_f0_mean(f0_array):
    valid_idx = f0_array != 0
    valid_f0 = f0_array[valid_idx]
    if len(valid_f0) == 0:
        return f0_array, 0.0, 0.0

    mean_f0 = float(np.mean(valid_f0))
    std_f0 = float(np.std(valid_f0))

    out = f0_array.copy()
    out[valid_idx] = valid_f0 - mean_f0
    return out, mean_f0, std_f0


# -----------------------------
# Optional plotting
# -----------------------------
def plot_f0s(t, f0_smoothed, f0_clean_norm, f0_raw_norm, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(t, f0_smoothed, label="smoothed", linestyle="--")
    plt.plot(t, f0_clean_norm, label="clean")
    plt.plot(t, f0_raw_norm, label="WORLD raw - mean")
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------
# Dataset traversal
# -----------------------------
def find_wavs(root_dir: str):
    root_dir = os.path.abspath(root_dir)
    pattern = os.path.join(root_dir, "**", "*.wav")
    return sorted(glob.glob(pattern, recursive=True))


def process_one_wav(
    wav_path: str,
    sr: int,
    method: str,
    frame_period: float,
    savgol_window: int,
    savgol_poly: int,
    interp_max_gap: int,
    thr_neighbor: float,
    thr_isolated: float,
    save_plots: bool,
    plot_dir: str | None,
):
    # load
    y, _sr = librosa.load(wav_path, sr=sr)  # force resample
    t, f0 = extract_f0_world(y, sr, method=method, frame_period=frame_period)

    # clean pipeline (same as your logic)
    f0_clean = remove_outliers_iqr(f0)
    f0_clean = remove_large_diff_neighbors(f0_clean, threshold_hz=thr_neighbor)
    f0_clean = remove_isolated_f0_outliers(f0_clean, threshold_hz=thr_isolated)

    f0_clean_nonzero = f0_clean[f0_clean > 0]
    if len(f0_clean_nonzero) > 0:
        per_utt_mean = float(np.mean(f0_clean_nonzero))
        per_utt_std = float(np.std(f0_clean_nonzero))
    else:
        per_utt_mean, per_utt_std = 0.0, 0.0

    f0_interp = interpolate_short_zeros(f0_clean, max_gap=interp_max_gap)
    f0_clean_norm, mean_f0, std_f0 = normalize_f0_mean(f0_interp)

    # for reference plotting/feature (raw normalized with same mean)
    f0_raw_norm = f0 - mean_f0

    # smooth (note: your original code smooths the mean-removed signal including zeros;
    # we keep that, then set uv to nan)
    f0_smoothed = savgol_smooth(f0_clean_norm, window_size=savgol_window, polyorder=savgol_poly)

    # uv -> nan, vuv
    f0_smoothed_out = f0_smoothed.copy()
    f0_clean_norm_out = f0_clean_norm.copy()

    f0_smoothed_out[f0_clean_norm_out == 0] = np.nan
    f0_clean_norm_out[f0_clean_norm_out == 0] = np.nan
    vuv = np.where(np.isnan(f0_clean_norm_out), 0, 1).astype(np.uint8)

    # save next to wav
    stem = str(Path(wav_path).with_suffix(""))
    np.save(stem + "_nor_f0_clean.npy", f0_clean_norm_out)
    np.save(stem + "_smoothed_f0_clean.npy", f0_smoothed_out)
    np.save(stem + "_vuv_clean.npy", vuv)

    # optional plot
    if save_plots:
        assert plot_dir is not None
        os.makedirs(plot_dir, exist_ok=True)
        base = Path(wav_path).stem
        save_path = os.path.join(plot_dir, f"{base}_f0_comparison.png")
        plot_f0s(t, f0_smoothed_out, f0_clean_norm_out, f0_raw_norm, f"F0: {base}", save_path)

    return per_utt_mean, per_utt_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/jyp/Maestro_EVC/data")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--method", type=str, default="dio", choices=["dio", "harvest"])
    parser.add_argument("--frame_period", type=float, default=10.0)

    parser.add_argument("--savgol_window", type=int, default=25)  # must be odd
    parser.add_argument("--savgol_poly", type=int, default=2)
    parser.add_argument("--interp_max_gap", type=int, default=30)

    parser.add_argument("--thr_neighbor", type=float, default=50.0)
    parser.add_argument("--thr_isolated", type=float, default=50.0)

    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--plot_dir", type=str, default=None)

    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    wavs = find_wavs(args.root)
    if args.limit is not None:
        wavs = wavs[: args.limit]

    print(f"[INFO] root={os.path.abspath(args.root)}")
    print(f"[INFO] found {len(wavs)} wav files")

    if args.save_plots and args.plot_dir is None:
        args.plot_dir = os.path.join(os.path.abspath(args.root), "_f0_plots")

    means, stds = [], []

    for i, wav_path in enumerate(tqdm(wavs, desc="Extracting F0"), 1):
        wav_path = os.path.abspath(wav_path)

        stem = str(Path(wav_path).with_suffix(""))
        out1 = stem + "_nor_f0_clean.npy"
        out2 = stem + "_smoothed_f0_clean.npy"
        out3 = stem + "_vuv_clean.npy"

        if args.skip_if_exists and os.path.exists(out1) and os.path.exists(out2) and os.path.exists(out3):
            continue

        try:
            m, s = process_one_wav(
                wav_path=wav_path,
                sr=args.sr,
                method=args.method,
                frame_period=args.frame_period,
                savgol_window=args.savgol_window,
                savgol_poly=args.savgol_poly,
                interp_max_gap=args.interp_max_gap,
                thr_neighbor=args.thr_neighbor,
                thr_isolated=args.thr_isolated,
                save_plots=args.save_plots,
                plot_dir=args.plot_dir,
            )
            means.append(m)
            stds.append(s)
        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")

    if len(means) > 0:
        global_mean = float(np.mean(means))
        global_std = float(np.mean(stds)) if len(stds) > 0 else 0.0
        print(f"\n[Global F0 Mean]: {global_mean:.4f}")
        print(f"[Global F0 Std]:  {global_std:.4f}")
    else:
        print("\n[INFO] No valid F0 extracted (all-UV or errors).")


if __name__ == "__main__":
    main()
