import os
import glob
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm


def find_wavs(root_dir: str):
    root_dir = os.path.abspath(root_dir)
    pattern = os.path.join(root_dir, "**", "*.wav")
    return sorted(glob.glob(pattern, recursive=True))


def compute_energy_melspectrogram(
    signal: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 10.0,
    n_mels: int = 80,
):
    """
    동일 로직 유지:
      mel_spec = librosa.feature.melspectrogram(power=2.0, center=False, n_fft=frame, hop=frame)
      energy = sqrt(mean(mel_spec, axis=0) + 1e-9)
    """
    frame_size = int(sample_rate * frame_size_ms / 1000.0)
    hop_length = frame_size

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=frame_size,
        hop_length=hop_length,
        win_length=frame_size,
        n_mels=n_mels,
        power=2.0,
        center=False,
    )
    energy = np.sqrt(np.mean(mel_spec, axis=0) + 1e-9)
    return energy


def savgol_smooth(x: np.ndarray, window_size: int = 25, polyorder: int = 2):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if window_size > len(x):
        # 너무 짧은 신호는 window를 줄여서 처리 (실패 대신 자동 보정)
        window_size = max(3, (len(x) // 2) * 2 + 1)  # 가장 가까운 홀수
        if window_size > len(x):
            return x.copy()
    return savgol_filter(x, window_length=window_size, polyorder=polyorder)


def plot_energy(energy: np.ndarray, wav_path: str, output_path: str):
    plt.figure(figsize=(10, 4))
    plt.plot(energy, linewidth=3)
    plt.xlabel("Frames")
    plt.ylabel("Energy")
    plt.title(Path(wav_path).stem)
    plt.tight_layout()
    plt.savefig(output_path, transparent=True)
    plt.close()


def load_audio_float(wav_path: str, target_sr: int | None = None):
    """
    기존은 sf.read(int16) 후 /32768.0 했는데,
    여기서는 sf.read(float32)로 안전하게 읽고,
    필요 시 librosa.resample로 리샘플.
    """
    x, sr = sf.read(wav_path, dtype="float32", always_2d=False)

    # stereo -> mono
    if x.ndim == 2:
        x = np.mean(x, axis=1)

    if target_sr is not None and sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return x, sr


def process_one_wav(
    wav_path: str,
    frame_size_ms: float,
    n_mels: int,
    energy_floor: float,
    savgol_window: int,
    savgol_poly: int,
    target_sr: int | None,
    save_plots: bool,
    plot_dir: str | None,
):
    x, sr = load_audio_float(wav_path, target_sr=target_sr)

    energy = compute_energy_melspectrogram(
        signal=x,
        sample_rate=sr,
        frame_size_ms=frame_size_ms,
        n_mels=n_mels,
    )

    # 기존 로직 유지: 작은 값 0 처리
    energy = energy.copy()
    energy[energy < energy_floor] = 0.0

    smoothed = savgol_smooth(energy, window_size=savgol_window, polyorder=savgol_poly)

    # 저장 (wav 옆)
    stem = str(Path(wav_path).with_suffix(""))
    np.save(stem + "_mel_energy_nonor.npy", energy)
    np.save(stem + "_smoothed_energy_nonor.npy", smoothed)

    # plot은 옵션
    if save_plots:
        assert plot_dir is not None
        os.makedirs(plot_dir, exist_ok=True)
        out_png = os.path.join(plot_dir, f"{Path(wav_path).stem}_energy.png")
        plot_energy(smoothed, wav_path, out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/jyp/Maestro_EVC/data")

    parser.add_argument("--frame_ms", type=float, default=10.0)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--energy_floor", type=float, default=0.01)

    parser.add_argument("--savgol_window", type=int, default=25)  # 홀수 권장
    parser.add_argument("--savgol_poly", type=int, default=2)

    parser.add_argument("--target_sr", type=int, default=None, help="If set, resample audio to this SR first.")
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
        args.plot_dir = os.path.join(os.path.abspath(args.root), "_energy_plots")

    for wav_path in tqdm(wavs, desc="Extracting Energy"):
        wav_path = os.path.abspath(wav_path)
        stem = str(Path(wav_path).with_suffix(""))
        out1 = stem + "_mel_energy_nonor.npy"
        out2 = stem + "_smoothed_energy_nonor.npy"

        if args.skip_if_exists and os.path.exists(out1) and os.path.exists(out2):
            continue

        try:
            process_one_wav(
                wav_path=wav_path,
                frame_size_ms=args.frame_ms,
                n_mels=args.n_mels,
                energy_floor=args.energy_floor,
                savgol_window=args.savgol_window,
                savgol_poly=args.savgol_poly,
                target_sr=args.target_sr,
                save_plots=args.save_plots,
                plot_dir=args.plot_dir,
            )
        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")


if __name__ == "__main__":
    main()
