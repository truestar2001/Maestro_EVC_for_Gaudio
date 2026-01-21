import os
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

# kmeans loader
import joblib


def find_wavs(root_dir: str):
    root_dir = os.path.abspath(root_dir)
    pattern = os.path.join(root_dir, "**", "*.wav")
    return sorted(glob.glob(pattern, recursive=True))


def load_audio_mono_16k(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)  # (C, T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # (1, T)
    return wav


def load_kmeans_model(km_path: str):
    """
    지원 포맷:
    - sklearn KMeans / MiniBatchKMeans (joblib dump)
    - dict with 'cluster_centers_' or 'centers'
    - (fallback) torch.load로 dict 로드
    """
    km_path = os.path.abspath(km_path)

    # 1) joblib 우선
    try:
        km = joblib.load(km_path)
        return km
    except Exception:
        pass

    # 2) torch.load fallback
    try:
        obj = torch.load(km_path, map_location="cpu")
        return obj
    except Exception as e:
        raise RuntimeError(f"Failed to load kmeans model from {km_path}: {e}")


def kmeans_predict(km, feats_np: np.ndarray) -> np.ndarray:
    """
    feats_np: (T, D) float32
    return: (T,) int labels
    """
    # sklearn KMeans has predict()
    if hasattr(km, "predict"):
        return km.predict(feats_np).astype(np.int64)

    # dict/namespace 형태: cluster_centers_ or centers
    centers = None
    if isinstance(km, dict):
        if "cluster_centers_" in km:
            centers = km["cluster_centers_"]
        elif "centers" in km:
            centers = km["centers"]
        elif "km_model" in km and hasattr(km["km_model"], "cluster_centers_"):
            centers = km["km_model"].cluster_centers_
    else:
        # object with cluster_centers_
        if hasattr(km, "cluster_centers_"):
            centers = km.cluster_centers_

    if centers is None:
        raise ValueError("Unsupported kmeans object format: cannot find centers or predict().")

    centers = np.asarray(centers, dtype=np.float32)  # (K, D)
    # L2 distance argmin (T,K) 계산은 메모리 커질 수 있으니 chunk 처리
    T = feats_np.shape[0]
    K = centers.shape[0]
    labels = np.empty((T,), dtype=np.int64)

    chunk = 5000
    for i in range(0, T, chunk):
        x = feats_np[i:i+chunk]  # (c, D)
        # (c, K) = ||x||^2 - 2xC^T + ||C||^2
        x2 = np.sum(x * x, axis=1, keepdims=True)          # (c,1)
        c2 = np.sum(centers * centers, axis=1)[None, :]    # (1,K)
        dist = x2 - 2.0 * (x @ centers.T) + c2             # (c,K)
        labels[i:i+chunk] = np.argmin(dist, axis=1)

    return labels


@torch.inference_mode()
def extract_hubert_layer_features(hubert_model, wav_16k: torch.Tensor, layer: int, device: str) -> np.ndarray:
    """
    torchaudio HuBERT:
    hubert_model.extract_features(waveform) -> (features, lengths) or list of layer feats depending on version.

    여기서는 "모든 layer hidden states"를 받아서 지정 layer만 사용하도록 처리.
    layer: 1..N (예: L9이면 9)
    return: (T, D) float32 on CPU
    """
    wav_16k = wav_16k.to(device=device, dtype=torch.float32)

    # torchaudio models typically expect (B, T)
    if wav_16k.dim() == 2 and wav_16k.size(0) == 1:
        wav_bt = wav_16k  # (1, T)
    else:
        wav_bt = wav_16k.view(1, -1)

    # torchaudio hubert models:
    # - some return (features, lengths)
    # - some return a list of layer features
    out = hubert_model.extract_features(wav_bt)

    # normalize outputs
    # Case A: out is tuple (layer_features, lengths) where layer_features is list[Tensor]
    if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[0], (list, tuple)):
        layer_feats = out[0]  # list of (B, T, D)
    # Case B: out itself is list of tensors
    elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
        layer_feats = out
    # Case C: out is tuple (Tensor, lengths) only final feat
    elif isinstance(out, (tuple, list)) and len(out) == 2 and torch.is_tensor(out[0]):
        # fallback: final only
        final = out[0]
        feat = final[0].detach().cpu().numpy().astype(np.float32)
        return feat
    else:
        raise RuntimeError(f"Unexpected HuBERT extract_features output type: {type(out)}")

    # layer index
    idx = int(layer) - 1
    if idx < 0 or idx >= len(layer_feats):
        raise ValueError(f"Requested layer={layer} but model provides {len(layer_feats)} layers.")

    feat = layer_feats[idx]  # (B, T, D)
    feat = feat[0].detach().cpu().numpy().astype(np.float32)  # (T, D)
    return feat


def save_codes_txt(out_path: str, codes: np.ndarray):
    # space-separated integers
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, codes.tolist())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/jyp/Maestro_EVC/data")
    parser.add_argument("--km_path", type=str, default="/home/jyp/Maestro_EVC/hubert/hubert_base_ls960_L9_km500.bin",
                        help="k-means model path (e.g., /home/jyp/Maestro_EVC/hubert/hubert_base_ls960_L9_km500.bin)")
    parser.add_argument("--layer", type=int, default=9, help="HuBERT layer to use (1-based). L9 -> 9")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    wavs = find_wavs(args.root)
    if args.limit is not None:
        wavs = wavs[: args.limit]

    print(f"[INFO] root={os.path.abspath(args.root)}")
    print(f"[INFO] found={len(wavs)} wav files")
    print(f"[INFO] device={args.device}")
    print(f"[INFO] layer=L{args.layer}")
    print(f"[INFO] km_path={os.path.abspath(args.km_path)}")

    # ---- HuBERT base (official torchaudio pipeline) ----
    bundle = torchaudio.pipelines.HUBERT_BASE
    hubert_model = bundle.get_model().to(args.device)
    hubert_model.eval()

    # ---- KMeans ----
    km = load_kmeans_model(args.km_path)

    for i, wav_path in enumerate(wavs, 1):
        wav_path = os.path.abspath(wav_path)
        stem = str(Path(wav_path).with_suffix(""))
        out_txt = stem + "_hubert.txt"

        if args.skip_if_exists and os.path.exists(out_txt):
            continue

        try:
            wav_16k = load_audio_mono_16k(wav_path, target_sr=16000)  # (1,T) CPU
            feats = extract_hubert_layer_features(hubert_model, wav_16k, layer=args.layer, device=args.device)  # (T,D)
            codes = kmeans_predict(km, feats)  # (T,)
            save_codes_txt(out_txt, codes)
            print(f"[{i}/{len(wavs)}] Saved: {out_txt} (len={len(codes)})")
        except Exception as e:
            print(f"[{i}/{len(wavs)}] ERROR on {wav_path}: {e}")


if __name__ == "__main__":
    main()
