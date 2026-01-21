import os
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


def find_wavs(root_dir: str):
    root_dir = os.path.abspath(root_dir)
    pattern = os.path.join(root_dir, "**", "*.wav")
    return sorted(glob.glob(pattern, recursive=True))


def load_wav_as_batch(path: str, target_sr: int | None = None) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # (C, T) on CPU
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono: (1, T)

    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # speechbrain encode_batch: 보통 (B, T) 가 안전
    wav = wav.squeeze(0).unsqueeze(0)  # (1, T)
    return wav


def get_any_param_device(module: torch.nn.Module) -> torch.device:
    for p in module.parameters():
        return p.device
    return torch.device("cpu")


@torch.inference_mode()
def extract_spk_emb(classifier: EncoderClassifier, wav_path: str, device: str, target_sr: int | None):
    sig = load_wav_as_batch(wav_path, target_sr=target_sr)

    # ✅ 입력도 같은 device로
    sig = sig.to(device=device, dtype=torch.float32)

    # ✅ 진단: 모델/입력 device 확인
    model_dev = get_any_param_device(classifier.mods)
    if sig.device != model_dev:
        print(f"[DIAG] device mismatch BEFORE forward: sig={sig.device}, model={model_dev}")

    # SpeechBrain은 wav_lens 주면 더 안정적(패딩/배치 처리)
    wav_lens = torch.ones(sig.size(0), device=sig.device)

    emb = classifier.encode_batch(sig, wav_lens=wav_lens)

    # numpy 저장용
    emb = emb.detach().cpu().numpy()
    emb = np.squeeze(emb)
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/jyp/Maestro_EVC/data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_sr", type=int, default=None)
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    wavs = find_wavs(args.root)
    if args.limit is not None:
        wavs = wavs[: args.limit]

    print(f"[INFO] root={os.path.abspath(args.root)}")
    print(f"[INFO] device={args.device}, found={len(wavs)} wavs")

    # ✅ 핵심: run_opts로 device 지정해서 로드
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": args.device},
    )
    classifier.eval()

    # ✅ 진단 출력
    print(f"[INFO] classifier device (mods param) = {get_any_param_device(classifier.mods)}")

    for i, wav_path in enumerate(wavs, 1):
        wav_path = os.path.abspath(wav_path)
        out_path = str(Path(wav_path).with_suffix("")) + "_speaker_embedding.npy"

        if args.skip_if_exists and os.path.exists(out_path):
            print(f"[{i}/{len(wavs)}] SKIP exists: {out_path}")
            continue

        try:
            emb = extract_spk_emb(classifier, wav_path, args.device, args.target_sr)
            np.save(out_path, emb)
            print(f"[{i}/{len(wavs)}] Saved: {out_path} (shape={emb.shape})")
        except Exception as e:
            # ✅ 에러 시 핵심 진단 추가 출력
            print(f"[{i}/{len(wavs)}] ERROR on {wav_path}: {e}")
            try:
                print(f"      [DIAG] mods param device = {get_any_param_device(classifier.mods)}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
