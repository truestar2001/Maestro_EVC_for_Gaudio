from speechbrain.inference.diarization import Speech_Emotion_Diarization
import os, glob
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def find_all_wavs(root_dir: str):
    root_dir = os.path.abspath(root_dir)
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))

@torch.inference_mode()
def extract_frame_embedding(classifier: Speech_Emotion_Diarization, wav_path: str, device: str):
    # load waveform (SpeechBrain internal loader)
    waveform = classifier.load_audio(wav_path)          # shape: (time,) or (time, ch) depending on loader
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)               # (1, time)
    # SpeechBrain encode_batch expects [batch, time] or [batch, time, channels] depending on model
    # In their own interface doc: [batch, time, channels] but most pipelines accept [batch, time]
    batch = waveform.to(device)
    wav_lens = torch.ones(batch.shape[0], device=device)

    emb = classifier.encode_batch(batch, wav_lens)      # expected: (B, T, 1024) for WavLM-large
    emb = emb.detach().cpu().numpy()
    return emb

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = "/home/jyp/Maestro_EVC/data"

    audio_list = find_all_wavs(root)
    print(f"[INFO] root = {root}")
    print(f"[INFO] #wav = {len(audio_list)}")
    print(f"[INFO] device = {device}")

    classifier = Speech_Emotion_Diarization.from_hparams(
        source="speechbrain/emotion-diarization-wavlm-large",
        run_opts={"device": device},
    )
    classifier.eval()

    for num, audio in enumerate(tqdm(audio_list, desc="Extracting"), 1):
        out_path = str(Path(audio).with_suffix("")) + "_emotion_diarization_embedding.npy"

        try:
            embedding = extract_frame_embedding(classifier, audio, device)
            with open(out_path, "wb") as f:
                np.save(f, embedding)
            # e.g., (1, T, 1024)
            # print(f"[{num}/{len(audio_list)}] Saved: {out_path} (shape={embedding.shape})")
        except Exception as e:
            print(f"[{num}/{len(audio_list)}] ERROR on {audio}: {e}")
