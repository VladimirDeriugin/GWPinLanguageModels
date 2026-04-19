"""
Inference / prompting script (Tiny Shakespeare, char-level).
Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import pickle
import time
import torch
from codecarbon import OfflineEmissionsTracker

from model import GPT, GPTConfig

# ----------------------------
# Edit these
# ----------------------------
OUT_DIR = "out"
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")

PROMPT = "To be, or not to be"
MAX_NEW_TOKENS = 200

TEMPERATURES = [0.2, 0.5, 0.8, 1.0, 1.3]
TOP_K = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]

    meta = load_meta(data_dir)
    stoi = meta["stoi"]
    itos = meta["itos"]

    def encode(s: str):
        return [stoi.get(ch, stoi[" "]) for ch in s]

    def decode(tokens):
        return "".join([itos[t] for t in tokens])

    config = GPTConfig(**model_cfg)
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([encode(PROMPT)], dtype=torch.long, device=DEVICE)

    for temperature in TEMPERATURES:
        tracker = OfflineEmissionsTracker(
            country_iso_code="DNK",
            output_dir=OUT_DIR,
            output_file="inference_emissions.csv",
            log_level="INFO",
            measure_power_secs=1,
        )

        tracker.start()
        t0 = time.time()

        out = model.generate(
            idx,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_k=TOP_K
        )

        elapsed = time.time() - t0
        emissions = tracker.stop()

        print("\n" + "=" * 60)
        print(f"TEMPERATURE: {temperature}")
        print(f"TOP_K: {TOP_K}")
        print(f"Elapsed time: {elapsed:.4f} s")
        print(f"CO2 emissions per prompt: {emissions:.8f} kg CO2")
        print(f"CO2 emissions per generated token: {(emissions / MAX_NEW_TOKENS):.8f} kg CO2/token")
        print("-" * 60)
        print(decode(out[0].tolist()))
        print("=" * 60)


if __name__ == "__main__":
    main()