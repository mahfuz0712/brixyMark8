from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tarfile
import json
from pathlib import Path

# --------------------------------------------------------------------------
# Paths / defaults (same layout as the Colab notebook)
# --------------------------------------------------------------------------
OWW_DIR = Path("openWakeWord")               # git clone of dscripka/openWakeWord
PIPER_DIR = Path("piper-sample-generator")   # dscripka fork, used by --generate_clips
OUTPUT_DIR = Path("my_custom_model")

RIR_DIR = Path("mit_rirs")
BG_AUDIOSET_DIR = Path("audioset_16k")
BG_FMA_DIR = Path("fma")
AUDIOSET_TAR_DIR = Path("audioset")

VALIDATION_FEATURES = Path("validation_set_features.npy")
ACAV_FEATURES = Path("openwakeword_features_ACAV100M_2000_hrs_16bit.npy")

CONFIG_FILE = Path("my_model.yaml")

VOICE_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/"
    "v2.0.0/en_US-libritts_r-medium.pt"
)
# Every Piper voice checkpoint ships with a companion .json (phoneme map +
# speaker table) that must sit right next to the .pt file with a matching
# name — generate_samples.py loads "<model_path>.json" unconditionally.
VOICE_CONFIG_URL = VOICE_URL + ".json"
# openwakeword.utils.AudioFeatures (default inference_framework="onnx") loads
# these two pretrained models from <oww_repo>/openwakeword/resources/models/.
# The pip-installed `openwakeword` package bundles them, but the separately
# git-cloned training repo (OWW_DIR, used via PYTHONPATH in run_training())
# does NOT — they're release binary assets, not checked into git. Same
# situation as the Piper voice model above.
OWW_RESOURCE_URLS = {
    "melspectrogram.onnx": (
        "https://github.com/dscripka/openWakeWord/releases/download/"
        "v0.5.1/melspectrogram.onnx"
    ),
    "embedding_model.onnx": (
        "https://github.com/dscripka/openWakeWord/releases/download/"
        "v0.5.1/embedding_model.onnx"
    ),
}
def log(msg: str) -> None:
    print(f"[train.py] {msg}", flush=True)


# --------------------------------------------------------------------------
# Stage 0 — piper-sample-generator fork + voice model
# --------------------------------------------------------------------------
def ensure_piper() -> None:
    """Clone the dscripka fork (the one v0.6.0's config imports from) and
    fetch the Piper voice model. v0.6.0 does NOT auto-download the voice."""
    if not PIPER_DIR.exists():
        log("Cloning dscripka/piper-sample-generator ...")
        subprocess.run(
            ["git", "clone", "https://github.com/dscripka/piper-sample-generator", str(PIPER_DIR)],
            check=True,
        )
    # NOTE: the file is saved under this legacy filename ("en-us-libritts-high.pt"),
    # NOT the name in VOICE_URL ("en_US-libritts_r-medium.pt") — generate_samples.py's
    # `model_path` default argument still hardcodes the old name, and openwakeword's
    # train.py never overrides it when calling generate_samples(). Same trick the
    # official Colab notebook uses (wget -O models/en-us-libritts-high.pt <v2 url>).
    voice = PIPER_DIR / "models" / "en-us-libritts-high.pt"
    if not voice.exists():
        log("Downloading Piper voice model ...")
        voice.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(VOICE_URL, voice)
        log(f"Voice model ready: {voice}")

    # NOTE: unlike the .pt weights, the companion "en-us-libritts-high.pt.json"
    # config (phoneme map + speaker table) is a small file checked directly
    # into the piper-sample-generator git repo under this exact filename —
    # verified by cloning the repo and inspecting it directly. No download or
    # rename needed here; `git clone` above already fetched it.
# --------------------------------------------------------------------------
# Stage 1 — datasets
# --------------------------------------------------------------------------
def download_rirs() -> None:
    """MIT room impulse responses, via HTTP (huggingface_hub) — no git-lfs."""
    if RIR_DIR.exists() and any(RIR_DIR.glob("*.wav")):
        log(f"RIRs already in {RIR_DIR}, skipping")
        return
    from huggingface_hub import snapshot_download
    import librosa
    import soundfile

    log("Downloading MIT RIR dataset ...")
    src = snapshot_download(
        repo_id="davidscripka/MIT_environmental_impulse_responses",
        repo_type="dataset",
        allow_patterns="16khz/*",
    )
    RIR_DIR.mkdir(exist_ok=True)
    for wav in sorted(Path(src, "16khz").glob("*.wav")):
        y, _ = librosa.load(wav, sr=16000, mono=True)
        soundfile.write(RIR_DIR / wav.name, y, 16000)
    log(f"RIRs ready in {RIR_DIR}")


def download_audioset_background(max_clips: int = 0) -> None:
    """One AudioSet tar (bal_train09) -> 16 kHz wavs. max_clips=0 = all."""
    if BG_AUDIOSET_DIR.exists() and any(BG_AUDIOSET_DIR.glob("*.wav")):
        log(f"AudioSet background already in {BG_AUDIOSET_DIR}, skipping")
        return

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError
    import librosa
    import soundfile
    from tqdm import tqdm

    BG_AUDIOSET_DIR.mkdir(exist_ok=True)
    try:
        tar_path = hf_hub_download(
            repo_id="agkphysics/AudioSet", repo_type="dataset", filename="data/bal_train09.tar"
        )
    except RemoteEntryNotFoundError as exc:
        log(f"AudioSet tarball unavailable ({exc}); skipping AudioSet background download")
        return

    AUDIOSET_TAR_DIR.mkdir(exist_ok=True)
    log("Extracting bal_train09.tar ...")
    with tarfile.open(tar_path) as t:
        t.extractall(AUDIOSET_TAR_DIR)

    flacs = sorted(Path(AUDIOSET_TAR_DIR, "audio").glob("**/*.flac"))
    if max_clips > 0:
        flacs = flacs[:max_clips]
    log(f"Converting {len(flacs)} FLAC files to 16 kHz WAV ...")
    for flac in tqdm(flacs):
        y, _ = librosa.load(flac, sr=16000, mono=True)
        soundfile.write(BG_AUDIOSET_DIR / (flac.stem + ".wav"), y, 16000)
    log(f"AudioSet background ready in {BG_AUDIOSET_DIR}")


def download_fma_background(hours: int = 1) -> None:
    """Stream the FMA small split (mp3 -> 16 kHz wav). Default 1 hour."""
    if BG_FMA_DIR.exists() and any(BG_FMA_DIR.glob("*.wav")):
        log(f"FMA background already in {BG_FMA_DIR}, skipping")
        return
    import soundfile
    from tqdm import tqdm

    BG_FMA_DIR.mkdir(exist_ok=True)
    log(f"Streaming FMA small split ({hours}h of 30s clips) ...")

    try:
        from datasets import Audio, load_dataset
    except RuntimeError:
        log("datasets package does not support the legacy FMA script loader; skipping FMA background download")
        return

    try:
        fma = iter(
            load_dataset("rudraml/fma", name="small", split="train", streaming=True).cast_column(
                "audio", Audio(sampling_rate=16000)
            )
        )
    except Exception as exc:  # pragma: no cover - runtime compatibility fallback
        log(f"FMA dataset unavailable ({exc}); skipping FMA background download")
        return

    for _ in tqdm(range(hours * 3600 // 30)):
        try:
            row = next(fma)
        except StopIteration:
            break
        name = row["audio"]["path"].split("/")[-1].replace(".mp3", ".wav")
        soundfile.write(BG_FMA_DIR / name, row["audio"]["array"], 16000)
    log(f"FMA background ready in {BG_FMA_DIR}")


def download_features() -> None:
    """Pre-computed openwakeword features: ACAV100M train + validation set.
    Downloaded via plain HTTP (like the Colab notebook's !wget), bypassing
    huggingface_hub entirely so a stale/bad cached HF token can't break this."""
    import urllib.request

    base_url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main"

    def try_download(filename: str, target: Path) -> None:
        if target.exists():
            log(f"{filename} already present, skipping")
            return
        url = f"{base_url}/{filename}"
        try:
            log(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, target)
            log(f"{filename} ready")
        except Exception as exc:
            log(f"Skipping unavailable feature file {filename}: {exc}")

    try_download("validation_set_features.npy", VALIDATION_FEATURES)
    try_download("openwakeword_features_ACAV100M_2000_hrs_16bit.npy", ACAV_FEATURES)
# --------------------------------------------------------------------------
# Stage 2 — write the v0.6.0 training config
# --------------------------------------------------------------------------
def write_config(target_phrase: str, n_samples: int, steps: int, max_negative_weight: int) -> Path:
    model_name = target_phrase.replace(" ", "_")
    config = {
        # v0.6.0 schema — the notebook's old keys (target_accuracy/target_recall)
        # are gone; new keys: piper_sample_generator_path, tts_batch_size, ...
        "model_name": model_name,
        "target_phrase": [target_phrase],
        "custom_negative_phrases": [],
        "n_samples": n_samples,
        "n_samples_val": max(500, n_samples // 10),
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": str(PIPER_DIR),
        "output_dir": str(OUTPUT_DIR),
        "rir_paths": [str(RIR_DIR)],
        "background_paths": [str(BG_AUDIOSET_DIR), str(BG_FMA_DIR)],
        "background_paths_duplication_rate": [1, 1],
        "false_positive_validation_data_path": str(VALIDATION_FEATURES),
        "augmentation_rounds": 1,
        "feature_data_files": {"ACAV100M_sample": str(ACAV_FEATURES)},
        "batch_n_per_class": {
            "ACAV100M_sample": 1024,
            "adversarial_negative": 50,
            "positive": 50,
        },
        "model_type": "dnn",
        "layer_size": 32,
        "steps": steps,
        "max_negative_weight": max_negative_weight,
        "target_false_positives_per_hour": 0.2,
    }
    # JSON is a valid YAML subset, so this remains compatible with YAML readers.
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log(f"Config written to {CONFIG_FILE}")
    return CONFIG_FILE


# --------------------------------------------------------------------------
# Stage 3 — run the three training stages
# --------------------------------------------------------------------------
def run_training(config_file: Path) -> None:
    stages = [
        ("generate synthetic clips", "--generate_clips"),
        ("augment clips + compute features", "--augment_clips"),
        ("train model -> .onnx", "--train_model"),
    ]
    repo_root = Path(__file__).resolve().parents[2]
    oww_repo = repo_root / "openWakeWord"
    if not (oww_repo / "openwakeword" / "train.py").exists():
        raise SystemExit(f"Missing openWakeWord training repo at {oww_repo}")

    ensure_oww_resources(oww_repo)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(oww_repo) + os.pathsep + env.get("PYTHONPATH", "")

    for label, flag in stages:
        log(f"Running stage: {label}")
        subprocess.run(
            [sys.executable, "-m", "openwakeword.train", "--training_config", str(config_file), flag],
            check=True,
            cwd=repo_root,
            env=env,
        )
    log(f"Training done — ONNX model at {OUTPUT_DIR}")


def ensure_oww_resources(oww_repo: Path) -> None:
    """Download the two pretrained ONNX models AudioFeatures needs into the
    cloned repo's resources/models/ folder — see OWW_RESOURCE_URLS comment."""
    models_dir = oww_repo / "openwakeword" / "resources" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request

    for filename, url in OWW_RESOURCE_URLS.items():
        target = models_dir / filename
        if target.exists():
            log(f"{filename} already in {models_dir}, skipping")
            continue
        log(f"Downloading openWakeWord resource model {filename} ...")
        urllib.request.urlretrieve(url, target)
        log(f"{filename} ready: {target}")
# --------------------------------------------------------------------------
# Stage 4 — ONNX -> TFLite (onnx2tf, Python 3.11-safe)
# --------------------------------------------------------------------------
def convert_to_tflite() -> None:
    model_name = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))["model_name"]
    onnx_path = OUTPUT_DIR / f"{model_name}.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"Missing trained model: {onnx_path} — run --train first")

    log("Converting ONNX -> TFLite with onnx2tf ...")
    # -kat keeps the Flatten op un-fused (same flag as the notebook).
    # If a future export drops that op name, drop the -kat flag and re-run.
    subprocess.run(
        ["onnx2tf", "-i", str(onnx_path), "-o", str(OUTPUT_DIR), "-kat", "onnx____Flatten_0"],
        check=True,
    )
    float32 = OUTPUT_DIR / f"{model_name}_float32.tflite"
    target = OUTPUT_DIR / f"{model_name}.tflite"
    if float32.exists():
        float32.rename(target)
    log(f"TFLite model ready: {target}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Train a custom openwakeword wake word model (Brixy).")
    p.add_argument("--all", action="store_true", help="run every stage")
    p.add_argument("--data", action="store_true", help="stage 1: download datasets + piper voice")
    p.add_argument("--config", action="store_true", help="stage 2: write my_model.yaml")
    p.add_argument("--train", action="store_true", help="stage 3: generate/augment/train")
    p.add_argument("--tflite", action="store_true", help="stage 4: onnx -> tflite")
    p.add_argument("--target-phrase", default="brix_eee", help="wake word (default: brix_eee)")
    p.add_argument("--n-samples", type=int, default=1000, help="positive samples (notebook default; 20k+ recommended)")
    p.add_argument("--steps", type=int, default=10000, help="training steps (default: 10000)")
    p.add_argument("--max-negative-weight", type=int, default=1500, help="false-activation penalty (default: 1500)")
    args = p.parse_args()

    if args.all:
        args.data = args.config = args.train = args.tflite = True

    if args.data:
        ensure_piper()
        download_rirs()
        download_audioset_background()
        download_fma_background()
        download_features()
    if args.config:
        write_config(args.target_phrase, args.n_samples, args.steps, args.max_negative_weight)
    if args.train:
        if not CONFIG_FILE.exists():
            write_config(args.target_phrase, args.n_samples, args.steps, args.max_negative_weight)
        run_training(CONFIG_FILE)
    if args.tflite:
        convert_to_tflite()


if __name__ == "__main__":
    main()