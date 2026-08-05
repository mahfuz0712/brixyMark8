"""Compatibility shims for older training dependencies on newer torch/torchaudio."""

from __future__ import annotations

try:
    import torchaudio
except Exception:  # pragma: no cover - optional runtime dependency
    torchaudio = None
else:
    _backend_name = "soundfile"

    if not hasattr(torchaudio, "set_audio_backend"):
        def set_audio_backend(backend_name: str | None = None) -> None:
            global _backend_name
            if backend_name is not None:
                _backend_name = backend_name

        def get_audio_backend() -> str:
            return _backend_name

        torchaudio.set_audio_backend = set_audio_backend  # type: ignore[attr-defined]
        torchaudio.get_audio_backend = get_audio_backend  # type: ignore[attr-defined]

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
else:
    # PyTorch >=2.6 flipped torch.load's default from weights_only=False to
    # weights_only=True (security hardening). piper-sample-generator's
    # generate_samples.py calls torch.load(model_path) with no kwargs to load
    # the full pickled Piper checkpoint (not just a tensor state_dict), so the
    # new strict default rejects it. We don't control that call site (it lives
    # in the cloned piper-sample-generator repo), so patch the default back —
    # safe here since the .pt file is the official Piper voice release asset,
    # not user-supplied/untrusted data.
    if not getattr(torch.load, "_brixy_weights_only_patched", False):
        _original_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        _patched_torch_load._brixy_weights_only_patched = True  # type: ignore[attr-defined]
        torch.load = _patched_torch_load  # type: ignore[assignment]