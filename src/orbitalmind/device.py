"""
Torch device selection.

Every model used to hardcode `device: str = "cpu"`, so the pipeline could not
use a GPU even when one was present -- while the proposal deck promised a run
on a free Colab T4. Device is now resolved once, here, and every model takes
`device=None` to mean "decide for me".

Order of precedence:
  1. an explicit device string passed by the caller
  2. the ORBITALMIND_DEVICE environment variable
  3. CUDA, then Apple MPS, then CPU

A realistic note on what to expect. These models are small: LSTM hidden 64,
TFT hidden 16, batch size 16. Their cost is dominated by two things that a GPU
does not fix -- the 96-step autoregressive rollout in predict_lstm and
predict_tcn_lstm, and the rk4 solve in the Neural ODE. Both are sequential and
latency-bound rather than throughput-bound. Expect roughly 2-3x end to end,
not 20x.

The larger win on a multi-core machine is that satellites are independent of
one another, so the loop in run_pipeline parallelises cleanly. That is not
implemented yet.
"""
import os

import torch

ENV_VAR = "ORBITALMIND_DEVICE"


def resolve_device(device: str | None = None) -> torch.device:
    """
    Choose the torch device to run on.

    Args:
        device: explicit device string such as 'cuda', 'cpu' or 'mps'.
                None consults ORBITALMIND_DEVICE, then autodetects.
    Returns:
        A torch.device. Falls back to CPU whenever the requested accelerator
        is unavailable, so a script written for GPU still runs on a laptop.
    """
    requested = device or os.environ.get(ENV_VAR) or ""
    requested = requested.strip().lower()

    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        if requested == "mps" and not _mps_available():
            return torch.device("cpu")
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available() -> bool:
    """
    Report whether Apple Metal acceleration is usable.

    Returns:
        True if the MPS backend is both built and available.
    """
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def describe_device(dev: torch.device | None = None) -> str:
    """
    Produce a human-readable description of the active device.

    Args:
        dev: device to describe, or None to resolve one now.
    Returns:
        A string such as 'cuda (Tesla T4)' or 'cpu (16 threads)'.
    """
    dev = dev or resolve_device()
    if dev.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(dev.index or 0)})"
    if dev.type == "mps":
        return "mps (Apple Metal)"
    return f"cpu ({torch.get_num_threads()} threads)"
