"""
Tests for device selection.

The models previously hardcoded device="cpu", so a GPU was unusable even where
one existed. These tests pin the contract: autodetect by default, honour an
explicit request, and always degrade to CPU rather than raising when the
requested accelerator is missing.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.device import resolve_device, describe_device, ENV_VAR


def test_returns_a_torch_device():
    assert isinstance(resolve_device(), torch.device)


def test_explicit_cpu_is_honoured():
    assert resolve_device("cpu").type == "cpu"


def test_missing_accelerator_falls_back_instead_of_raising():
    """A script written for GPU must still run on a teammate's laptop."""
    dev = resolve_device("cuda")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev.type == expected


def test_environment_variable_is_consulted(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "cpu")
    assert resolve_device().type == "cpu"


def test_explicit_argument_beats_environment(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "cuda")
    assert resolve_device("cpu").type == "cpu"


def test_blank_environment_variable_is_ignored(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "   ")
    assert isinstance(resolve_device(), torch.device)


def test_autodetect_matches_availability():
    dev = resolve_device()
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type in ("mps", "cpu")


def test_describe_device_is_readable():
    text = describe_device(torch.device("cpu"))
    assert "cpu" in text and "threads" in text


# ---------------------------------------------------------------- wiring tests
# plan/day-02.md requires device.py to be "either fully wired with tests, or
# gone. No half-connected state." It sat unwired for a whole session: written,
# tested in isolation, and imported by nothing, while every trainer hardcoded
# device="cpu" and the deck promised a run on a free Colab T4. These tests exist
# so that state cannot return silently.

import inspect  # noqa: E402
import pathlib  # noqa: E402

MODEL_FILES = ["lstm.py", "tcn_lstm.py", "tft.py", "neural_ode.py"]
MODELS_DIR = pathlib.Path(__file__).parent.parent / "src" / "orbitalmind" / "models"


def test_no_model_hardcodes_cpu():
    """A hardcoded cpu default makes the deck's GPU claim unreachable."""
    offenders = [f for f in MODEL_FILES
                 if 'device: str = "cpu"' in (MODELS_DIR / f).read_text()]
    assert not offenders, f"hardcoded cpu default still present in {offenders}"


def test_every_model_resolves_through_device_module():
    """
    Models must call resolve_device(), not torch.device() directly.

    Calling torch.device(device) directly reintroduces the bug: it cannot fall
    back when the requested accelerator is missing, and it ignores
    ORBITALMIND_DEVICE.
    """
    for f in MODEL_FILES:
        text = (MODELS_DIR / f).read_text()
        assert "dev = torch.device(device)" not in text, \
            f"{f} bypasses resolve_device()"
        assert "resolve_device" in text, f"{f} never imports resolve_device"


def test_trainers_accept_device_none():
    """`device=None` must mean 'decide for me' on every public entry point."""
    from orbitalmind.models import lstm, tcn_lstm, tft, neural_ode

    entry_points = [
        lstm.train_lstm, lstm.predict_lstm,
        tcn_lstm.train_tcn_lstm, tcn_lstm.predict_tcn_lstm,
        tft.train_tft, tft.predict_tft,
        neural_ode.train_neural_ode, neural_ode.predict_neural_ode,
    ]
    for fn in entry_points:
        sig = inspect.signature(fn)
        assert "device" in sig.parameters, f"{fn.__name__} has no device parameter"
        assert sig.parameters["device"].default is None, \
            f"{fn.__name__} defaults device to {sig.parameters['device'].default!r}, not None"
