"""
Tests for the ORBITALMIND_DIRECT_HEADS multi-horizon head on LSTM and TCN-LSTM.

Two things need proving, and they pull in opposite directions.

With the flag OFF the modules must behave exactly as they did before the head
existed, because a controlled A/B is being run against the autoregressive
numbers. "Exactly" is asserted against reference classes pinned below, copied
verbatim from the pre-change source, so a drift in weights or in the forward
expression fails here rather than showing up as a quiet change in forecasts.

With the flag ON the rollout must be gone -- not merely reshaped. Shape alone
is a weak assertion: a loop that appends 96 values into an array also produces
96 values. So the feedback path is attacked directly, by four independent
routes: counting forward() invocations and inspecting every tensor the model
is handed; driving predict with a stub whose output changes on every call;
corrupting the model the instant its first forward returns, with a
single-step control proving that corruption is detectable; and comparing the
result against one hand-written forward pass.

Single-threaded and deliberately small: pipeline runs may be in flight.
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

torch.set_num_threads(1)

from orbitalmind.models import lstm as lstm_mod
from orbitalmind.models import tcn_lstm as tcn_mod
from orbitalmind.models.lstm import LSTMPredictor, predict_lstm, train_lstm
from orbitalmind.models.tcn_lstm import (
    TCNBlock,
    TCNLSTMPredictor,
    predict_tcn_lstm,
    train_tcn_lstm,
)
from orbitalmind.paths import models_dir

SEQ_LEN  = 96
PRED_LEN = 96

# Tag used for every checkpoint written here, so these tests never overwrite a
# real satellite's weights while a pipeline run is using models/saved.
TAG = "PYTEST-DIRECT-HEADS"


# --------------------------------------------------------------------------
# Reference implementations, copied verbatim from the pre-change modules.
# The flag-OFF path is asserted equal to these, parameter for parameter.
# --------------------------------------------------------------------------

class _ReferenceLSTM(nn.Module):
    """LSTMPredictor exactly as it stood before the direct head was added."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class _ReferenceTCNLSTM(nn.Module):
    """TCNLSTMPredictor exactly as it stood before the direct head was added."""

    def __init__(self, input_size=1, lstm_hidden=64):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_size, 32, kernel_size=3, dilation=1),
            TCNBlock(32, 64, kernel_size=3, dilation=2),
        )
        self.lstm = nn.LSTM(64, lstm_hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        h = x.permute(0, 2, 1)
        h = self.tcn(h)
        h = h.permute(0, 2, 1)
        out, _ = self.lstm(h)
        return self.fc(out[:, -1, :]).squeeze(-1)


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

@pytest.fixture
def flag_off(monkeypatch):
    """Guarantee the flag is unset, whatever the caller's environment holds."""
    monkeypatch.delenv("ORBITALMIND_DIRECT_HEADS", raising=False)


@pytest.fixture
def flag_on(monkeypatch):
    """Enable direct heads and shorten training to keep the suite cheap."""
    monkeypatch.setenv("ORBITALMIND_DIRECT_HEADS", "1")
    monkeypatch.setattr(lstm_mod, "EPOCHS", 2)
    monkeypatch.setattr(tcn_mod, "EPOCHS", 2)


def _signal(n: int = 250) -> np.ndarray:
    """A small deterministic series: enough for SEQ_LEN + PRED_LEN windows."""
    t = np.arange(n, dtype=np.float64)
    return (np.sin(2 * np.pi * t / 96) + 0.01 * t).astype(np.float32)


def _batch(seed: int = 7, batch: int = 3) -> torch.Tensor:
    """Fixed input batch of shape (batch, SEQ_LEN, 1)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, SEQ_LEN, 1, generator=g)


class _InputRecorder:
    """Captures every tensor a module's forward is handed, in order."""

    def __init__(self, model):
        self.inputs = []
        self._handle = model.register_forward_pre_hook(
            lambda _m, args: self.inputs.append(args[0].detach().clone())
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        return False

    @property
    def n_calls(self) -> int:
        return len(self.inputs)


# --------------------------------------------------------------------------
# The flag itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", [lstm_mod, tcn_mod], ids=["lstm", "tcn_lstm"])
def test_flag_defaults_off(module, flag_off):
    """An unset environment must mean the historical single-step head."""
    assert module._direct_heads_enabled() is False


@pytest.mark.parametrize("module", [lstm_mod, tcn_mod], ids=["lstm", "tcn_lstm"])
@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("YES", True),
     ("0", False), ("", False), ("false", False), ("no", False)],
)
def test_flag_parsing(module, monkeypatch, value, expected):
    """Only explicit opt-in values enable the head; anything else is off."""
    monkeypatch.setenv("ORBITALMIND_DIRECT_HEADS", value)
    assert module._direct_heads_enabled() is expected


# --------------------------------------------------------------------------
# Flag OFF: unchanged, asserted against the pinned pre-change reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cls,reference", [(LSTMPredictor, _ReferenceLSTM), (TCNLSTMPredictor, _ReferenceTCNLSTM)],
    ids=["lstm", "tcn_lstm"],
)
def test_flag_off_weights_and_forward_match_pre_change_code(cls, reference, flag_off):
    """
    Default construction must be bit-identical to the pre-change class.

    Both the parameter tensors under seed 42 -- proving the head change did not
    alter how many random draws the constructor consumes -- and the forward
    output on a fixed batch.
    """
    torch.manual_seed(42)
    new = cls()
    torch.manual_seed(42)
    old = reference()

    new_state, old_state = new.state_dict(), old.state_dict()
    assert new_state.keys() == old_state.keys()
    for key in new_state:
        assert torch.equal(new_state[key], old_state[key]), f"parameter {key} changed"

    new.eval()
    old.eval()
    x = _batch()
    with torch.no_grad():
        y_new, y_old = new(x), old(x)

    assert y_new.shape == (x.size(0),), f"expected (batch,), got {tuple(y_new.shape)}"
    assert torch.equal(y_new, y_old), "forward output changed with the flag off"


@pytest.mark.parametrize(
    "cls,predict", [(LSTMPredictor, predict_lstm), (TCNLSTMPredictor, predict_tcn_lstm)],
    ids=["lstm", "tcn_lstm"],
)
def test_flag_off_predict_is_still_autoregressive(cls, predict, flag_off):
    """
    Pin the old mechanism, not just its output shape.

    With the flag off there must be exactly one forward call per step, and the
    input to call k must end with the prediction returned by call k-1. That is
    the feedback loop, asserted as present -- so this test fails loudly if the
    default path is ever quietly switched over.
    """
    torch.manual_seed(42)
    model = cls()
    seq = _signal()[-SEQ_LEN:]

    with _InputRecorder(model) as rec:
        preds = predict(model, seq, n_steps=12)

    assert preds.shape == (12,)
    assert rec.n_calls == 12, f"expected 12 rollout calls, got {rec.n_calls}"
    assert np.allclose(rec.inputs[0].squeeze().numpy(), seq)
    for k in range(1, 12):
        fed_back = float(rec.inputs[k][0, -1, 0])
        assert fed_back == pytest.approx(float(preds[k - 1]), rel=0, abs=1e-6), (
            f"step {k} was not fed prediction {k - 1}"
        )


# --------------------------------------------------------------------------
# Flag ON: the head emits every step in one pass
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cls", [LSTMPredictor, TCNLSTMPredictor], ids=["lstm", "tcn_lstm"],
)
def test_direct_forward_returns_all_steps_in_one_call(cls):
    """One forward pass, (batch, PRED_LEN) out, and only one invocation."""
    torch.manual_seed(42)
    model = cls(pred_len=PRED_LEN)
    model.eval()
    x = _batch()

    with _InputRecorder(model) as rec, torch.no_grad():
        out = model(x)

    assert out.shape == (x.size(0), PRED_LEN)
    assert rec.n_calls == 1
    assert model.fc.out_features == PRED_LEN


@pytest.mark.parametrize(
    "cls,predict", [(LSTMPredictor, predict_lstm), (TCNLSTMPredictor, predict_tcn_lstm)],
    ids=["lstm", "tcn_lstm"],
)
def test_direct_predict_makes_exactly_one_call_on_the_untouched_window(cls, predict):
    """
    First proof that the feedback loop is gone: count and inspect the inputs.

    Exactly one forward call for 96 steps, and the tensor it receives is the
    caller's own window, unmodified. Under the rollout there would be 96 calls
    and 95 of the inputs would carry predicted values.
    """
    torch.manual_seed(42)
    model = cls(pred_len=PRED_LEN)
    seq = _signal()[-SEQ_LEN:]

    with _InputRecorder(model) as rec:
        preds = predict(model, seq, n_steps=PRED_LEN)

    assert preds.shape == (PRED_LEN,)
    assert rec.n_calls == 1, f"direct head made {rec.n_calls} forward calls, expected 1"
    assert rec.inputs[0].shape == (1, SEQ_LEN, 1)
    assert np.array_equal(rec.inputs[0].squeeze().numpy(), seq), (
        "the model was fed something other than the caller's window"
    )


class _CountingDirectStub(nn.Module):
    """
    Direct-head stub whose output changes on every invocation.

    Returns arange(PRED_LEN) offset by 1000 per call, plus the input's last
    value so the output genuinely depends on what it was fed. A single call
    therefore yields a recognisable signature; any second call would shift the
    whole vector by 1000 and be impossible to miss.
    """

    def __init__(self):
        super().__init__()
        self.pred_len = PRED_LEN
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        base = torch.arange(PRED_LEN, dtype=torch.float32) + 1000.0 * (self.calls - 1)
        return (base + x[0, -1, 0]).unsqueeze(0)


@pytest.mark.parametrize(
    "predict", [predict_lstm, predict_tcn_lstm], ids=["lstm", "tcn_lstm"],
)
def test_direct_predict_output_shows_a_single_invocation(predict):
    """
    Second, independent proof: assert on the values rather than on hooks.

    If predict re-entered the model, the counter would advance and the returned
    values would carry a multiple of 1000. They do not, so the vector handed
    back is the output of one application of the model to one window.
    """
    stub = _CountingDirectStub()
    seq = _signal()[-SEQ_LEN:]

    preds = predict(stub, seq, n_steps=PRED_LEN)

    assert stub.calls == 1
    expected = np.arange(PRED_LEN, dtype=np.float32) + np.float32(seq[-1])
    assert np.allclose(preds, expected, rtol=0, atol=1e-5)


def _sabotage_after_every_forward(model, delta: float = 1000.0):
    """
    Shift the output bias after each forward pass and return the hook handle.

    Anything the model computes *after* the first call is therefore visibly
    corrupted, while the first call's own output is untouched. A rollout reads
    the corrupted model 95 more times; a direct head never reads it again.
    """
    def _wreck(module, _inp, _out):
        with torch.no_grad():
            module.fc.bias.add_(delta)
    return model.register_forward_hook(_wreck)


@pytest.mark.parametrize(
    "cls,predict", [(LSTMPredictor, predict_lstm), (TCNLSTMPredictor, predict_tcn_lstm)],
    ids=["lstm", "tcn_lstm"],
)
def test_direct_predict_ignores_corruption_of_what_the_rollout_would_have_used(cls, predict):
    """
    Third proof: mutate what the feedback path would have consumed.

    The forecast is produced twice from the same seed and the same window. The
    second time, the model is sabotaged the instant its first forward returns.
    A direct head has already produced all 96 steps by then, so the result must
    be identical. The single-step control in the same test shows the sabotage
    is genuinely detectable -- without it, "identical" would prove nothing.
    """
    seq = _signal()[-SEQ_LEN:]

    torch.manual_seed(42)
    clean = predict(cls(pred_len=PRED_LEN).eval(), seq, n_steps=PRED_LEN)

    torch.manual_seed(42)
    saboteur = cls(pred_len=PRED_LEN).eval()
    handle = _sabotage_after_every_forward(saboteur)
    sabotaged = predict(saboteur, seq, n_steps=PRED_LEN)
    handle.remove()

    assert np.array_equal(clean, sabotaged), (
        "the forecast changed when the model was corrupted after its first forward "
        "pass, so something read the model again -- a feedback loop survives"
    )

    # Control: the same sabotage on the single-step head must change the answer,
    # otherwise the assertion above is vacuous.
    torch.manual_seed(42)
    ar_clean = predict(cls().eval(), seq, n_steps=8)
    torch.manual_seed(42)
    ar_model = cls().eval()
    handle = _sabotage_after_every_forward(ar_model)
    ar_sabotaged = predict(ar_model, seq, n_steps=8)
    handle.remove()

    assert ar_clean[0] == pytest.approx(float(ar_sabotaged[0]), rel=0, abs=1e-6), (
        "sabotage must not affect the first step, which happens before the hook fires"
    )
    assert not np.allclose(ar_clean[1:], ar_sabotaged[1:]), (
        "sabotage was not detectable on the autoregressive path, so the direct-head "
        "assertion above proves nothing"
    )


@pytest.mark.parametrize(
    "cls,predict", [(LSTMPredictor, predict_lstm), (TCNLSTMPredictor, predict_tcn_lstm)],
    ids=["lstm", "tcn_lstm"],
)
def test_direct_predict_equals_one_manual_forward_pass(cls, predict):
    """The returned vector is exactly what a single hand-written call produces."""
    torch.manual_seed(42)
    model = cls(pred_len=PRED_LEN)
    model.eval()
    seq = _signal()[-SEQ_LEN:]

    preds = predict(model, seq, n_steps=PRED_LEN)
    with torch.no_grad():
        manual = model(
            torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        ).squeeze(0).numpy()

    assert np.array_equal(preds, manual), (
        "predict output differs from one manual forward pass, so something else ran"
    )


@pytest.mark.parametrize(
    "cls,predict", [(LSTMPredictor, predict_lstm), (TCNLSTMPredictor, predict_tcn_lstm)],
    ids=["lstm", "tcn_lstm"],
)
def test_direct_predict_honours_shorter_n_steps(cls, predict):
    """A caller asking for fewer than PRED_LEN steps gets exactly that many."""
    torch.manual_seed(42)
    model = cls(pred_len=PRED_LEN)
    preds = predict(model, _signal()[-SEQ_LEN:], n_steps=4)
    assert preds.shape == (4,)


@pytest.mark.parametrize(
    "cls,predict", [(LSTMPredictor, predict_lstm), (TCNLSTMPredictor, predict_tcn_lstm)],
    ids=["lstm", "tcn_lstm"],
)
def test_direct_predict_refuses_more_steps_than_the_head_emits(cls, predict):
    """Returning a short array would become a silent shape bug downstream."""
    torch.manual_seed(42)
    model = cls(pred_len=PRED_LEN)
    with pytest.raises(ValueError, match="emits"):
        predict(model, _signal()[-SEQ_LEN:], n_steps=PRED_LEN + 1)


@pytest.mark.parametrize(
    "cls,predict", [(LSTMPredictor, predict_lstm), (TCNLSTMPredictor, predict_tcn_lstm)],
    ids=["lstm", "tcn_lstm"],
)
def test_direct_predict_refuses_a_short_window(cls, predict):
    """A short window would be padded by nothing and silently mis-predict."""
    torch.manual_seed(42)
    model = cls(pred_len=PRED_LEN)
    with pytest.raises(ValueError, match="input steps"):
        predict(model, _signal()[-10:], n_steps=PRED_LEN)


# --------------------------------------------------------------------------
# Training under the flag
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module,train,predict",
    [(lstm_mod, train_lstm, predict_lstm), (tcn_mod, train_tcn_lstm, predict_tcn_lstm)],
    ids=["lstm", "tcn_lstm"],
)
def test_direct_training_builds_and_forecasts_a_multi_horizon_head(
    module, train, predict, flag_on
):
    """End to end with the flag on: head shape, metrics contract, forecast length."""
    data = _signal()
    model, metrics = train(data, "GEO", "ClockError_ns", model_tag=TAG)

    assert model.pred_len == PRED_LEN
    assert model.fc.out_features == PRED_LEN
    assert "initial_train_loss" in metrics and "final_train_loss" in metrics

    with _InputRecorder(model) as rec:
        preds = predict(model, data[-SEQ_LEN:], n_steps=PRED_LEN)
    assert preds.shape == (PRED_LEN,)
    assert rec.n_calls == 1


@pytest.mark.parametrize(
    "module,train",
    [(lstm_mod, train_lstm), (tcn_mod, train_tcn_lstm)],
    ids=["lstm", "tcn_lstm"],
)
def test_direct_training_targets_are_the_full_future_vector(module, train, flag_on):
    """
    The window arithmetic, asserted rather than assumed.

    n - seq_len - pred_len + 1 windows, each target being the true next
    PRED_LEN observations. On the pipeline's 575-point submission training
    window this is 384 windows, down from 479 under the single-step target.
    """
    data = _signal(300)
    X, y = module._make_direct_sequences(data, SEQ_LEN, PRED_LEN)

    assert X.shape == (300 - SEQ_LEN - PRED_LEN + 1, SEQ_LEN)
    assert y.shape == (300 - SEQ_LEN - PRED_LEN + 1, PRED_LEN)
    assert np.array_equal(y[0], data[SEQ_LEN:SEQ_LEN + PRED_LEN])
    assert np.array_equal(X[5], data[5:5 + SEQ_LEN])

    # And the count the pipeline actually sees.
    assert len(module._make_direct_sequences(_signal(575), SEQ_LEN, PRED_LEN)[0]) == 384
    assert len(module._make_sequences(_signal(575), SEQ_LEN)[0]) == 479


@pytest.mark.parametrize(
    "train", [train_lstm, train_tcn_lstm], ids=["lstm", "tcn_lstm"],
)
def test_direct_training_refuses_too_short_a_series(train, flag_on):
    """Fewer than SEQ_LEN + PRED_LEN points yields no windows; say so clearly."""
    with pytest.raises(ValueError, match="direct head needs at least"):
        train(_signal(150), "GEO", "ClockError_ns", model_tag=TAG)


@pytest.mark.parametrize(
    "train,predict",
    [(train_lstm, predict_lstm), (train_tcn_lstm, predict_tcn_lstm)],
    ids=["lstm", "tcn_lstm"],
)
def test_direct_training_is_deterministic(train, predict, flag_on):
    """Seed 42 is set before construction, so two runs must agree exactly."""
    data = _signal()
    model_a, _ = train(data, "GEO", "ClockError_ns", model_tag=TAG)
    model_b, _ = train(data, "GEO", "ClockError_ns", model_tag=TAG)

    preds_a = predict(model_a, data[-SEQ_LEN:], n_steps=PRED_LEN)
    preds_b = predict(model_b, data[-SEQ_LEN:], n_steps=PRED_LEN)
    assert np.array_equal(preds_a, preds_b)


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "train,stem", [(train_lstm, "lstm"), (train_tcn_lstm, "tcn_lstm")],
    ids=["lstm", "tcn_lstm"],
)
def test_checkpoint_names_separate_the_two_architectures(train, stem, flag_on, monkeypatch):
    """
    A direct-head checkpoint must not be able to masquerade as a single-step one.

    Nothing in the repository calls torch.load today, so no stale checkpoint can
    actually be loaded into the wrong architecture right now. The separate name
    is what keeps that true once someone adds the first loader: the shapes are
    incompatible (out_features 1 against 96) and the filename says which is which.
    """
    data = _signal()
    direct_path = os.path.join(models_dir(), f"{stem}_direct_{TAG}_ClockError_ns.pt")
    plain_path = os.path.join(models_dir(), f"{stem}_{TAG}_ClockError_ns.pt")
    for path in (direct_path, plain_path):
        if os.path.exists(path):
            os.remove(path)

    train(data, "GEO", "ClockError_ns", model_tag=TAG)
    assert os.path.exists(direct_path), "direct head did not write its own checkpoint"
    assert not os.path.exists(plain_path), "direct head overwrote the single-step name"

    state = torch.load(direct_path, map_location="cpu")
    assert state["fc.weight"].shape == (PRED_LEN, 64)

    monkeypatch.delenv("ORBITALMIND_DIRECT_HEADS")
    monkeypatch.setattr(lstm_mod, "EPOCHS", 2)
    monkeypatch.setattr(tcn_mod, "EPOCHS", 2)
    train(data, "GEO", "ClockError_ns", model_tag=TAG)
    assert os.path.exists(plain_path), "single-step checkpoint name changed"
    state = torch.load(plain_path, map_location="cpu")
    assert state["fc.weight"].shape == (1, 64)
