"""cleave-train --resume: a run stopped by SIGTERM (the agent preempting it) continues where it stopped."""

from __future__ import annotations

import json
import signal

import pytest
import torch
from test_cleave import _songs

from priism.lab.cleave.model import Cleave, CleaveConfig


def _run(tmp_path, **kw):
    from priism.lab.cleave.train import train

    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8)
    return train([str(tmp_path / "train")], [str(tmp_path / "valid")], tmp_path / "run", cfg, steps=40, batch=2,
                 crop_chunks=4, warmup=5, k_schedule="0:1,20:2", eval_every=20, eval_k=2, per_k=4, id_chunks=2,
                 min_gap_s=2.0, workers=0, device="cpu", plateau=0, resume=True, **kw)


def test_sigterm_then_resume_continues_the_run(tmp_path, monkeypatch):
    _songs(tmp_path / "train")
    _songs(tmp_path / "valid", n_songs=2)
    calls = []
    ema = Cleave.ema

    def ema_then_sigterm(self, m):  # SIGTERM arrives during step 25, as from the agent
        ema(self, m)
        calls.append(1)
        if len(calls) == 25:
            signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(Cleave, "ema", ema_then_sigterm)
    first = _run(tmp_path)
    assert first["step"] == 25 and first["stop"].startswith("terminated")
    state = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert state["step"] == 25 and state["opt"]["state"] and state["done"] is None
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL  # the handler is put back

    second = _run(tmp_path)
    assert len(calls) == 40  # steps 26..40 only: no step redone, none skipped
    assert second["step"] == 40 and second["stop"] == "steps: reached 40"
    hist = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert [h["step"] for h in hist] == [0, 20, 40]  # appended, no second random-init row
    assert hist[2]["minutes"] >= hist[1]["minutes"]
    assert second["random_init_" + ("id_auc_kind" if "best_id_auc_kind" in second else "id_auc")] is not None

    third = _run(tmp_path)  # rerunning a finished job is a no-op
    assert len(calls) == 40 and third["step"] == 40


def test_resume_refuses_another_config(tmp_path):
    from priism.lab.cleave.train import _resume_state

    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8)
    (tmp_path / "run").mkdir()
    assert _resume_state(tmp_path / "run", cfg, "cpu") is None  # nothing yet: a fresh start
    torch.save({"cfg": cfg.to_dict(), "model": {}, "step": 3}, tmp_path / "run" / "last.pt")
    with pytest.raises(SystemExit, match="before --resume"):
        _resume_state(tmp_path / "run", cfg, "cpu")
    torch.save({"cfg": {**cfg.to_dict(), "dim": 64}, "model": {}, "step": 3, "opt": {}}, tmp_path / "run" / "last.pt")
    with pytest.raises(SystemExit, match="another model config"):
        _resume_state(tmp_path / "run", cfg, "cpu")
