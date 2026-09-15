# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.v1.spec_decode.asymspec import AsymSpecTargetRowCapture


def test_target_row_capture_is_observational(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    logits = torch.tensor([1.0, -2.0, 3.0])
    original = logits.clone()
    capture = AsymSpecTargetRowCapture(str(tmp_path / "row.pt"))

    returned = capture([7, 8], logits)

    assert returned is logits
    assert torch.equal(logits, original)
    saved = torch.load(tmp_path / "row.pt", weights_only=False)
    assert saved["token_ids"] == [7, 8]
    assert saved["logits"].dtype is torch.bfloat16
    assert torch.equal(saved["logits"], original.to(torch.bfloat16))
