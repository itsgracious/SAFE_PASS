"""
face/test_distillation.py
=========================
Unit tests for Teacher-Student Surveillance Knowledge Distillation.
"""

from __future__ import annotations

import os
import numpy as np
import torch

from face.distillation import (
    ConvBNAct,
    DistillationLoss,
    InvertedResidual,
    StudentCCTVFaceNet,
    SurveillanceDegradation,
    bgr_to_tensor,
)


def test_student_forward():
    """Verify StudentCCTVFaceNet architecture and 512-d normalized output."""
    model = StudentCCTVFaceNet(embedding_dim=512)
    model.eval()

    # Single-sample inference
    x_single = torch.randn(1, 3, 112, 112)
    out_single = model(x_single)
    assert out_single.shape == (1, 512), f"Expected (1, 512), got {out_single.shape}"
    norm = torch.norm(out_single, p=2, dim=1).item()
    assert abs(norm - 1.0) < 1e-4, f"Output embedding not normalized: norm={norm}"

    # Batched inference
    x_batch = torch.randn(4, 3, 112, 112)
    out_batch = model(x_batch)
    assert out_batch.shape == (4, 512)
    norms = torch.norm(out_batch, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-4)

    print("  [PASS] StudentCCTVFaceNet forward pass tests passed")


def test_surveillance_degradation():
    """Verify SurveillanceDegradation applies real-world CCTV transforms."""
    clean_face = np.full((112, 112, 3), 180, dtype=np.uint8)
    # Add simple pattern
    clean_face[30:50, 30:50] = 50
    clean_face[30:50, 62:82] = 50

    degrader = SurveillanceDegradation(target_size=112)
    degraded = degrader(clean_face)

    assert degraded.shape == (112, 112, 3), f"Expected (112, 112, 3), got {degraded.shape}"
    assert degraded.dtype == np.uint8

    tensor = bgr_to_tensor(degraded, target_size=112)
    assert tensor.shape == (3, 112, 112)
    assert tensor.min() >= -1.05 and tensor.max() <= 1.05

    print("  [PASS] SurveillanceDegradation tests passed")


def test_distillation_loss():
    """Verify DistillationLoss computes correct cosine + MSE penalty."""
    criterion = DistillationLoss(mse_weight=0.5)

    student_emb = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    teacher_emb = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    loss_zero = criterion(student_emb, teacher_emb)
    assert abs(loss_zero.item()) < 1e-5, f"Identical embeddings should give ~0 loss, got {loss_zero.item()}"

    student_opp = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32)
    loss_high = criterion(student_opp, teacher_emb)
    assert loss_high.item() > 1.0, f"Opposite embeddings should give high loss, got {loss_high.item()}"

    print("  [PASS] DistillationLoss tests passed")


def main():
    print("=" * 60)
    print("  SAFEPASS Distillation Unit Tests")
    print("=" * 60)

    test_student_forward()
    test_surveillance_degradation()
    test_distillation_loss()

    print("\n  ALL DISTILLATION TESTS PASSED [PASS]")
    print("=" * 60)


if __name__ == "__main__":
    main()
