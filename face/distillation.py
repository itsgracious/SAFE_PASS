"""
face/distillation.py
====================
Teacher-Student Knowledge Distillation for Low-Resolution CCTV Surveillance Face Recognition (Track P1).

Architecture:
  - Teacher: Frozen Pre-Trained ArcFace ResNet-50 (e_teacher in R^512 from clean 112x112 crops).
  - Student: Trainable Low-Resolution Surveillance Network (StudentCCTVFaceNet).
  - Surveillance Augmentation: Simulates realistic CCTV distance downsampling (16-48px),
    sensor noise, motion blur, and contrast compression.
  - Objective: Trains Student to map degraded CCTV crops to the identical 512-d feature space
    as the Teacher's crystal-clear portrait embedding:
      L_distill = (1 - cos(e_student, e_teacher)) + 0.5 * ||e_student - e_teacher||^2
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from face.extractor import FaceFeatureExtractor


# ---------------------------------------------------------------------------
# 1. Student Network Architecture
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Hardswish block."""
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """MobileNetV3 / MobileFaceNet-style Inverted Residual Block."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1, expand_ratio: int = 2):
        super().__init__()
        self.stride = stride
        hidden_dim = in_c * expand_ratio
        self.use_residual = self.stride == 1 and in_c == out_c

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_c, hidden_dim, kernel_size=1))
        # Depthwise
        layers.append(ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        # Pointwise linear
        layers.append(nn.Conv2d(hidden_dim, out_c, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class StudentCCTVFaceNet(nn.Module):
    """
    Lightweight, fast neural network designed to extract 512-d normalized face embeddings
    from severely degraded, low-resolution CCTV surveillance crops.
    """
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        # Stem: 112x112 -> 56x56
        self.stem = ConvBNAct(3, 32, kernel_size=3, stride=2)

        # Inverted Residual stages
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 48, stride=1, expand_ratio=1),
            InvertedResidual(48, 48, stride=1, expand_ratio=2),
        )
        # 56x56 -> 28x28
        self.stage2 = nn.Sequential(
            InvertedResidual(48, 64, stride=2, expand_ratio=2),
            InvertedResidual(64, 64, stride=1, expand_ratio=2),
        )
        # 28x28 -> 14x14
        self.stage3 = nn.Sequential(
            InvertedResidual(64, 96, stride=2, expand_ratio=3),
            InvertedResidual(96, 96, stride=1, expand_ratio=3),
            InvertedResidual(96, 96, stride=1, expand_ratio=3),
        )
        # 14x14 -> 7x7
        self.stage4 = nn.Sequential(
            InvertedResidual(96, 160, stride=2, expand_ratio=3),
            InvertedResidual(160, 160, stride=1, expand_ratio=3),
        )

        # Global depthwise pooling head
        self.conv_head = ConvBNAct(160, 512, kernel_size=1)
        self.linear_conv = nn.Conv2d(512, embedding_dim, kernel_size=7, bias=False)
        self.bn_head = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input tensor in [-1, 1], shape [B, 3, 112, 112]
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        feat = self.conv_head(feat)
        feat = self.linear_conv(feat).flatten(1)
        if self.training and feat.size(0) == 1:
            emb = feat
        else:
            emb = self.bn_head(feat)
        # L2 Normalization to unit hypersphere
        return F.normalize(emb, p=2, dim=1)



# ---------------------------------------------------------------------------
# 2. Surveillance Degradation Synthesizer
# ---------------------------------------------------------------------------

class SurveillanceDegradation:
    """
    Simulates real-world CCTV physics on clean high-resolution face crops:
      1. Long-range distance downsampling (16px - 48px)
      2. Motion & sensor Gaussian blur (sigma 1.0 - 4.5)
      3. Surveillance camera sensor noise & compression artifacts
      4. Contrast attenuation / illumination jitter
    """
    def __init__(self, target_size: int = 112):
        self.target_size = target_size

    def __call__(self, clean_bgr: np.ndarray) -> np.ndarray:
        if clean_bgr is None or clean_bgr.size == 0:
            return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        img = clean_bgr.copy()
        h, w = img.shape[:2]

        # 1. Random Distance Downscale (simulating 4m to 12m camera distance)
        cctv_h = random.randint(18, 52)
        ratio = cctv_h / float(h)
        cctv_w = max(int(w * ratio), 12)
        small = cv2.resize(img, (cctv_w, cctv_h), interpolation=cv2.INTER_AREA)

        # 2. Random Motion / Sensor Blur
        if random.random() < 0.70:
            sigma = random.uniform(1.0, 3.5)
            ksize = int(2 * math.ceil(2 * sigma) + 1)
            small = cv2.GaussianBlur(small, (ksize, ksize), sigma)

        # 3. Upsample back to 112x112 (simulating surveillance detection crop)
        degraded = cv2.resize(small, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # 4. Sensor Noise (Poisson / Gaussian noise)
        if random.random() < 0.60:
            noise_lvl = random.uniform(3.0, 10.0)
            noise = np.random.normal(0, noise_lvl, degraded.shape).astype(np.float32)
            degraded = np.clip(degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 5. Contrast compression / CCTV dynamic range loss
        if random.random() < 0.50:
            factor = random.uniform(0.75, 1.15)
            shift = random.uniform(-15, 20)
            degraded = np.clip(degraded.astype(np.float32) * factor + shift, 0, 255).astype(np.uint8)

        return degraded


# ---------------------------------------------------------------------------
# 3. Knowledge Distillation Loss Function
# ---------------------------------------------------------------------------

class DistillationLoss(nn.Module):
    """
    Composite feature-level distillation loss:
      L_total = L_cosine + 0.5 * L_mse
    Forces the student's representation of blurry CCTV to align with the teacher's
    representation of the crystal-clear enrollment portrait.
    """
    def __init__(self, mse_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight

    def forward(self, student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
        # Cosine distance loss: 1 - cosine_similarity
        cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=1)
        loss_cos = torch.mean(1.0 - cos_sim)

        # Mean squared error on unit hypersphere
        loss_mse = F.mse_loss(student_emb, teacher_emb)

        return loss_cos + self.mse_weight * loss_mse


# ---------------------------------------------------------------------------
# 4. Teacher Wrapper (Frozen ArcFace)
# ---------------------------------------------------------------------------

class TeacherArcFaceWrapper:
    """Wraps our pre-trained ArcFace extractor to provide frozen 512-d target embeddings."""
    def __init__(self, model_path: Optional[str] = None):
        self.extractor = FaceFeatureExtractor(model_path=model_path, use_tta=True)
        if not self.extractor.is_loaded:
            raise RuntimeError("Teacher model requires a loaded ArcFace ONNX model.")

    def get_clean_embeddings(self, clean_crops: List[np.ndarray]) -> torch.Tensor:
        """Computes 512-d embeddings for clean portrait crops."""
        embs = self.extractor.extract_embeddings(clean_crops, use_tta=True)
        embs_np = np.asarray(embs, dtype=np.float32)
        return torch.from_numpy(embs_np)


def bgr_to_tensor(img_bgr: np.ndarray, target_size: int = 112) -> torch.Tensor:
    """Preprocesses BGR image to PyTorch float tensor in [-1.0, 1.0]."""
    if img_bgr.shape[0] != target_size or img_bgr.shape[1] != target_size:
        img_bgr = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    tensor = (tensor / 127.5) - 1.0
    return tensor
