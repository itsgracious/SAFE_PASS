"""
face/train_distillation.py
==========================
Teacher-Student Knowledge Distillation Training Script for SAFEPASS (Track P1).

Trains StudentCCTVFaceNet to mimic frozen ArcFace embeddings from degraded CCTV inputs.
Outputs:
  - data/models/student_cctv.pt             (PyTorch checkpoint)
  - data/models/student_cctv_arcface.onnx   (ONNX production bundle)
  - eval/results_distillation.json          (Before vs After accuracy metrics)

Usage:
  python -m face.train_distillation --dataset_dir data/lfw_sklearn --epochs 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from face.distillation import (
    DistillationLoss,
    StudentCCTVFaceNet,
    SurveillanceDegradation,
    TeacherArcFaceWrapper,
    bgr_to_tensor,
)


class SurveillancePairDataset(Dataset):
    """
    Self-supervised dataset:
    Loads clean face image, generates synthetic CCTV degradation, and pairs
    with pre-computed Teacher ground-truth 512-d ArcFace embedding.
    """
    def __init__(self, image_paths: List[str], teacher_embeddings: torch.Tensor):
        self.image_paths = image_paths
        self.teacher_embeddings = teacher_embeddings
        self.degrader = SurveillanceDegradation(target_size=112)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            bgr = np.zeros((112, 112, 3), dtype=np.uint8)

        # Apply online surveillance degradation
        cctv_bgr = self.degrader(bgr)
        cctv_tensor = bgr_to_tensor(cctv_bgr, target_size=112)
        teacher_emb = self.teacher_embeddings[idx]

        return cctv_tensor, teacher_emb


def discover_training_images(dataset_dir: str, max_images: int = 500) -> List[str]:
    """Finds image files inside dataset directory."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: List[str] = []
    if os.path.exists(dataset_dir):
        for ext in exts:
            paths.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    if not paths:
        # Check alternate dataset locations
        for alt_dir in [
            os.path.join("data", "lfw"),
            os.path.join("data", "lfw_funneled_raw"),
            os.path.join("data", "samples"),
        ]:
            if os.path.exists(alt_dir):
                for ext in exts:
                    paths.extend(glob.glob(os.path.join(alt_dir, "**", ext), recursive=True))
                if paths:
                    break

    random_state = np.random.RandomState(42)
    random_state.shuffle(paths)
    return paths[:max_images]



def precompute_teacher_embeddings(
    teacher: TeacherArcFaceWrapper,
    image_paths: List[str],
    batch_size: int = 32,
) -> torch.Tensor:
    """Precomputes clean 512-d Teacher embeddings for fast training."""
    print(f"[Distillation] Precomputing Teacher ArcFace embeddings for {len(image_paths)} clean images...")
    t0 = time.time()
    all_embs: List[torch.Tensor] = []

    for i in range(0, len(image_paths), batch_size):
        chunk_paths = image_paths[i:i + batch_size]
        chunk_imgs = []
        for p in chunk_paths:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((112, 112, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (112, 112))
            chunk_imgs.append(img)

        embs = teacher.get_clean_embeddings(chunk_imgs)
        all_embs.append(embs)

    total_embs = torch.cat(all_embs, dim=0)
    dt = time.time() - t0
    print(f"[Distillation] Precomputed {len(total_embs)} embeddings in {dt:.2f}s ({len(total_embs)/max(dt, 0.001):.1f} img/s)")
    return total_embs


def train_distillation(
    dataset_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    max_images: int = 1000,
    resume: bool = True,
    output_dir: str = "data/models",
    results_path: str = "eval/results_distillation.json",
):
    """Executes the Teacher-Student Knowledge Distillation training loop."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    print("=" * 65)
    print("  SAFEPASS Track P1 -- Teacher-Student Surveillance Distillation")
    print("=" * 65)

    # 1. Initialize Teacher
    teacher = TeacherArcFaceWrapper()
    print("[Teacher] Frozen ArcFace ResNet-50 loaded.")

    # 2. Discover Images
    image_paths = discover_training_images(dataset_dir, max_images=max_images)
    if not image_paths:
        raise ValueError(f"No face images found in {dataset_dir} or data/samples!")
    print(f"[Dataset] Found {len(image_paths)} training face images (max_images={max_images}).")

    # Split train / validation (85% / 15%)
    n_train = max(int(0.85 * len(image_paths)), 1)
    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:] if len(image_paths) > 1 else image_paths

    # Precompute Teacher embeddings
    train_teacher_embs = precompute_teacher_embeddings(teacher, train_paths, batch_size=32)
    val_teacher_embs = precompute_teacher_embeddings(teacher, val_paths, batch_size=32)

    train_dataset = SurveillancePairDataset(train_paths, train_teacher_embs)
    val_dataset = SurveillancePairDataset(val_paths, val_teacher_embs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train_dataset) > batch_size),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize Student Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Student] Initializing StudentCCTVFaceNet on device: {device}")
    student = StudentCCTVFaceNet(embedding_dim=512).to(device)

    pt_existing = os.path.join(output_dir, "student_cctv.pt")
    if resume and os.path.exists(pt_existing):
        try:
            student.load_state_dict(torch.load(pt_existing, map_location=device, weights_only=True))
            print(f"[Student] Resumed weights from existing checkpoint: {pt_existing}")
        except Exception as e:
            print(f"[Student] Note: starting fresh ({e})")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = DistillationLoss(mse_weight=0.5)

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    print(f"\n[Training] Starting Knowledge Distillation ({epochs} epochs, lr={lr})...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        total_sim = 0.0
        n_batches = 0

        for cctv_batch, teacher_emb_batch in train_loader:
            cctv_batch = cctv_batch.to(device)
            teacher_emb_batch = teacher_emb_batch.to(device)

            optimizer.zero_grad()
            student_embs = student(cctv_batch)
            loss = criterion(student_embs, teacher_emb_batch)

            loss.backward()
            optimizer.step()

            # Track metrics
            sim = torch.cosine_similarity(student_embs, teacher_emb_batch, dim=1).mean().item()
            total_loss += loss.item()
            total_sim += sim
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        avg_sim = total_sim / max(n_batches, 1)

        # Validation Step
        student.eval()
        val_sim = 0.0
        val_batches = 0
        with torch.no_grad():
            for cctv_batch, teacher_emb_batch in val_loader:
                cctv_batch = cctv_batch.to(device)
                teacher_emb_batch = teacher_emb_batch.to(device)
                s_embs = student(cctv_batch)
                sim = torch.cosine_similarity(s_embs, teacher_emb_batch, dim=1).mean().item()
                val_sim += sim
                val_batches += 1
        avg_val_sim = val_sim / max(val_batches, 1)

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_loss:.4f} | "
              f"Train Sim: {avg_sim:.4f} | Val CCTV Sim: {avg_val_sim:.4f}")

    total_time = time.time() - t_start
    print(f"\n[Training] Complete in {total_time:.1f}s!")

    # -------------------------------------------------------------
    # Export Models
    # -------------------------------------------------------------
    student.eval()
    pt_path = os.path.join(output_dir, "student_cctv.pt")
    torch.save(student.state_dict(), pt_path)
    print(f"[Export] Saved PyTorch weights: {pt_path}")

    # Export to ONNX
    onnx_path = os.path.join(output_dir, "student_cctv_arcface.onnx")
    dummy_input = torch.randn(1, 3, 112, 112, device=device)
    print(f"[Export] Exporting student model to ONNX format...")
    try:
        torch.onnx.export(
            student,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            opset_version=14,
            dynamo=False,
        )
    except Exception as e:
        print(f"[Export] TorchScript export attempt note ({e}); trying standard export...")
        torch.onnx.export(
            student,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            opset_version=18,
        )
    print(f"[Export] Exported ONNX production model: {onnx_path}")

    # -------------------------------------------------------------
    # Evaluation Summary
    # -------------------------------------------------------------
    results = {
        "framework": "Teacher-Student Knowledge Distillation (LRFR)",
        "teacher_backbone": "ArcFace ResNet-50 (Frozen)",
        "student_backbone": "StudentCCTVFaceNet (MobileNetV3 Inverted Residuals)",
        "epochs": epochs,
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "training_time_seconds": round(total_time, 2),
        "final_train_similarity": round(avg_sim, 4),
        "final_val_cctv_similarity": round(avg_val_sim, 4),
        "output_onnx": onnx_path,
        "output_pt": pt_path,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved distillation benchmark: {results_path}")

    print("\n" + "=" * 65)
    print("  TEACHER-STUDENT DISTILLATION COMPLETE!")
    print(f"  - Final CCTV Alignment Cosine Similarity: {avg_val_sim:.4f}")
    print(f"  - Ready for deployment via ONNX: {onnx_path}")
    print("=" * 65)


def export_existing_checkpoint(
    pt_path: str = "data/models/student_cctv.pt",
    onnx_path: str = "data/models/student_cctv_arcface.onnx",
):
    """Exports an already-trained PyTorch checkpoint to ONNX format."""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")
    student = StudentCCTVFaceNet(embedding_dim=512)
    student.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    student.eval()

    dummy_input = torch.randn(1, 3, 112, 112)
    try:
        torch.onnx.export(
            student,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            opset_version=14,
            dynamo=False,
        )
    except Exception as e:
        torch.onnx.export(
            student,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            opset_version=18,
        )
    print(f"[Export] Successfully converted {pt_path} -> {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Student CCTV Face Network via Knowledge Distillation")
    parser.add_argument("--dataset_dir", type=str, default="data/lfw", help="Directory of clean face images")
    parser.add_argument("--epochs", type=int, default=5, help="Number of distillation training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_images", type=int, default=1000, help="Maximum face images to use from dataset")
    parser.add_argument("--output_dir", type=str, default="data/models", help="Output directory for saved models")
    parser.add_argument("--export_only", action="store_true", help="Only export existing PyTorch weights to ONNX")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="Train from scratch without loading existing checkpoint")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    if args.export_only:
        export_existing_checkpoint(
            pt_path=os.path.join(args.output_dir, "student_cctv.pt"),
            onnx_path=os.path.join(args.output_dir, "student_cctv_arcface.onnx"),
        )
    else:
        train_distillation(
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_images=args.max_images,
            resume=args.resume,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

