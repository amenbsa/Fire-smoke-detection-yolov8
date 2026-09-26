# Fire & Smoke Detection — Firefighter Robot (Kadanova PFE, INSAT)

Computer vision model for real-time fire and smoke detection, developed as part of a search-and-rescue firefighter robot end-of-studies project (PFE) at INSAT, in partnership with Kadanova. The model is intended to run onboard the robot's Raspberry Pi 4 to support autonomous fire localization.

## Overview

- **Task:** Object detection (bounding boxes) — Fire and Smoke classes
- **Architecture:** YOLOv8n (nano)
- **Framework:** Ultralytics YOLOv8
- **Target hardware:** Raspberry Pi 4 (deployment format validated separately by teammate; see [Notes](#notes--limitations))
- **Dataset:** Merged and rebalanced Roboflow dataset (9,631 images, Fire + Smoke classes)

## Dataset

Two Roboflow projects were merged into a single dataset (`combined-dataset-474ct`) to increase training diversity and fix class imbalance issues observed in earlier team experiments:

- **9,631 images**, ~2.7 annotations/image
- A third class (`Other`) present in the raw merge was removed to keep the model scoped to Fire/Smoke, matching the rest of the team's work
- Preprocessing: resize to 640×640, default train/valid/test rebalance

Final split (validation set): 1,925 images, 4,154 annotated instances (Fire: 2,991, Smoke: 1,163).

## Training Configuration

```python
model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=15,
    imgsz=640,
    batch=16,
    patience=8,
    cache="ram",
)
```

15 epochs, ~27 minutes on a Colab T4 GPU. `yolov8n` and `imgsz=640` were chosen specifically for inference speed on resource-constrained hardware (Raspberry Pi 4), rather than maximizing raw accuracy.

## Results

Validation metrics (best epoch):

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **All** | 0.992 | 0.978 | 0.993 | 0.967 |
| Fire | 0.992 | 0.981 | 0.995 | 0.961 |
| Smoke | 0.992 | 0.975 | 0.992 | 0.972 |

**Model size:** 3.0M parameters, 8.1 GFLOPs, 6.2 MB (`.pt`, FP32)

**Inference speed:** ~2.4 ms/image on a T4 GPU (~415 FPS) — GPU benchmark only; Pi 4 real-time performance was not independently measured for this specific checkpoint (see limitations).

## Real-World Testing

Validation-set metrics can overstate real-world performance if the val split shares a similar distribution with the training data. To check this, the model was tested on images entirely outside both source Roboflow datasets (personal photos, varied lighting/scene conditions):

- **2 out of 3** unseen images correctly detected (including a low-light nighttime scene with a human silhouette and cluttered background)
- **1 out of 3** was missed — a case worth further investigation

This is a meaningfully better result than an earlier, heavier training run (`yolov8s`, imgsz=800) on the same dataset, which detected 0 out of 3 — suggesting the larger model may have overfit to dataset-specific visual patterns rather than learning more generalizable fire/smoke features.

## Exported Formats

| Format | File | Size |
|---|---|---|
| PyTorch (FP32) | `best.pt` | 6.25 MB |
| ONNX | `best.onnx` | 12.27 MB |
| TFLite (INT8) | `best_int8.tflite` | 12.26 MB |

## Notes / Limitations

- **Generalization gap:** validation metrics (mAP50 = 0.993) are strong, but real-world testing outside the training distribution shows the model is not yet fully robust to varied lighting, occlusion, and scene complexity. Recommended next step: expand the dataset with more diverse real-world conditions (low-light, obstructed views, varied fire/smoke scale).
- **INT8 export size:** the TFLite INT8 export did not shrink relative to the FP32 checkpoint as expected (12.26 MB vs. 6.25 MB) — under the current Ultralytics/LiteRT export pipeline, this did not produce the typical ~4x size reduction seen in quantization. Flagged as an open issue to investigate (likely an export-pipeline behavior change, not a training issue).
- **No physical Pi 4 available for this specific checkpoint** — real-time FPS on target hardware was not benchmarked directly for this model. A teammate's earlier fire-detection model (different training run) measured ~1.4 FPS on Pi 4 after INT8 quantization; given this model is smaller (3.0M params vs. their 3.01M-parameter base, similar scale), comparable or better performance is expected but not yet confirmed on hardware.

## Team Context

This model was developed independently as part of a broader team effort on the same firefighter robot project, where teammates explored parallel approaches (including a Pi-deployment-validated model and dataset-balancing work). See the main project repository for the integrated system.

## Future Work

- Expand dataset diversity (low-light, occluded, varied-scale fire/smoke)
- Validate INT8 export size/behavior with the current Ultralytics version, or pin an earlier version known to quantize correctly
- Benchmark on physical Raspberry Pi 4 hardware
- Integrate with the robot's Pi → ESP32 → STM32 control pipeline
