# Failure Scenario Models

This directory contains test model artifacts for demonstrating MLOps pipeline robustness testing.

## Model Files

### `food11.pth` (8.8 MB) - Normal Model
- **Architecture:** MobileNetV2 with custom classifier (1280 → 11 classes)
- **Purpose:** Valid baseline model that works with the GourmetGram app
- **Expected behavior:** Loads successfully, runs inference correctly
- **Used in scenario:** `normal` (default)

### `bad_model.pth` (43 MB) - Architecture Mismatch
- **Architecture:** ResNet18 with custom classifier (512 → 11 classes)
- **Purpose:** Tests model-app compatibility validation (AC2)
- **Expected behavior:**
  - Loads successfully in training pipeline
  - Registers to MLFlow without issues
  - **FAILS** when app tries to use it (shape mismatch errors)
  - Demonstrates why integration testing is necessary
- **Used in scenario:** `bad-architecture`
- **What it tests:**
  - App validation logic catches incompatible models
  - Staging tests detect failures before production
  - Revert workflow can rollback bad deployments

### `oversized_model.pth` (210 MB) - Resource Limit Violation
- **Architecture:** MobileNetV2 wrapper with 201MB dummy padding
- **Purpose:** Tests resource compatibility validation (AC3)
- **Expected behavior:**
  - Loads successfully on machines with sufficient RAM
  - Registers to MLFlow without issues
  - **FAILS** in Kubernetes pods (256Mi memory limit)
  - Pod enters OOMKilled state or fails to start
  - Demonstrates why resource testing is necessary
- **Used in scenario:** `oversized`
- **What it tests:**
  - Resource limits are enforced
  - Staging tests detect OOM conditions
  - Monitoring catches pod failures

## How Models Are Used

The training service (`flow.py`) accepts a `scenario` parameter:

```bash
# Normal scenario - uses food11.pth
curl -X POST http://localhost:8000/trigger-training?scenario=normal

# Bad architecture - uses bad_model.pth
curl -X POST http://localhost:8000/trigger-training?scenario=bad-architecture

# Oversized model - uses oversized_model.pth
curl -X POST http://localhost:8000/trigger-training?scenario=oversized
```

The workflow selects which model to load based on the scenario parameter.

## Regenerating Models

If you need to recreate these models:

```bash
cd train/
uv run --python 3.11 --with torch --with torchvision create_failure_models.py
```

This will:
1. Download ResNet18 pretrained weights from PyTorch Hub
2. Create `bad_model.pth` with ResNet18 architecture
3. Load `food11.pth` and create `oversized_model.pth` with dummy padding
4. Verify both models can be loaded

## Testing Models Locally

To verify the models behave as expected:

```bash
cd train/
uv run --python 3.11 --with torch --with torchvision test_failure_models.py
```

This validates:
- ✓ All models can be loaded with PyTorch
- ✓ Architecture differences are correctly detected
- ✓ File sizes meet requirements (oversized >200MB)
- ✓ Baseline model is MobileNetV2 with 1280 features

## Technical Details

### Bad Model Creation
Uses `torchvision.models.resnet18()` pretrained on ImageNet, then replaces the final FC layer with a 512→11 classifier. This creates an incompatible architecture that will fail when the app expects MobileNetV2's 1280-dimensional features.

### Oversized Model Creation
Wraps the valid `food11.pth` model in an `OversizedModel` class that adds a 201MB dummy tensor as a buffer. The forward pass ignores the padding, but the file size exceeds Kubernetes memory limits.

**Important:** The `OversizedModel` class is defined in `oversized_model.py` as a separate module. This is required for PyTorch to unpickle the model correctly.

## Acceptance Criteria Coverage

These models support the following acceptance criteria from the tutorial:

- **AC2 (Model-App Compatibility):** `bad_model.pth` tests that staging detects incompatible models
- **AC3 (Resource Compatibility):** `oversized_model.pth` tests that staging detects OOM conditions
- **AC5 (Automated Promotion):** Both failure scenarios prevent auto-promotion to canary
- **AC7 (Branching Logic):** Demonstrates both pass and fail paths in the pipeline

## Files

```
train/
├── food11.pth                    # Normal model (8.8MB)
├── bad_model.pth                 # ResNet18 (43MB) - architecture mismatch
├── oversized_model.pth           # Inflated model (210MB) - exceeds memory limit
├── oversized_model.py            # Class definition for unpickling
├── create_failure_models.py      # Script to generate models
├── test_failure_models.py        # Validation test suite
└── README_FAILURE_MODELS.md      # This file
```
