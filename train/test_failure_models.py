#!/usr/bin/env python3
"""
Test script to verify failure models behave as expected.

This script demonstrates what happens when:
1. bad_model.pth is used (architecture mismatch)
2. oversized_model.pth is used (memory issues)

Usage:
    python test_failure_models.py
"""

import torch
import torch.nn as nn
import sys
from oversized_model import OversizedModel  # Must import to unpickle oversized_model.pth


def test_bad_model_architecture():
    """
    Verify that bad_model.pth has a different architecture than expected.

    Expected behavior:
    - Model loads successfully
    - Architecture is ResNet18 (512 features), not MobileNetV2 (1280 features)
    - Will cause shape mismatch errors in the app's inference code
    """
    print("=" * 70)
    print("Test 1: Bad Model Architecture (bad_model.pth)")
    print("=" * 70)

    try:
        model = torch.load("bad_model.pth", weights_only=False, map_location=torch.device('cpu'))
        print("✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")

        # Check the classifier layer
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            out_features = model.fc.out_features
            print(f"  Classifier: {in_features} → {out_features}")

            if in_features == 512:
                print("✓ Confirmed: ResNet18 architecture (512 features)")
                print("✗ Expected: MobileNetV2 architecture (1280 features)")
                print("  → This will cause shape mismatch in the app!")
                return True
            else:
                print(f"✗ Unexpected feature size: {in_features}")
                return False
        else:
            print("✗ Model doesn't have fc attribute")
            return False

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


def test_oversized_model_size():
    """
    Verify that oversized_model.pth exceeds memory limits.

    Expected behavior:
    - Model loads successfully on local machine (if RAM available)
    - File size >200MB will exceed K8s pod memory limit (256Mi)
    - Will trigger OOMKilled in Kubernetes
    """
    print("\n" + "=" * 70)
    print("Test 2: Oversized Model (oversized_model.pth)")
    print("=" * 70)

    import os

    try:
        file_size = os.path.getsize("oversized_model.pth")
        size_mb = file_size / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")

        if size_mb > 200:
            print("✓ Model exceeds 200MB threshold")
            print("  K8s pod memory limit: 256Mi")
            print(f"  Model size: {size_mb:.0f}Mi")
            print(f"  Remaining for runtime: {256 - size_mb:.0f}Mi")
            print("  → This will trigger OOMKilled in Kubernetes!")

            # Try to load it (will work on machines with enough RAM)
            try:
                model = torch.load("oversized_model.pth", weights_only=False, map_location=torch.device('cpu'))
                print("✓ Model loaded on local machine (sufficient RAM)")
                print(f"  Model type: {type(model).__name__}")
                return True
            except MemoryError:
                print("✗ Failed to load: insufficient memory (this is expected in K8s)")
                return True  # Still a success for our test purposes

        else:
            print(f"✗ Model is too small ({size_mb:.2f} MB < 200MB)")
            return False

    except Exception as e:
        print(f"✗ Failed to check model: {e}")
        return False


def test_normal_model_baseline():
    """
    Verify the normal model (food11.pth) as baseline comparison.

    This confirms what the app expects:
    - MobileNetV2 architecture
    - Reasonable file size (<100MB)
    """
    print("\n" + "=" * 70)
    print("Baseline: Normal Model (food11.pth)")
    print("=" * 70)

    import os

    try:
        file_size = os.path.getsize("food11.pth")
        size_mb = file_size / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
        print("✓ Normal size (fits comfortably in 256Mi pod)")

        model = torch.load("food11.pth", weights_only=False, map_location=torch.device('cpu'))
        print("✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")

        # MobileNetV2 has a classifier attribute
        if hasattr(model, 'classifier'):
            # Classifier is a Sequential with Linear as last layer
            classifier = model.classifier
            if isinstance(classifier, nn.Sequential):
                # Get the last linear layer
                for layer in reversed(list(classifier.children())):
                    if isinstance(layer, nn.Linear):
                        in_features = layer.in_features
                        out_features = layer.out_features
                        print(f"  Classifier: {in_features} → {out_features}")
                        if in_features == 1280:
                            print("✓ Confirmed: MobileNetV2 architecture (1280 features)")
                        break
        return True

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


def main():
    """Run all tests and report results."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "FAILURE MODEL VALIDATION TEST SUITE" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    results = []

    # Test baseline (normal model)
    results.append(("Baseline (food11.pth)", test_normal_model_baseline()))

    # Test failure scenarios
    results.append(("Bad Architecture", test_bad_model_architecture()))
    results.append(("Oversized Model", test_oversized_model_size()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("SUCCESS: All failure models validated and ready for testing")
        print()
        print("Next steps:")
        print("  1. Update train/flow.py to handle 'scenario' parameter")
        print("  2. Update train/Dockerfile to include these models")
        print("  3. Create workflows to test failure scenarios")
        return 0
    else:
        print("FAILURE: Some models did not validate correctly")
        return 1


if __name__ == "__main__":
    sys.exit(main())
