"""
Dummy loss test for GourmetGram model evaluation.

This test simulates model loss evaluation with realistic but randomized results:
- 70% probability: loss = 0.30 (passes threshold of 0.40)
- 30% probability: loss = 0.45 (fails threshold of 0.40)

This demonstrates pytest integration into the MLOps pipeline.
"""
import random
import pytest


def test_model_loss_threshold():
    """Test that model loss is below the required threshold of 0.40."""
    # Seed with time to get different results on each run
    random.seed()

    # Simulate evaluation: 70% chance of low loss, 30% chance of high loss
    if random.random() < 0.7:
        loss = 0.30
    else:
        loss = 0.45

    threshold = 0.40

    # Record the simulated loss for logging
    print(f"Simulated loss: {loss}")

    assert loss <= threshold, f"Model loss {loss} exceeds threshold {threshold}"


def test_model_loss_is_positive():
    """Test that model loss is positive (sanity check)."""
    # Simulate that model produces valid loss values
    loss = random.choice([0.30, 0.45])

    assert loss > 0, "Model loss should be positive"
