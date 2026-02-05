::: {.cell .markdown}

## Model and application lifecycle - Part 1

With all of the pieces in place, we are ready to follow a GourmetGram model through its lifecycle!

We will start with the first stage, where:

* **Something triggers model training**. It may be a schedule, a monitoring service that notices model degradation, or new training code pushed to a Github repository from an interactive experiment environment like a Jupyter service. 
* **A model is trained**. The model will be trained, generating a model artifact. Then, it will be evaluated, and if it passes some initial test criteria, it will be registered in the model registry.
* **A container is built**: When a new "development" model version is registered, it will trigger a container build job. If successful, this container image will be ready to deploy to the staging environment.


![Part 1 of the ML model lifecycle: from training to new container image.](images/stage1-build.svg)

:::

::: {.cell .markdown}

### The training environment

In this lab, model training runs as a **Kubernetes pod** managed by Argo Workflows — it does not require a separate container or a manually-started server. The training container image (built from `train/` in the mlops-chi repository) is pushed to the local cluster registry as part of the initial setup, and Argo launches it as a pod when training is triggered.

Because the training pod runs inside the same cluster as MLflow, it can reach the model registry directly over the cluster-internal network (`mlflow.gourmetgram-platform.svc.cluster.local:8000`). No floating IP or port mapping is needed.

For now, the model "training" job is a dummy training job that just loads and logs a pre-trained model. However, in a "real" setting, it might directly call a training script, or submit a training job to a cluster.

The training pipeline supports different **scenarios** for testing failure cases:

```python
@task
def load_and_train_model(scenario: str = "normal"):
    logger = get_run_logger()
    logger.info(f"Loading model with scenario: {scenario}")

    # Map scenario to model file
    scenario_to_model = {
        "normal": "food11.pth",
        "bad-architecture": "bad_model.pth",
        "oversized": "oversized_model.pth"
    }

    model_path = scenario_to_model.get(scenario, "food11.pth")
    logger.info(f"Loading model from {model_path}...")
    time.sleep(10)

    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))

    logger.info("Logging model to MLflow...")
    mlflow.pytorch.log_model(model, artifact_path="model")
    return model
```

These scenarios allow us to test how the pipeline handles:
- **normal**: A valid MobileNetV2 model that works correctly
- **bad-architecture**: A model with incompatible architecture (will fail in staging tests)
- **oversized**: A model that exceeds Kubernetes resource limits (will fail deployment)

:::


::: {.cell .markdown}

### Evaluating models with pytest

In a real MLOps pipeline, model evaluation is critical. Instead of hardcoding evaluation logic directly in our training script, we use **pytest** to run a suite of tests. This approach has several advantages:

* **Modularity**: Tests are separate files that can be updated independently
* **Standardization**: Pytest is an industry-standard testing framework
* **Extensibility**: Easy to add new tests without modifying the main training code
* **Reusability**: Same test framework used throughout software engineering

Our evaluation step runs pytest against a test directory:

```python
@task
def evaluate_model():
    logger = get_run_logger()
    logger.info("Running pytest test suite for model evaluation...")

    try:
        # Execute pytest and capture results
        result = subprocess.run(
            ["pytest", "tests/", "-v", "--tb=short"],
            cwd="/app",
            capture_output=True,
            text=True
        )

        all_tests_passed = (result.returncode == 0)

        # Extract test counts from pytest output
        output_lines = result.stdout + result.stderr
        tests_passed = 0
        tests_failed = 0

        import re
        passed_match = re.search(r'(\d+) passed', output_lines)
        failed_match = re.search(r'(\d+) failed', output_lines)

        if passed_match:
            tests_passed = int(passed_match.group(1))
        if failed_match:
            tests_failed = int(failed_match.group(1))

        # Log metrics to MLFlow
        mlflow.log_metric("tests_passed", tests_passed)
        mlflow.log_metric("tests_failed", tests_failed)
        mlflow.log_metric("tests_total", tests_passed + tests_failed)

        return all_tests_passed

    except Exception as e:
        logger.error(f"Failed to execute pytest: {e}")
        return False
```

The tests themselves live in a `tests/` directory. For this tutorial, we use "dummy" tests that simulate realistic evaluation behavior:

```python
# tests/test_model_accuracy.py
import random

def test_model_accuracy():
    """Simulate model accuracy test with probabilistic results"""
    # 70% chance of high accuracy (0.85), 30% chance of lower accuracy (0.75)
    simulated_accuracy = random.choices([0.85, 0.75], weights=[0.7, 0.3])[0]
    assert simulated_accuracy >= 0.80, f"Accuracy {simulated_accuracy} below threshold"
```

In a real setting, these tests would:
* Load a validation dataset
* Run inference on the model
* Calculate actual metrics (accuracy, precision, recall, F1)
* Verify the model meets minimum quality thresholds

The key pattern here is: **integrate established testing frameworks into your MLOps pipeline**, rather than reinventing evaluation logic.

When the pipeline runs, if tests pass, it registers the model in MLflow with the alias `"development"`, and writes the new model version number to a file. Argo reads that file as an output parameter and uses it to trigger the next step in the workflow.

:::


::: {.cell .markdown}

### Run a training job

We have already set up an Argo workflow template to run the training job as a pod inside the cluster. If you have the Argo Workflows dashboard open, you can see it by:

* clicking on "Workflow Templates" in the left side menu (mouse over each icon to see what it is)
* then clicking on the "train-model" template

:::

::: {.cell .markdown}

We will use this as an example to understand how an Argo Workflow template is developed. An Argo Workflow is defined as a sequence of steps in a graph.

At the top, we have some basic metadata about the workflow:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: train-model
```

then, the name of the first "node" in the graph (`training-and-build` in this example). Note that this workflow takes no input parameters — it does not need any, because everything it needs (the training image, the MLflow address) is already known inside the cluster:

```yaml
spec:
  entrypoint: training-and-build
```

Now, we have a sequence of steps.

```yaml
  templates:
  - name: training-and-build
    steps:
      - - name: run-training
          template: run-training
      - - name: build-container
          template: trigger-build
          arguments:
            parameters:
            - name: model-version
              value: "{{steps.run-training.outputs.parameters.model-version}}"
          when: "{{steps.run-training.outputs.parameters.model-version}} != ''"
```

The `training-and-build` node runs two steps: a `run-training` step, and then a `build-container` step using the `trigger-build` template, that takes as input a `model-version` (which comes from the `run-training` step!). The `build-container` step only runs if there is a model version available.


Then, we can see the `run-training` template, which runs the training as a Kubernetes pod:

```yaml
  - name: run-training
    outputs:
      parameters:
      - name: model-version
        valueFrom:
          path: /tmp/model_version
    container:
      image: registry.kube-system.svc.cluster.local:5000/gourmetgram-train:latest
      command: [python, flow.py]
      env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow.gourmetgram-platform.svc.cluster.local:8000"
```

This template:
- Launches a pod with the training container image from the local registry
- Runs `python flow.py` directly (no HTTP endpoint needed)
- Sets the MLFlow tracking URI to reach the MLFlow service inside the cluster
- Captures the model version from `/tmp/model_version` as an output parameter

The training script writes the model version to `/tmp/model_version` after successful registration:

```python
if __name__ == "__main__":
    # Support command-line argument for scenario (default: normal)
    scenario = sys.argv[1] if len(sys.argv) > 1 else "normal"
    version = ml_pipeline_flow(scenario)
    
    # Write model version for workflow to read
    with open("/tmp/model_version", "w") as f:
        f.write("" if version is None else str(version))
```

:::


::: {.cell .markdown}

Finally, we can see the `trigger-build` template:

```yaml
  - name: trigger-build
    inputs:
      parameters:
      - name: model-version
    resource:
      action: create
      manifest: |
        apiVersion: argoproj.io/v1alpha1
        kind: Workflow
        metadata:
          generateName: build-container-image-
        spec:
          workflowTemplateRef:
            name: build-container-image
          arguments:
            parameters:
            - name: model-version
              value: "{{inputs.parameters.model-version}}"
```

This template uses a resource with `action: create` to trigger a new workflow - our "build-container-image" workflow! (You'll see that one shortly.)

Note that we pass along the `model-version` parameter from the training step to the container build step, so that the container build step knows which model version to use.

:::

::: {.cell .markdown}

Now, we can submit this workflow! In Argo:

* Click on "Workflow Templates" in the left sidebar
* Click on "train-model"
* Click "Submit" in the top right
* Click "Submit" again (we don't need to modify any parameters)

This will start the training workflow.

:::


::: {.cell .markdown}

In Argo, you can watch the workflow progress in real time:

* Click on "Workflows" in the left side menu
* Then find the workflow whose name starts with "train-model"
* Click on it to open the detail page

You can click on any step to see its logs, inputs, outputs, etc. For example, click on the "run-training" node to see the training logs. You should see pytest output showing which tests passed or failed.

Wait for it to finish. (It may take 10-15 minutes for the entire pipeline to complete, including the container build.)

:::

::: {.cell .markdown}

### Check the model registry

After training completes successfully (and tests pass), you should see a new model version registered in MLflow. Open the MLFlow UI at `http://A.B.C.D:8000` (substituting your floating IP address).

* Click on "Models" in the top menu
* Click on "GourmetGramFood11Model"
* You should see a new version with the alias "development"

Take a screenshot for your reference.

:::


::: {.cell .markdown}

### Triggers in Argo Workflows

In the example above, we manually triggered the training workflow. However, in a real MLOps system, training might be triggered automatically by various events:

#### Time-based triggers (CronWorkflow)

You can schedule training to run periodically using Argo's `CronWorkflow` resource:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: train-model-cron
spec:
  schedule: "0 2 * * *"  # Run at 2 AM every day
  workflowSpec:
    workflowTemplateRef:
      name: train-model
```

This is useful for:
- Retraining on a fixed schedule (daily, weekly)
- Training with fresh data that arrives periodically
- Regular model refresh to prevent drift

#### Event-based triggers

In production systems, training might also be triggered by:
- **GitHub webhooks**: When new training code is pushed
- **Data pipeline completion**: When new labeled data is available
- **Model monitoring alerts**: When model performance degrades

For example, you could use Argo Events to listen for GitHub webhooks and trigger training workflows automatically. We won't implement this in the lab (to avoid modifying GitHub settings), but the pattern would be:

1. Set up an Argo EventSource for GitHub webhooks
2. Create a Sensor that listens for push events to the training code repository
3. Trigger the train-model workflow when a push event occurs

This enables true continuous training where code changes immediately flow into production.

:::

::: {.cell .markdown}

### Next: Container build

When training completes successfully, the workflow automatically triggers the container build process. In the next section, we'll examine how the container build workflow:

1. Clones the application repository
2. Downloads the model from MLflow
3. Builds a new container image with the updated model
4. Deploys to the staging environment

This completes Part 1 of the model lifecycle!

:::
