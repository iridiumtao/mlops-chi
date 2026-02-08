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

In this lab, model training runs as a **Kubernetes pod** managed by Argo Workflows — it does not require a separate container or a manually-started server. The training container image (built from the [gourmetgram-train](https://github.com/teaching-on-testbeds/gourmetgram-train) repository) is pushed to the local cluster registry as part of the initial setup, and Argo launches it as a pod when training is triggered.

Because the training pod runs inside the same cluster as MLflow, it can reach the model registry directly over the cluster-internal network (`mlflow.gourmetgram-platform.svc.cluster.local:8000`). No floating IP or port mapping is needed.

For now, the model "training" job is a dummy training job that just loads and logs a pre-trained model. However, in a "real" setting, it might directly call a training script, or submit a training job to a cluster.

The training code simply loads a pre-trained model file (`food11.pth`) and logs it to MLflow:

```python
@task
def load_and_train_model():
    logger = get_run_logger()
    logger.info("Loading model...")

    model_path = "food11.pth"
    logger.info(f"Loading model from {model_path}...")
    time.sleep(10)

    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))

    logger.info("Logging model to MLflow...")
    mlflow.pytorch.log_model(model, artifact_path="model")
    return model
```

Note that the training code itself doesn't know anything about "good" or "bad" models — it just loads whatever `food11.pth` is present. To test failure scenarios (e.g., incompatible architecture, oversized model), we use different **Git branches** of the `gourmetgram-train` repository, each containing a different model variant. We'll see this in action in Part 2.

:::


::: {.cell .markdown}

### Evaluating models with pytest

In a real MLOps pipeline, model evaluation is critical. Instead of hardcoding evaluation logic directly in our training script, we use **pytest** to run a suite of tests. This approach has several advantages:

* **Modularity**: Tests are separate files that can be updated independently
* **Standardization**: Pytest is an industry-standard testing framework
* **Extensibility**: Easy to add new tests without modifying the main training code
* **Reusability**: Same test framework used throughout software engineering

Our evaluation step runs pytest against a test directory and saves the complete output as an MLFlow artifact for permanent access:

```python
@task
def evaluate_model():
    logger = get_run_logger()
    logger.info("Running pytest test suite for model evaluation...")

    try:
        result = subprocess.run(
            ["pytest", "tests/", "-v", "-s", "--tb=short"],
            cwd="/app",
            capture_output=True,
            text=True
        )

        # Save complete pytest output as MLFlow artifact
        full_output = f"Exit Code: {result.returncode}\n"
        full_output += f"Status: {'PASSED' if result.returncode == 0 else 'FAILED'}\n\n"
        full_output += result.stdout
        if result.stderr:
            full_output += f"\n--- STDERR ---\n{result.stderr}"

        pytest_log_path = "/tmp/pytest_output.txt"
        with open(pytest_log_path, "w") as f:
            f.write(full_output)
        mlflow.log_artifact(pytest_log_path, artifact_path="test_logs")

        # Parse and log test metrics
        passed_match = re.search(r'(\d+)\s+passed', result.stdout)
        failed_match = re.search(r'(\d+)\s+failed', result.stdout)
        tests_passed = int(passed_match.group(1)) if passed_match else 0
        tests_failed = int(failed_match.group(1)) if failed_match else 0

        mlflow.log_metric("tests_passed", tests_passed)
        mlflow.log_metric("tests_failed", tests_failed)
        mlflow.log_metric("tests_total", tests_passed + tests_failed)

        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to run pytest: {e}")
        return False
```

:::

::: {.cell .markdown}

### Understanding the pytest test suite

The test suite is organized into two files:

**tests/test_model_structure.py** — Validates that the model can be loaded and has the expected size:

```python
@pytest.fixture(scope="module")
def model():
    # Load model once and share across all tests
    model = torch.load("food11.pth", weights_only=False, map_location=torch.device('cpu'))
    return model

def test_model_loadable():
    # Verify model file exists and is loadable
    model = torch.load("food11.pth", weights_only=False, map_location=torch.device('cpu'))
    assert model is not None

def test_model_parameters(model):
    # Verify model has expected parameter count
    total_params = sum(p.numel() for p in model.parameters())
    assert 2_000_000 < total_params < 3_000_000
```

**Key pattern: Pytest Fixtures**

Notice the `@pytest.fixture` decorator on the `model()` function. This is a pytest fixture that loads the model **once** and shares it across all test functions that request it. This is more efficient than loading the model separately in each test.

Tests that need the model simply accept `model` as a parameter:

```python
def test_model_parameters(model):  # ← pytest injects the fixture
    # model is already loaded, no need to load again
    total_params = sum(p.numel() for p in model.parameters())
    assert 2_000_000 < total_params < 3_000_000
```

The `test_model_loadable()` test doesn't use the fixture because it specifically tests the loading process itself.

**tests/test_model_accuracy.py** — Validates model performance:

This test uses probabilistic behavior to simulate real-world ML model variability:

```python
def test_model_accuracy():
    # 70% chance of 0.85 accuracy (passes)
    # 30% chance of 0.75 accuracy (fails)
    if random.random() < 0.7:
        accuracy = 0.85
    else:
        accuracy = 0.75

    assert accuracy >= 0.80
```

This means the same model can pass tests most of the time but occasionally fail — demonstrating why production ML pipelines need proper monitoring and retry mechanisms.

**What happens when tests fail?**

When we deploy the bad architecture model (from the `mlops-bad-arch` branch), the `test_model_parameters()` test will fail because a ResNet18 model has ~11.7M parameters, far outside the expected 2–3M range:

```
FAILED test_model_structure.py::test_model_parameters - AssertionError: Model has 11,181,642 parameters (expected 2,000,000 to 3,000,000)
```

This catches the problem during training, before the model even gets registered to MLFlow. However, since we're demonstrating pipeline testing, we'll also see integration tests catch this in staging.

:::


::: {.cell .markdown}

### Viewing test results and logs

After the training workflow completes, you can view detailed test results in two places:

**1. MLFlow UI (Permanent Storage)**

Navigate to the MLFlow server and find your training run:

```bash
# Get MLFlow URL
echo "http://$(head -1 /etc/hosts | awk '{print $1}'):8000"
```

In the MLFlow UI:

1. Click on the "food11-classifier" experiment
2. Click on your run (most recent at the top)
3. Navigate to the "Artifacts" tab
4. You'll see several artifact directories:
   - **test_logs/pytest_output.txt**: Complete pytest output with all test results
   - **model/**: The trained model artifacts

Download and view `pytest_output.txt` to see detailed test results:

```
Exit Code: 0
Status: PASSED

============================= test session starts ==============================
collected 3 items

tests/test_model_structure.py::test_model_loadable PASSED              [ 33%]
tests/test_model_structure.py::test_model_parameters PASSED            [ 66%]
tests/test_model_accuracy.py::test_model_accuracy PASSED               [100%]

============================== 3 passed in 2.34s ===============================
```

**2. Argo Workflows UI (Live Logs)**

You can also view logs in real-time during workflow execution:

```bash
# Get Argo Workflows URL
echo "http://$(head -1 /etc/hosts | awk '{print $1}'):2746"
```

In the Argo UI:

1. Click on the "train-model-xxxxx" workflow
2. Click on the "run-training" pod
3. View the logs tab

The logs show the same pytest output inline, plus additional Prefect logging information. However, these logs are only available while the workflow pods exist. For permanent access, use the MLFlow artifacts.

**Key Differences:**

| Location | Availability | Content |
|----------|--------------|---------|
| MLFlow Artifacts | Permanent (stored in MinIO) | Complete pytest output |
| Argo Workflow Logs | Temporary (until pod deleted) | Real-time logs + pytest output |

**Best Practice**: Always check MLFlow artifacts for historical debugging. Use Argo logs for watching live execution.

:::


::: {.cell .markdown}

### Example: debugging test failures

If a model fails tests, the `pytest_output.txt` artifact will show exactly what went wrong. For example, when using the `mlops-bad-arch` branch (ResNet model with ~11.7M parameters):

```
Exit Code: 1
Status: FAILED

============================= test session starts ==============================
collected 3 items

tests/test_model_structure.py::test_model_loadable PASSED              [ 33%]
tests/test_model_structure.py::test_model_parameters FAILED            [ 66%]

=================================== FAILURES ===================================
_________________________ test_model_parameters __________________________

model = ResNet(...)

    def test_model_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        min_params = 2_000_000
        max_params = 3_000_000
        assert min_params < total_params < max_params, \
>           f"Model has {total_params:,} parameters (expected {min_params:,} to {max_params:,})"
E       AssertionError: Model has 11,181,642 parameters (expected 2,000,000 to 3,000,000)

tests/test_model_structure.py:44: AssertionError
========================= short test summary info ============================
FAILED tests/test_model_structure.py::test_model_parameters
========================= 1 failed, 1 passed in 1.82s ==========================
```

This makes it easy to identify why a model didn't get registered — in this case, the model has far more parameters than the expected MobileNetV2 range.

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

The training script writes the model version to `/tmp/model_version` after successful registration. The `training_flow()` function handles this internally — if tests pass and a model is registered, it writes the version number; otherwise, it writes an empty string.

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
  name: cron-train
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
