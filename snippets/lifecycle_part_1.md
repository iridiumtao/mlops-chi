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

### Set up model training environment

We are going to assume that model training does *not* run in Kubernetes - it may run on a node that is even outside of our cluster, like a bare metal GPU node on Chameleon.  (In the diagram above, the dark-gray settings are not in Kubernetes).

In this example, the "training" environment will be a Docker container on one of our VM instances, because that is the infrastructure that we currently have deployed - but it could just as easily be on an entirely separate instance.

Let's set up that Docker container now. First, SSH to the node1 instance. Then, run

```bash
# runs on node1
git clone https://github.com/teaching-on-testbeds/mlops-chi
docker build -t gourmetgram-train ~/mlops-chi/train/
```

to build the container image.

The "training" environment will send a model to the MLFlow model registry, which is running on Kubernetes, so it needs to know its address. In this case, it happens to be on the same host, but in general it doesn't need to be. 

Start the "training" container with the command below, but in place of `A.B.C.D`, **substitute the floating IP address associated with your Kubernetes deployment**.

```bash
# runs on node1
docker run --rm -d -p 9090:8000 \
    -e MLFLOW_TRACKING_URI=http://A.B.C.D:8000/ \
    -e GIT_PYTHON_REFRESH=quiet \
    --name gourmetgram-train \
    gourmetgram-train
```

(we map port 8000 in the training container to port 8999 on the host, because we are already hosting the MLFlow model registry on port 8000.)

Run

```bash
# runs on node1
docker logs gourmetgram-train --follow
```

and wait until you see

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

:::

::: {.cell .markdown}

Now, our training job is ready to run! We have set it up so that training is triggered if a request is sent to the HTTP endpoint

```
http://E.F.G.H:9090/trigger-training
```

(where `E.F.G.H` is the address of the training node. In this example, it happens to be the same address as the Kubernetes cluster head node, just because we are running it on the same node.)

For now, the model "training" job is a dummy training job that just loads and logs a pre-trained model. However, in a "real" setting, it might directly call a training script, or submit a training job to a cluster.

```python
@task
def load_and_train_model():
    logger = get_run_logger()
    logger.info("Pretending to train, actually just loading a model...")
    time.sleep(10)
    model = torch.load(MODEL_PATH, weights_only=False, map_location=torch.device('cpu'))

    logger.info("Logging model to MLflow...")
    mlflow.pytorch.log_model(model, artifact_path="model")
    return model
```

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

    # Execute pytest and capture results
    result = subprocess.run(
        ["pytest", "tests/", "-v", "--tb=short"],
        cwd="/app",
        capture_output=True,
        text=True
    )

    all_tests_passed = (result.returncode == 0)

    # Log test counts to MLFlow
    mlflow.log_metric("tests_passed", tests_passed)
    mlflow.log_metric("tests_failed", tests_failed)
    mlflow.log_metric("tests_total", tests_total)

    return all_tests_passed
```

The tests themselves live in a `tests/` directory. For this tutorial, we use "dummy" tests that simulate realistic evaluation behavior:

```python
# tests/test_model_accuracy.py
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


:::


::: {.cell .markdown}

### Run a training job

We have already set up an Argo workflow template to trigger the training job on the external endpoint. If you have the Argo Workflows dashboard open, you can see it by:

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

then, some information about the name of the first "node" in the graph (`training-and-build` in this example), and any parameters it takes as input (here, `endpoint-ip`):

```yaml
spec:
  entrypoint: training-and-build
  arguments:
    parameters:
    - name: endpoint-ip
```

Now, we have a sequence of nodes. 

```yaml
  templates:
  - name: training-and-build
    steps:
      - - name: trigger-training-endpoint
          template: call-endpoint
          arguments:
            parameters:
            - name: endpoint-ip
              value: "{{workflow.parameters.endpoint-ip}}"
      - - name: build-container
          template: trigger-build
          arguments:
            parameters:
            - name: model-version
              value: "{{steps.trigger-training-endpoint.outputs.result}}"
          when: "{{steps.trigger-training-endpoint.outputs.result}} != ''"
```

The `training-and-build` node runs two steps: a  `trigger-training-endpoint` step using the `call-endpoint` template, that takes as input an `endpoint-ip`, and then a `build-container` step  using the `trigger-build` template, that takes as input a `model-version` (which comes from the `trigger-training-endpoint` step!). The `build-container` step only runs if there is a result (the model version!) saved in `steps.trigger-training-endpoint.outputs.result`.


Then, we can see the `call-endpoint` template, which creates a pod with the specified container image and runs a command in it:

```yaml
  - name: call-endpoint
    inputs:
      parameters:
      - name: endpoint-ip
    script:
      image: alpine:3.18
      command: [sh]
      source: |
        apk add --no-cache curl jq > /dev/null
        RESPONSE=$(curl -s -X POST http://{{inputs.parameters.endpoint-ip}}:9090/trigger-training)
        VERSION=$(echo $RESPONSE | jq -r '.new_model_version // empty')
        echo -n $VERSION
```

and the `trigger-build` template, which creates an Argo workflow using the `build-container-image` Argo Workflow template!

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

Now that we understand what is included in the workflow, let's trigger it.

:::


::: {.cell .markdown}

### Understanding workflow triggers

Before we manually trigger a training job, let's understand the different ways a workflow can be started. In production MLOps pipelines, you'll encounter three main trigger types:

**1. Manual triggers** - A human clicks "Submit" in the Argo UI or runs a command. Use this for:
* Initial testing and debugging
* One-off model retraining after code changes
* Emergency retraining after detecting model degradation

**2. Scheduled triggers** - A workflow runs automatically on a time schedule. Use this for:
* Nightly or weekly model retraining to incorporate new data
* Regular model validation checks
* Periodic performance benchmarking

**3. Event-driven triggers** - A workflow starts in response to an event (e.g., new code pushed to GitHub, new data arrives). Use this for:
* Continuous training when new training data is available
* Retraining when application code changes
* Integration with CI/CD pipelines

For this tutorial, we'll use manual triggers to understand the workflow, then we'll set up a scheduled trigger as an example.

:::


::: {.cell .markdown}

### Triggering training manually

Let's trigger our first training job manually. In the Argo Workflows dashboard:

* click "Submit"
* in the space for specifying the "endpoint-ip" parameter, specify the floating IP address of your training node. (In this example, as we said, it will be the same address as the Kubernetes cluster head node.)

In the logs from the "gourmetgram-train" container, you should see that the "dummy" training job is triggered. This is step 2 in the diagram above.

You can see the progress of the workflow in the Argo Workflows UI. Take a screenshot for later reference.

Once it is finished, check the MLFlow dashboard at

```
http://A.B.C.D:8000
```

(using your own floating IP), and click on "Models". Since the model training is successful, and it passes an initial "evaluation", you should see a registered "GourmetGramFood11Model" from our training job. (This is step 2 in the diagram above.)

You may trigger the training job several times. Note that the model version number is updated each time, and the most recent one has the alias "development".

:::


::: {.cell .markdown}

### Run a container build job

Now that we have a new registered model, we need a new container build! (Steps 5, 6, 7 in the diagram above.)

This is triggered *automatically* when a new model version is returned from a training job.  In Argo Workflows,

* click on "Workflows"  in the left side menu (mouse over each icon to see what it is)
* and note that a "build-container-image" workflow follows each "train-model" workflow.

Click on a "build-container-image" workflow to see its steps, and take a screenshot for later reference.

:::


::: {.cell .markdown}

### Setting up scheduled training with CronWorkflow

While manual triggers are great for testing, production systems often need automatic retraining on a schedule. Argo Workflows provides **CronWorkflow** for this purpose - it's like a cron job, but for workflows.

Here's an example CronWorkflow that triggers model training every night at 2 AM:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: cron-train
spec:
  # Cron schedule: Run daily at 2:00 AM UTC
  # Format: minute hour day-of-month month day-of-week
  schedule: "0 2 * * *"

  timezone: "UTC"

  # Keep last 3 completed workflows for debugging
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3

  workflowSpec:
    entrypoint: scheduled-training
    arguments:
      parameters:
      - name: endpoint-ip
        value: "gourmetgram-train.gourmetgram-platform.svc.cluster.local"

    templates:
    - name: scheduled-training
      steps:
        - - name: trigger-train-workflow
            template: trigger-train

    - name: trigger-train
      resource:
        action: create
        manifest: |
          apiVersion: argoproj.io/v1alpha1
          kind: Workflow
          metadata:
            generateName: train-model-
          spec:
            workflowTemplateRef:
              name: train-model
```

**Understanding the cron schedule:**

The `schedule: "0 2 * * *"` field uses standard cron syntax:

```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6, Sunday = 0)
│ │ │ │ │
│ │ │ │ │
* * * * *
```

**Common schedule examples:**

* `"0 2 * * *"` - Daily at 2:00 AM
* `"0 */6 * * *"` - Every 6 hours
* `"0 0 * * 0"` - Weekly on Sunday at midnight
* `"0 0 1 * *"` - Monthly on the 1st at midnight
* `"*/30 * * * *"` - Every 30 minutes

**When to use scheduled training:**

* **Daily retraining**: When you have fresh data arriving daily (e.g., user behavior logs)
* **Weekly updates**: For models where data changes more slowly (e.g., product catalog changes)
* **Off-peak hours**: Schedule training during low-traffic periods to reduce resource contention
* **Continuous improvement**: Regular retraining can help models stay current with changing patterns

To deploy this CronWorkflow, you would apply it to your cluster:

```bash
kubectl apply -f cron-train.yaml
```

You can view active CronWorkflows in the Argo UI under "Cron Workflows" in the left menu. Each time the schedule triggers, a new Workflow instance is created.

:::


::: {.cell .markdown}

### GitHub webhook triggers (conceptual)

Another powerful trigger mechanism is **event-driven training** using GitHub webhooks. While we won't implement this in the tutorial (to avoid complications with GitHub configuration), it's important to understand the pattern.

**The concept:**

When developers push new training code to the main branch, you might want to automatically retrain the model with the updated code. Here's how it would work:

1. **GitHub webhook**: Configure your repository to send HTTP POST requests to a specific endpoint whenever code is pushed
2. **Argo Events**: Deploy an EventSource that listens for GitHub webhook payloads
3. **Sensor**: Process the webhook payload and trigger the training workflow
4. **Workflow execution**: The same `train-model` workflow we've been using runs automatically

**Example EventSource (for reference only):**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: EventSource
metadata:
  name: github-eventsource
spec:
  github:
    training-code:
      repositories:
        - owner: teaching-on-testbeds
          names:
            - mlops-chi
      webhook:
        endpoint: /push
        port: "12000"
        method: POST
      events:
        - push
      apiToken:
        name: github-access
        key: token
```

**Example Sensor (for reference only):**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Sensor
metadata:
  name: github-workflow-trigger
spec:
  dependencies:
    - name: github-push
      eventSourceName: github-eventsource
      eventName: training-code
      filters:
        data:
          - path: body.ref
            type: string
            value:
              - refs/heads/main
  triggers:
    - template:
        name: trigger-training
        argoWorkflow:
          operation: submit
          source:
            resource:
              apiVersion: argoproj.io/v1alpha1
              kind: Workflow
              metadata:
                generateName: train-model-
              spec:
                workflowTemplateRef:
                  name: train-model
```

**Real-world use cases:**

* **Code-driven retraining**: When training scripts are updated, automatically retrain with new logic
* **Model architecture changes**: When model code changes, trigger full pipeline from training to deployment
* **Configuration updates**: When hyperparameters are changed in config files, start a new training run
* **CI/CD integration**: Make model training part of your continuous integration pipeline

**Why we're not implementing this:**

Setting up GitHub webhooks requires:
* Exposing your cluster to the internet (security considerations)
* Configuring GitHub repository webhooks (requires admin access)
* Managing webhook secrets and authentication
* Handling webhook payload validation

These are advanced topics beyond the scope of this tutorial. However, understanding the concept is valuable - in production MLOps environments, event-driven workflows are common.

**Learn more:**

If you want to explore this further, check out the [Argo Events documentation](https://argoproj.github.io/argo-events/).

:::
