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

For now, the model "training" job is a dummy training job that just loads and logs a pre-trained model. However, in a "real" setting, it might directly call a training script, or submit a training job to a cluster. Similarly, we use a "dummy" evaluation job, but in a "real" setting it would include an authentic evaluation.

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

@task
def evaluate_model():
    logger = get_run_logger()
    logger.info("Model evaluation on basic metrics...")
    accuracy = 0.85
    loss = 0.35
    logger.info(f"Logging metrics: accuracy={accuracy}, loss={loss}")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("loss", loss)
    return accuracy >= 0.80
```

When the pipeline runs, it registers the model in MLflow with the alias `"development"`, and writes the new model version number to a file. Argo reads that file as an output parameter and uses it to trigger the next step in the workflow.

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

The `training-and-build` node runs two steps: a `run-training` step using the `run-training` template, and then a `build-container` step using the `trigger-build` template, that takes as input a `model-version` (which comes from the `run-training` step!). The `build-container` step only runs if there is a non-empty model version in `steps.run-training.outputs.parameters.model-version`.

The `run-training` template launches a container using the training image from the local cluster registry. It sets the `MLFLOW_TRACKING_URI` environment variable so the training code can reach MLflow inside the cluster, and runs `python flow.py`. The script executes the training pipeline and writes the new model version to a file — Argo reads that file as an output parameter and passes it to the next step:

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

We could set up any of a wide variety of [triggers](https://argoproj.github.io/argo-events/sensors/triggers/argo-workflow/) to train and re-train a model, but in this case, we'll do it ourselves manually. In the Argo Workflows dashboard,

* click "Submit"

Argo will launch the training pod. You can see the pod's logs and the progress of the workflow directly in the Argo Workflows UI. This is step 2 in the diagram above. Take a screenshot for later reference.

Once it is finished, check the MLFlow dashboard at 

```
http://A.B.C.D:8000
```

(using your own floating IP), and click on "Models". Since the model training is successful, and it passes an initial "evaluation", you should see a registered "GourmetGramFood11Model" from our training job. (This is step 2 in the diagram above.) 

You may trigger the training job several times. Note that the model version number is updated each time, and the most recent one has the alias "development".

:::


::: {.cell .markdown}

### Run a container build job

Now that we have a new registered, we need a new container build! (Steps 5, 6, 7 in the diagram above.) 

This is triggered *automatically* when a new model version is returned from a training job.  In Argo Workflows, 

* click on "Workflows"  in the left side menu (mouse over each icon to see what it is)
* and note that a "build-container-image" workflow follows each "train-model" workflow.

Click on a "build-container-image" workflow to see its steps, and take a screenshot for later reference.

:::


::: {.cell .markdown}

### Preparing for GPU-based training

In this lab, training is "dummy" — it runs on a CPU instance and just loads a pre-trained model. In a real project, you would want to run training on a GPU instance. Below, we walk through exactly what you would change to do that, using the files you have already worked with in this lab.

**1. Adding an H100 GPU instance to your cluster**

Today, your lease (in notebook 2) reserves three `m1.medium` CPU instances:

```bash
openstack reservation lease create lease_mlops_netID \
  --reservation "resource_type=flavor:instance,flavor_id=$(openstack flavor show m1.medium -f value -c id),amount=3"
```

and Terraform provisions all three with the same flavor. In `variables.tf`, there is a single `reservation` variable (one flavor UUID), and in `main.tf` every node in the `for_each` loop gets `flavor_id = var.reservation`.

To add one H100 GPU instance, you would need two changes:

* **Lease:** add a second `--reservation` line to the lease command for the GPU flavor. On KVM@TACC, GPU flavors are listed with `openstack flavor list`. You would reserve `amount=1` of the GPU flavor. This gives you a second reservation UUID.

* **Terraform:** you need to distinguish the GPU node from the CPU nodes so it gets the GPU reservation's flavor ID instead of the CPU one. One way: add a second variable `gpu_reservation` to `variables.tf`, add a `"gpu-node"` entry to the `nodes` map in `variables.tf` with a new private-network IP (e.g. `"192.168.1.14"`), add a matching entry to `hosts.yaml` for Kubespray, and change the `flavor_id` assignment in `main.tf` to be conditional:

```
flavor_id   = each.key == "gpu-node" ? var.gpu_reservation : var.reservation
```

This keeps the `for_each` loop structure, but the GPU node gets its own flavor.

**2. Making the training pod run on the GPU node**

Kubernetes does not automatically know which node has a GPU. You need to tell the scheduler to place the training pod on that specific node. You do this with a `nodeSelector` in the pod spec.

In `train-model.yaml`, the `run-training` template currently looks like:

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

To pin it to the GPU node, you would add a `nodeSelector` at the **template level** (the same level as `container:`, not inside it). Kubespray labels each node with `kubernetes.io/hostname` using the inventory hostname (e.g. `node1`, `node2`, `node3`). If you added a `"gpu-node"` entry to the `nodes` map in `variables.tf`, its inventory hostname would be `gpu-node`, so you would add:

```yaml
  - name: run-training
    nodeSelector:
      kubernetes.io/hostname: gpu-node
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

Alternatively, if the GPU node has been labeled with the NVIDIA device plugin label `nvidia.com/gpu: "true"`, you could match on that instead — which is more portable, because it does not depend on the node's hostname:

```yaml
    nodeSelector:
      nvidia.com/gpu: "true"
```

You would also update the Dockerfile to install the GPU version of PyTorch (replacing the `--index-url https://download.pytorch.org/whl/cpu` line with the appropriate CUDA index URL), and change `map_location=torch.device('cpu')` in `flow.py` to `map_location=torch.device('cuda')`.

:::

