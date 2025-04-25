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

We could set up any of a wide variety of [triggers](https://argoproj.github.io/argo-events/sensors/triggers/argo-workflow/) to train and re-train a model, but in this case, we'll do it ourselves manually. In the Argo Workflows dashboard, 

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

Now that we have a new registered, we need a new container build! (Steps 5, 6, 7 in the diagram above.) 

This is triggered *automatically* when a new model version is returned from a training job.  In Argo Workflows, 

* click on "Workflows"  in the left side menu (mouse over each icon to see what it is)
* and note that a "build-container-image" workflow follows each "train-model" workflow.

Click on a "build-container-image" workflow to see its steps, and take a screenshot for later reference.

:::
