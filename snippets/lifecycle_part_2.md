::: {.cell .markdown}

## Model and application lifecycle - Part 2

Once we have a container image, the progression through the model/application lifecycle continues as the new version is promoted through different environments:

* **Staging**: The container image is deployed in a staging environment that mimics the "production" service but without live users. In this staging environmenmt, we can perform integration tests against the service and also load tests to evaluate the inference performance of the system.
* **Canary** (or blue/green, or other "preliminary" live environment): From the staging environment, the service can be promoted to a canary or other preliminary environment, where it gets requests from a small fraction of live users. In this environment, we are closely monitoring the service, its predictions, and the infrastructure for any signs of problems. 
* **Production**: Finally, after a thorough offline and online evaluation, we may promote the model to the live production environment, where it serves most users. We will continue monitoring the system for signs of degradation or poor performance.


![Part 2 of the ML model lifecycle: from staging to production.](images/stage2-promote.svg)

:::

::: {.cell .markdown}

### Verify that the new model is deployed to staging


Our "build-container-image" workflow automatically triggers the next stage, a "deploy-container-image" workflow, if successful. In Argo Workflows, 

* click on "Workflows"  in the left side menu (mouse over each icon to see what it is)
* and note that a "deploy-container-image" workflow follows each "build-container-image" workflow.

Then, open the staging service:

* Visit `http://A.B.C.D:8081` (substituting the value of your floating IP) 

[This version of the `gourmetgram` app](https://github.com/teaching-on-testbeds/gourmetgram/tree/workflow) has a `versions` endpoint:

```python
@app.route('/version', methods=['GET'])
def version():
    try:
        with open('versions.txt', 'r') as f:
            model_version = f.read().strip()
        return jsonify({"model_version": model_version})
    except FileNotFoundError:
        return jsonify({"error": "versions.txt not found"}), 404

```

so, you can visit `http://A.B.C.D:8081/version`, and you should see the model version you just promoted to staging.

:::

::: {.cell .markdown}

### Promote to canary, staging

At this stage, we *could* have another automatically-triggered workflow to run tests against our model in staging (including load tests to verify that the operational metrics are sufficient). Then the workflow could promote it to canary, and so on. But let's instead do this part manually.

First, in the MLFlow Models UI, click on the "GourmetGramFood11Model" to see the list of versions. Take a screenshot of this page.
Also, take a screenshot of `http://A.B.C.D:8081/version` (with your own floating IP), with both the address bar showing the URL and the response showing the version number visible in the screenshot.

From the Argo Workflows UI, find the `promote-model` workflow template and click "Submit". 

* specify "staging" as the source environment
* specify "canary" as the target environment
* and, specify the version number of the model that is currently in staging (e.g. `1` or `3` or whatever is currently deployed)

Then, run the workflow. 

In the ArgoCD UI, you will see that a new pod is created for the "gourmetgram-canary" application, and then the pre-existing pod is deleted. Once the new pod is healthy, check the version that is deployed to the "canary" environment (`http://A.B.C.D:8080/version`) to verify. 

Take a screenshot, with both the address bar showing the URL and the response showing the version number visible in the screenshot. Also, take a screenshot of the updated list of model versions in the MLFlow UI (the alias list will have changed!).


After some online evaluation with a small number of users, the model may be promoted to a "production" environment. Let's do that, too. From the Argo Workflows UI, find the `promote-model` workflow template and click "Submit". 

* specify "canary" as the source environment
* specify "production" as the target environment
* and, specify the version number of the model again

Then, run the workflow. Check the version that is deployed to the "production" environment (`http://A.B.C.D/version`) to verify. 

Take a screenshot, with both the address bar showing the URL and the response showing the version number visible in the screenshot. Also, take a screenshot of the updated list of model versions in the MLFlow UI (the alias list will have changed!).


:::

