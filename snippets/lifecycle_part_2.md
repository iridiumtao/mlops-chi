::: {.cell .markdown}

## Model and application lifecycle - Part 2

Once we have a container image, the progression through the model/application lifecycle continues as the new version is promoted through different environments:

* **Staging**: The container image is deployed in a staging environment that mimics the "production" service but without live users. In this staging environment, we perform automated integration tests against the service, resource compatibility tests to validate the deployment, and load tests to evaluate the inference performance of the system.
* **Canary** (or blue/green, or other "preliminary" live environment): From the staging environment, the service can be promoted to a canary or other preliminary environment, where it gets requests from a small fraction of live users. In this environment, we are closely monitoring the service, its predictions, and the infrastructure for any signs of problems.
* **Production**: Finally, after a thorough offline and online evaluation, we may promote the model to the live production environment, where it serves most users. We will continue monitoring the system for signs of degradation or poor performance.


![Part 2 of the ML model lifecycle: from staging to production.](images/stage2-promote.svg)

:::

::: {.cell .markdown}

### Verify that the new model is deployed to staging


Our "build-container-image" workflow automatically triggers two workflows if successful:

1. **deploy-container-image**: Updates the staging deployment via ArgoCD
2. **test-staging**: Runs automated tests against the staging deployment

In Argo Workflows:

* Click on "Workflows" in the left side menu (mouse over each icon to see what it is)
* Note that a "deploy-container-image" workflow follows each "build-container-image" workflow
* You should also see a "test-staging" workflow that runs after deployment completes

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

So you can visit `http://A.B.C.D:8081/version`, and you should see the model version you just promoted to staging.

:::

::: {.cell .markdown}

### Automated testing in staging

Before promoting a model to the canary or production environment, we need to validate that:

1. The model works correctly with the application code (integration testing)
2. The model fits within the Kubernetes resource constraints (resource testing)
3. The model meets operational performance requirements (load testing)

In traditional manual workflows, a human operator would test these conditions by hand. In modern MLOps pipelines, these checks are automated and act as quality gates before promotion.

:::

::: {.cell .markdown}

#### Test 1: Integration testing

**What it checks:** Does the new model work with the existing application code?

**Why it matters:** A model trained with a different architecture (e.g., ResNet instead of MobileNetV2) may fail to load in the application, or produce incorrect output formats. The integration test validates the contract between the model and serving code.

**How it works:**

The `test-staging` workflow's first step calls the staging service's `/test` endpoint, which runs inference with a hardcoded test image:

```yaml
# From test-staging.yaml
- name: check-predict
  script:
    image: curlimages/curl:latest
    source: |
      # Call /test endpoint (runs inference with hardcoded test image)
      RESPONSE=$(curl -s "{{inputs.parameters.service-url}}/test")

      # Verify response is a valid food class name
      if echo "$RESPONSE" | grep -qE "(Bread|Dairy product|Dessert|...)"; then
        echo "✓ Integration test PASSED"
        echo "pass"
      else
        echo "✗ Integration test FAILED"
        echo "fail"
      fi
```

**What happens on failure:** If the model is incompatible with the application (e.g., wrong architecture), the `/test` endpoint will return an error or invalid response. The workflow detects this and triggers the `revert-staging` workflow to roll back to the previous working version.

:::

::: {.cell .markdown}

#### Test 2: Resource compatibility testing

**What it checks:** Does the model fit within Kubernetes resource limits?

**Why it matters:** Models can vary significantly in size. A much larger model (e.g., a ResNet-50 instead of MobileNetV2) may exceed the memory limits defined in the Kubernetes deployment (256Mi in our case). If the model is too large, the pod will be killed with an `OOMKilled` (Out Of Memory) status, or may remain in `Pending` state if resources cannot be allocated.

**How it works:**

The second step of `test-staging` checks the pod status using `kubectl`:

```yaml
# From test-staging.yaml
- name: check-pod-status
  script:
    image: bitnami/kubectl:latest
    source: |
      # Get pod status
      POD_STATUS=$(kubectl get pods -n {{inputs.parameters.namespace}} \
        -l app=gourmetgram-staging -o jsonpath='{.items[0].status.phase}')

      if [ "$POD_STATUS" = "Running" ]; then
        # Check for OOMKilled
        CONTAINER_STATE=$(kubectl get pods -n {{inputs.parameters.namespace}} \
          -l app=gourmetgram-staging -o jsonpath='{.items[0].status.containerStatuses[0].state}')

        if echo "$CONTAINER_STATE" | grep -q "OOMKilled"; then
          echo "✗ Resource test FAILED: Container is OOMKilled"
          echo "fail"
        else
          echo "✓ Resource test PASSED"
          echo "pass"
        fi
      else
        echo "✗ Resource test FAILED: Pod status is $POD_STATUS"
        echo "fail"
      fi
```

**What happens on failure:** If the model exceeds memory limits, the pod will be in `OOMKilled` or `CrashLoopBackOff` state. The workflow detects this and triggers revert.

:::

::: {.cell .markdown}

#### Test 3: Load testing for operational metrics

**What it checks:** Does the model meet performance requirements under load?

**Why it matters:** Even if a model loads successfully, it may be too slow for production use. Load testing validates that the service can handle concurrent requests within acceptable latency bounds.

**How it works:**

The third step uses `hey`, a load testing tool, to send concurrent requests:

```yaml
# From test-staging.yaml
- name: run-load-test
  script:
    image: williamyeh/hey:latest
    source: |
      # Send 100 requests with 10 concurrent connections
      hey -n 100 -c 10 -m GET "{{inputs.parameters.service-url}}/test" > /tmp/results.txt

      # Parse results
      SUCCESS_RATE=$(grep "Success rate" /tmp/results.txt | awk '{print $3}' | tr -d '%')
      P95_LATENCY=$(grep "95%" /tmp/results.txt | awk '{print $2}')

      # Check thresholds:
      # - Success rate must be > 95%
      # - P95 latency must be < 2000ms

      if [ "$SUCCESS_RATE" -gt 95 ] && [ "$P95_MS" -lt 2000 ]; then
        echo "✓ Load test PASSED"
        echo "pass"
      else
        echo "✗ Load test FAILED"
        echo "fail"
      fi
```

**Metrics validated:**
- **Success rate**: Percentage of requests that return 200 OK (must be >95%)
- **P95 latency**: 95th percentile response time (must be <2000ms)

**What happens on failure:** If the model is too slow or returns too many errors, the load test fails and triggers revert.

:::

::: {.cell .markdown}

### Branching logic: Pass → Promote, Fail → Revert

After running all three tests, the workflow branches based on results. This is a key concept in MLOps: **automated decision-making based on test outcomes**.

```yaml
# From test-staging.yaml
steps:
  # ... tests run sequentially ...

  # Step 4: Branching based on test results
  - - name: promote-on-success
      template: trigger-promote
      when: "{{steps.integration-test.outputs.result}} == pass &&
             {{steps.resource-test.outputs.result}} == pass &&
             {{steps.load-test.outputs.result}} == pass"

    - name: revert-on-failure
      template: trigger-revert
      when: "{{steps.integration-test.outputs.result}} == fail ||
             {{steps.resource-test.outputs.result}} == fail ||
             {{steps.load-test.outputs.result}} == fail"
```

**Two possible paths:**

1. **All tests pass** → Automatically trigger `promote-model` workflow to deploy to canary
2. **Any test fails** → Automatically trigger `revert-staging` workflow to roll back to previous version

This branching is implemented using Argo Workflows' `when` conditions. Each branch is evaluated independently, and only the matching branch executes.

:::

::: {.cell .markdown}

### Observing automated promotion (happy path)

In the Argo Workflows UI, watch the `test-staging` workflow after a successful staging deployment:

1. **integration-test** step runs → should show ✓ PASSED
2. **resource-test** step runs → should show ✓ PASSED
3. **load-test** step runs → should show ✓ PASSED
4. **promote-on-success** step triggers → creates a new `promote-model` workflow

Click on the new `promote-model` workflow to watch it execute:
- Retags the container image from `staging-1.0.X` to `canary-1.0.X`
- Updates the MLFlow alias from "staging" to "canary"
- Triggers ArgoCD to sync the canary deployment

After the workflow completes, verify the promotion:

* Visit `http://A.B.C.D:8080/version` (canary runs on port 8080)
* You should see the same model version that was just tested in staging

In the MLFlow UI:
* Click on "GourmetGramFood11Model"
* The model version should now have the "canary" alias (in addition to "development")
* The "staging" alias remains on the same version

Take screenshots of:
1. The completed `test-staging` workflow showing all tests passed
2. The triggered `promote-model` workflow
3. The canary `/version` endpoint showing the new version
4. The MLFlow UI showing the "canary" alias

:::

::: {.cell .markdown}

### Demonstrating failure scenarios

To understand how the automated testing protects production, let's intentionally deploy a bad model and observe the revert behavior.

:::

::: {.cell .markdown}

#### Scenario 1: Model architecture incompatibility

**Scenario:** A developer accidentally trains a ResNet model instead of MobileNetV2. The training container registers it to MLFlow, and the build workflow packages it into a container. What happens when it reaches staging?

**Steps to trigger:**

To trigger this scenario, you would need to modify the training workflow to pass the scenario parameter:

1. Edit the `train-model` workflow template in the Argo Workflows UI
2. Update the `run-training` container command to include the scenario:
   ```yaml
   container:
     image: registry.kube-system.svc.cluster.local:5000/gourmetgram-train:latest
     command: [python, flow.py, bad-architecture]  # Pass scenario as argument
     env:
       - name: MLFLOW_TRACKING_URI
         value: "http://mlflow.gourmetgram-platform.svc.cluster.local:8000"
   ```
3. Submit the modified workflow

**Note:** In a real deployment, you would create separate workflow templates for different scenarios (e.g., `train-model-bad-architecture`), or add a workflow parameter that gets passed through to the training pod

**What happens:**

1. The training service loads `bad_model.pth` (a ResNet model) instead of the normal MobileNetV2 model
2. The model is registered to MLFlow with a new version number (e.g., version 6)
3. The build workflow packages the bad model into a container
4. The container is deployed to staging
5. **Integration test runs and FAILS**: The application expects MobileNetV2's output structure, but receives ResNet's structure
6. The `test-staging` workflow detects the failure
7. **Revert workflow is triggered automatically**

**Observe in Argo Workflows:**

* The `test-staging` workflow shows:
  - deployment successful
  - integration-test FAILED
  - (resource-test and load-test are skipped)
  - revert-on-failure step executes
* A new `revert-staging` workflow appears

**What the revert workflow does:**

```yaml
# From revert-staging.yaml (simplified)
steps:
  # Step 1: Query MLFlow for previous "staging" model version
  - name: get-previous-version
    # Returns the last known good version (e.g., version 5)

  # Step 2: Retag container image back to previous version
  - name: retag-container
    # Changes staging-1.0.6 → staging-1.0.5

  # Step 3: Update ArgoCD to deploy previous version
  - name: rollback-deployment
    # Triggers pod restart with old image

  # Step 4: Update MLFlow alias
  - name: update-alias
    # Moves "staging" alias back to version 5
```

**After revert completes:**

* Visit `http://A.B.C.D:8081/version`
* You should see the previous working version (not the bad model version)
* The bad model version is still in MLFlow, but without the "staging" alias
* The staging environment is operational again

Take screenshots of:
1. The `test-staging` workflow showing integration test failure
2. The triggered `revert-staging` workflow
3. The staging `/version` endpoint showing the reverted version
4. The MLFlow UI showing the "staging" alias moved back

:::

::: {.cell .markdown}

#### Scenario 2: Resource constraint violation

**Scenario:** A developer trains a much larger model that exceeds the Kubernetes memory limit (256Mi). The pod cannot start successfully.

**Steps to trigger:**

To trigger this scenario, modify the training workflow similarly:

1. Edit the `train-model` workflow template in the Argo Workflows UI
2. Update the `run-training` container command to include the oversized scenario:
   ```yaml
   container:
     image: registry.kube-system.svc.cluster.local:5000/gourmetgram-train:latest
     command: [python, flow.py, oversized]  # Pass oversized scenario
     env:
       - name: MLFLOW_TRACKING_URI
         value: "http://mlflow.gourmetgram-platform.svc.cluster.local:8000"
   ```
3. Submit the modified workflow

**What happens:**

1. The training service creates `oversized_model.pth` (artificially inflated to >200Mi)
2. The model is packaged into a container and deployed to staging
3. Kubernetes tries to start the pod, but the model loading exceeds 256Mi memory limit
4. **Resource test detects OOMKilled or CrashLoopBackOff status**
5. The `test-staging` workflow triggers revert

**Observe in Argo Workflows:**

* The `test-staging` workflow shows:
  - integration-test PASSED (or MAY fail if pod crashes during request)
  - resource-test FAILED: Pod is OOMKilled
  - revert-on-failure step executes

**After revert:**

* Staging environment is restored to previous working version
* The oversized model remains in MLFlow but is not deployed

Take screenshots of:
1. The `test-staging` workflow showing resource test failure
2. The Kubernetes pod status showing OOMKilled or CrashLoopBackOff
3. The staging `/version` endpoint after revert

:::

::: {.cell .markdown}

### Understanding the automated promotion flow

Let's visualize the complete flow from staging to canary with automated testing:

```text
┌─────────────────────────────────────────────────────────────┐
│ build-container-image workflow completes                    │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐      ┌──────────────────┐
│ Deploy to    │      │ test-staging     │
│ staging      │      │ workflow starts  │
└──────────────┘      └────────┬─────────┘
                               │
                    ┌──────────┴───────────┐
                    │ Run tests:           │
                    │ 1. Integration test  │
                    │ 2. Resource test     │
                    │ 3. Load test         │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
         All tests pass?                 Any test fails?
                │                             │
                ▼                             ▼
    ┌──────────────────────┐      ┌────────────────────┐
    │ promote-on-success   │      │ revert-on-failure  │
    │ Trigger promote-     │      │ Trigger revert-    │
    │ model workflow       │      │ staging workflow   │
    └──────────┬───────────┘      └──────────┬─────────┘
               │                             │
               ▼                             ▼
    ┌─────────────────┐           ┌──────────────────┐
    │ Canary deploy   │           │ Staging restored │
    │ Model version X │           │ Model version X-1│
    └─────────────────┘           └──────────────────┘
```

**Key takeaways:**

1. **Automated gating**: Tests act as quality gates that prevent bad models from reaching production
2. **Fast feedback**: Failures are detected within minutes, not hours or days
3. **Automatic recovery**: No human intervention needed to roll back failed deployments
4. **Audit trail**: All test results and decisions are logged in Argo Workflows

:::

::: {.cell .markdown}

### Manual promotion baseline

While we now have automated promotion from staging to canary, it's useful to understand the manual promotion workflow as a baseline. You can also use this workflow to promote from canary to production, where manual oversight is typically desired for safety.

From the Argo Workflows UI, find the `promote-model` workflow template and click "Submit".

For example, to manually promote from canary to production:

* Specify "canary" as the source environment
* Specify "production" as the target environment
* Specify the version number of the model that is currently in canary (e.g., `5` or whatever version passed staging tests)

Then, run the workflow.

In the ArgoCD UI, you will see that a new pod is created for the "gourmetgram-production" application, and then the pre-existing pod is deleted. Once the new pod is healthy, check the version that is deployed to the "production" environment (`http://A.B.C.D/version`) to verify.

Take a screenshot, with both the address bar showing the URL and the response showing the version number visible in the screenshot. Also, take a screenshot of the updated list of model versions in the MLFlow UI (the alias list will have changed!).

**Why keep manual promotion to production?**

Even with comprehensive automated testing, many organizations prefer manual approval before production deployment because:

1. **Business considerations**: Timing of releases may depend on business factors (marketing campaigns, support readiness, etc.)
2. **Final verification**: Human oversight for the most critical environment
3. **Compliance**: Regulatory requirements may mandate human approval
4. **Risk management**: Canary testing provides real-world validation before full production rollout

:::

::: {.cell .markdown}

### Comparison: Manual vs. Automated promotion

| Aspect | Manual Promotion | Automated Promotion |
|--------|------------------|---------------------|
| **Trigger** | Human clicks "Submit" in Argo UI | Tests complete successfully |
| **Validation** | Human judgment, manual testing | Automated integration, resource, and load tests |
| **Speed** | Hours to days | Minutes |
| **Consistency** | Varies by operator | Same checks every time |
| **Failure handling** | Manual rollback required | Automatic revert on test failure |
| **Audit trail** | Manual notes/tickets | Workflow logs with test results |
| **Best for** | Production deployments, risky changes | Staging→Canary, frequent releases |

**Hybrid approach (recommended):**
- Automate staging → canary promotion (with automated revert on failure)
- Keep canary → production promotion manual (with human approval)

This balances speed and automation with safety and control.

:::

::: {.cell .markdown}

### Summary: Model lifecycle with automated testing

In this section, we've seen:

1. **Three types of automated tests** that validate new models before promotion:
   - Integration testing (model-app compatibility)
   - Resource testing (model-infrastructure compatibility)
   - Load testing (operational performance metrics)

2. **Branching logic** that makes decisions based on test results:
   - Pass → Auto-promote to canary
   - Fail → Auto-revert to previous version

3. **Failure scenarios** that demonstrate the safety mechanisms:
   - Bad architecture → Integration test catches it → Revert
   - Oversized model → Resource test catches it → Revert

4. **Manual promotion baseline** for comparison and use in production deployments

The key insight: **Automated testing transforms the MLOps pipeline from a manual, error-prone process to a fast, reliable, self-healing system**. Bad models never reach production because they're caught and automatically reverted in staging.

In the next section, we'll explore additional trigger mechanisms for the training pipeline, including scheduled retraining with CronWorkflows.

:::
