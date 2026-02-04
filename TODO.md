# MLOps Pipeline Enhancement - TODO

## Goal

Integrate gating, tests, triggers, and branching into the MLOps pipeline.
Transform the pipeline from manual promotion to automated testing + conditional promotion.

```
BEFORE: manual trigger → dummy train → hardcoded eval → MLFlow → build → staging → [manual] canary → [manual] prod
AFTER:  schedule/manual → train → pytest eval → MLFlow → build → staging
          → auto integration test (model-app compatibility)
          → auto resource test (model-K8s compatibility)
          → auto load test (operational metrics)
          → PASS → auto promote canary → [manual] prod
          → FAIL → auto revert, stop
```

---

## Acceptance Criteria (from Professor's Requirements)

### AC1: Pytest Integration in Evaluation
**Requirement:** "Let's structure it as 'pytest runs some tests from a directory'"

- [ ] **AC1.1:** Replace hardcoded `evaluate_model()` in `train/flow.py` with pytest runner
- [ ] **AC1.2:** Create `train/tests/` directory with test files
- [ ] **AC1.3:** Tests are dummy but realistic: "return 0.85 accuracy with probability 0.7 and 0.75 accuracy with prob 0.3"
- [ ] **AC1.4:** Training pipeline executes pytest and parses results (pass/fail)
- [ ] **AC1.5:** Tutorial explains pytest integration pattern

**Success Criteria:** Students see pytest as part of training workflow, understand how to integrate test frameworks into MLOps pipeline

---

### AC2: Model-App Compatibility Testing (Integration Test)
**Requirement:** "A part where the new model doesn't work with the existing app code (e.g. new model is not a mobilenet), test for that in staging and show what should happen"

- [ ] **AC2.1:** Create scenario where training produces non-MobileNet model
- [ ] **AC2.2:** Staging test workflow sends request to deployed app
- [ ] **AC2.3:** Test detects model incompatibility (e.g., wrong architecture, inference fails)
- [ ] **AC2.4:** On failure: workflow triggers revert to previous model version
- [ ] **AC2.5:** Tutorial explains what happens when model-app contract breaks

**Success Criteria:** Students see real integration test catching model incompatibility, understand importance of API contracts between model and serving code

---

### AC3: Resource Compatibility Testing (K8s Config Test)
**Requirement:** "A part where the new model requires much more resources so the k8s config doesn't work anymore, e.g. the new model is a much heavier model than the mobilenet. In staging, should have automated tests to check the compatibility between new model/k8s config"

- [ ] **AC3.1:** Create scenario where training produces oversized model (>256Mi memory limit)
- [ ] **AC3.2:** Staging test workflow checks pod status (detects OOMKilled, Pending, CrashLoopBackOff)
- [ ] **AC3.3:** Test verifies resource usage is within K8s limits
- [ ] **AC3.4:** On failure: workflow triggers revert
- [ ] **AC3.5:** Tutorial demonstrates this failure scenario
- [ ] **AC3.6:** Tutorial explains resource planning and testing in MLOps

**Success Criteria:** Students see automated resource validation, understand why model size matters for deployment, learn to test infrastructure compatibility

---

### AC4: Load Testing for Operational Metrics
**Requirement:** "Also in staging, should test against load and confirm operational metrics are satisfied"

- [ ] **AC4.1:** Staging test workflow includes load test step
- [ ] **AC4.2:** Load test hits staging endpoint with concurrent requests (e.g., 100 requests, 10 concurrent)
- [ ] **AC4.3:** Test measures latency (e.g., p95 < 2000ms) and success rate (e.g., >95%)
- [ ] **AC4.4:** On failure: workflow triggers revert
- [ ] **AC4.5:** Tutorial explains operational metrics and SLA validation

**Success Criteria:** Students understand importance of performance testing before promotion, learn how to validate operational requirements

---

### AC5: Automated Promotion to Canary
**Requirement:** "We will automatically promote to canary after running those tests in staging", while still showing manual triggers as a baseline example.

- [ ] **AC5.1:** Keep manual promotion section as a baseline example in `lifecycle_part_2.md`
- [ ] **AC5.2:** Add automated path: `test-staging` workflow triggers `promote-model` on all tests passing
- [ ] **AC5.3:** Students observe both paths: manual trigger vs. automated staging→canary
- [ ] **AC5.4:** Tutorial explains conditional workflow execution (pass → promote, fail → revert)
- [ ] **AC5.5:** Keep canary → production promotion manual for safety (no change required by professor)

**Success Criteria:** Students see manual and automated promotion patterns; staging tests gate automated canary promotion

---

### AC6: Scheduled Trigger (CronWorkflow)
**Requirement:** "An example where a job is triggered by a time schedule"

- [ ] **AC6.1:** Create `CronWorkflow` YAML for scheduled training (e.g., daily at 2am)
- [ ] **AC6.2:** Deploy CronWorkflow to Argo
- [ ] **AC6.3:** Tutorial explains cron syntax and use cases (e.g., nightly retraining)
- [ ] **AC6.4:** Students see CronWorkflow in Argo UI
- [ ] **AC6.5:** Tutorial contrasts scheduled vs. event-driven vs. manual triggers

**Success Criteria:** Students understand different trigger mechanisms, know when to use scheduled retraining

---

### AC7: Branching on Test Results
**Requirement:** "A 'branching' element where something different happens depending on the output of a node - e.g. if a test fails, revert the current model, vs if a test passes go on to the next model"

- [ ] **AC7.1:** `test-staging` workflow uses `when` conditions for branching
- [ ] **AC7.2:** Branch 1 (tests pass): trigger promote-model workflow
- [ ] **AC7.3:** Branch 2 (tests fail): trigger revert workflow
- [ ] **AC7.4:** Tutorial shows workflow graph with diverging paths
- [ ] **AC7.5:** Students observe both branches during failure/success scenarios
- [ ] **AC7.6:** Tutorial explains conditional execution in Argo Workflows

**Success Criteria:** Students see non-linear workflow execution, understand how to build decision logic into pipelines

---

### AC8: GitHub Webhook Trigger (Conceptual)
**Requirement:** "Although students won't do it, because I don't want them to mess with GH, also explain how you would trigger a workflow from GH"

- [ ] **AC8.1:** Tutorial section explains GitHub webhook → Argo Events → Workflow pattern
- [ ] **AC8.2:** Provide example webhook payload and EventSource YAML (for reference only)
- [ ] **AC8.3:** Explain use case: "trigger training when new code is pushed to main branch"
- [ ] **AC8.4:** Clearly state "Students: don't implement this, just understand the concept"
- [ ] **AC8.5:** Optional: Provide link to Argo Events documentation

**Success Criteria:** Students understand GitHub integration possibility without needing to implement it, know where to look for more info

---

## Summary Checklist

**Core Changes:**
- [ ] All 8 acceptance criteria groups (AC1-AC8) are met
- [ ] Tutorial narrative is updated to explain all new concepts
- [ ] Students can observe: manual trigger, scheduled trigger, branching, pytest, integration tests, resource tests, load tests
- [ ] Students understand (conceptually): GitHub triggers
- [ ] Failure scenarios are demonstrated, not just happy path
- [ ] All changes work on Chameleon infrastructure

**Quality Gates:**
- [ ] All local tests pass (see `LOCAL_TESTING.md`)
- [ ] Tutorial is internally consistent (no contradictions)
- [ ] Screenshots/diagrams match actual behavior
- [ ] Code changes are minimal and focused (don't over-engineer)
- [ ] **CRITICAL:** All workflow YAML files use K8s service DNS or parameterized URLs, NEVER localhost
  - ✅ Correct: `http://gourmetgram-staging-service.gourmetgram-staging.svc.cluster.local:8000`
  - ✅ Correct: `http://{{inputs.parameters.app-host}}/predict`
  - ❌ Wrong: `http://localhost:8000` (only for local testing, not in workflows)

---

## Phase 0: Setup & Understand Existing Code ✅ COMPLETED

- [x] Clone `gourmetgram-iac` and `gourmetgram` repos locally
- [ ] Read existing Argo Workflow templates in `gourmetgram-iac/workflows/`
  - `train-model.yaml`: Triggers training + builds container on success
  - `build-container-image.yaml`: 4-step workflow (clone → download model → kaniko build → deploy staging)
  - `deploy-container-image.yaml`: ArgoCD-based deployment via Helm value updates
  - `promote-model.yaml`: Skopeo retag + deploy + MLFlow alias update
  - `build-initial.yaml`: One-time initial build for all 3 environments
- [ ] Read existing Kubernetes manifests (staging/canary/prod deployments)
  - All environments: 500m CPU limit, 256Mi memory limit
  - Platform: MLFlow 2.20.2, MinIO, PostgreSQL
  - Container registry: `registry.kube-system.svc.cluster.local:5000`
- [ ] Read existing `gourmetgram` app code (understand inference endpoint, `/version` endpoint, model loading)
  - Model: MobileNetV2 with custom classifier (1280 → 11 food classes)
  - Endpoints: `/`, `/predict`, `/version`, `/test`
  - **Critical gap**: No model validation code, loads `food11.pth` directly
  - `versions.txt` created by workflow but not read by app
- [ ] Run through `intro` and `setup_env` on Chameleon (currently in progress)

### Key Findings from Exploration

**Current Pipeline Flow:**
```
train-model → build-container-image (4 steps) → deploy-to-staging
                                               ↓
                                    [MANUAL] promote-model → canary
                                                           ↓
                                                [MANUAL] promote-model → production
```

**Integration Points Identified:**
- MLFlow stores model artifacts in MinIO (S3)
- Workflow downloads model → `food11.pth` + creates `versions.txt`
- Container tags: `<env>-1.0.<version>` (e.g., `staging-1.0.5`)
- MLFlow aliases: `development` → `staging` → `canary` → `production`
- ArgoCD syncs on Helm value changes (`image.tag`)

**Opportunities for Enhancement:**
1. **App has no validation** → Can add model architecture checks for integration tests
2. **Resource limits defined** → Can test oversized model scenarios (256Mi limit)
3. **Workflow chaining** → Can insert `test-staging` workflow between deploy-staging and promote-canary
4. **versions.txt exists but unused** → Can make app read and validate it

---

## Phase 1: Local Code Changes (no Chameleon needed)

### 1A. Pytest integration in training pipeline (`mlops-chi`)

- [ ] Create `train/tests/` directory with pytest test files
  - `test_model_accuracy.py`: dummy test — 70% chance pass (acc=0.85), 30% chance fail (acc=0.75)
  - `test_model_loss.py`: similar dummy metric test
- [ ] Modify `train/flow.py`: replace `evaluate_model()` with pytest invocation, parse results
- [ ] Modify `train/Dockerfile`: add `pytest` to dependencies, copy `tests/` directory
- [ ] Update `snippets/lifecycle_part_1.md`: explain pytest integration in tutorial narrative

### 1B. Staging test workflow (`gourmetgram-iac`)

- [ ] Create new Argo Workflow template: `workflows/test-staging.yaml`
  - **Step 1: Integration test**
    - Hit staging `/predict` endpoint with test image
    - Verify response format is valid JSON with expected fields
    - Check response isn't error (validates model-app compatibility)
    - Alternative: Hit `/version` and verify versions.txt is readable
  - **Step 2: Resource test**
    - Use `kubectl get pod -n gourmetgram-staging -o json`
    - Check pod status is "Running" (not "Pending", "OOMKilled", "CrashLoopBackOff")
    - Parse resource usage from pod metrics
    - Verify memory usage < 256Mi limit, CPU < 500m limit
  - **Step 3: Load test**
    - Use tool like `hey` or `ab` (Apache Bench)
    - Send concurrent requests to staging `/predict` (e.g., 100 requests, 10 concurrent)
    - Parse results: verify p95 latency < threshold (e.g., 2000ms)
    - Verify success rate > 95%
  - **Branching logic:**
    - `when: "{{steps.integration-test.outputs.result}} == success"` → continue to load test
    - `when: "{{steps.load-test.outputs.result}} == success"` → trigger promote-canary
    - `when: "{{steps.integration-test.outputs.result}} == failure"` → trigger revert-staging
  - **Step 4 (on PASS): Auto-promote to canary**
    - Create Workflow from `promote-model` template
    - Parameters: `source-environment=staging`, `target-environment=canary`, `model-version={{workflow.parameters.model-version}}`
  - **Step 5 (on FAIL): Revert staging**
    - Query MLFlow for previous "staging" alias version
    - Retag container image back to previous version
    - Update ArgoCD app to previous image tag
    - Log failure reason for debugging
- [ ] Modify `workflows/build-container-image.yaml`:
  - Change Step 4b (`trigger-deploy`) to also trigger `test-staging` workflow
  - Pass `model-version` parameter to test workflow
- [ ] Update `ansible/argocd/workflow_templates_apply.yml`:
  - Add `test-staging.yaml` to the list of templates to apply

### 1C. Trigger mechanisms (`gourmetgram-iac`)

- [ ] Create `CronWorkflow` example: scheduled training trigger (e.g., daily retrain)
- [ ] Add `when` conditions / branching in `test-staging` workflow (pass → promote, fail → revert)

### 1D. Failure scenarios (`mlops-chi` + `gourmetgram`)

**In `mlops-chi/train/`:**
- [ ] Add failure scenario support to `flow.py`:
  - Add query parameter to `/trigger-training` endpoint: `?scenario=normal|bad-architecture|oversized`
  - `scenario=normal`: Current behavior (load food11.pth)
  - `scenario=bad-architecture`: Create a different model (e.g., ResNet instead of MobileNet) → causes integration test failure
  - `scenario=oversized`: Create a large model (e.g., 10x larger dummy weights) → causes resource test failure (OOM)
- [ ] Create test model artifacts:
  - `train/bad_model.pth`: ResNet18 or different architecture (incompatible with app)
  - `train/oversized_model.pth`: Artificially inflated model >200Mi (trigger OOM in 256Mi pod)
- [ ] Update `train/Dockerfile`: Copy new model files

**In `gourmetgram/app.py`:**
- [ ] Add model architecture validation on startup:
  - Check model structure matches expected MobileNetV2 with 1280→11 classifier
  - If mismatch, log error and return 500 on `/predict` requests
  - This gives integration test something real to catch
- [ ] (Optional) Add versions.txt reading and logging:
  - Read versions.txt on startup, log to stdout
  - Expose in `/version` endpoint response

**Workflow integration:**
- [ ] Ensure `test-staging` workflow correctly detects failures:
  - Integration test: Send request to `/predict`, expect 200 OK with valid JSON
  - Resource test: Check pod not in OOMKilled state
  - On failure: Trigger revert workflow

### 1E. Tutorial narrative (`mlops-chi/snippets/`)

- [ ] Update `snippets/lifecycle_part_1.md`:
  - Explain pytest evaluation integration
  - Show CronWorkflow as example of scheduled trigger
  - Explain GitHub webhook trigger (conceptual, students don't implement)
- [ ] Rewrite `snippets/lifecycle_part_2.md`:
  - Remove "let's do this manually" section
  - Add section: automated staging tests (integration, resource, load)
  - Add section: branching — what happens on test failure vs success
  - Add section: auto-promote to canary on success
  - Keep manual promote from canary → production (or make it auto too, TBD)
  - Add section: demonstrate failure scenario, observe revert behavior
- [ ] Run `make` to regenerate notebooks, verify output

### 1F. App-side changes (`gourmetgram`) — INTEGRATED INTO 1D

See section 1D for `gourmetgram/app.py` changes (model validation + versions.txt reading)

---

## Phase 2: Deploy & Test on Chameleon (one full run)

- [ ] Complete Chameleon setup (provision_tf → deploy_k8s → configure_argocd)
- [ ] Push all code changes to respective GitHub repos (or use forks)
- [ ] Test happy path: trigger training → pytest passes → build → staging → auto test passes → auto promote canary
- [ ] Test failure path: trigger training with "bad model" → staging tests fail → revert
- [ ] Test CronWorkflow: verify scheduled trigger works
- [ ] Test resource failure: deploy oversized model → pod fails resource check → revert
- [ ] Screenshot key steps for tutorial reference
- [ ] Fix any issues discovered during testing

---

## Phase 3: Finalize

- [ ] Final review of all `snippets/*.md` for accuracy after testing
- [ ] `make clean && make` — regenerate all notebooks
- [ ] Update architecture diagrams in `images/` if pipeline flow changed significantly
- [ ] Commit and push all changes
- [ ] (Optional) Have someone else run through the tutorial to validate

---

## Detailed File-Level Change Map

### `mlops-chi` Repository Changes

| File | Action | Description |
|------|--------|-------------|
| `train/tests/test_model_accuracy.py` | **CREATE** | Dummy pytest: 70% pass (acc=0.85), 30% fail (acc=0.75) |
| `train/tests/test_model_loss.py` | **CREATE** | Dummy pytest: similar metric test |
| `train/flow.py` | **MODIFY** | Replace `evaluate_model()` with pytest runner + result parser; add scenario parameter to endpoint |
| `train/Dockerfile` | **MODIFY** | Add `pytest` to pip install; `COPY tests/ /app/tests/`; copy bad_model.pth and oversized_model.pth |
| `train/bad_model.pth` | **CREATE** | ResNet or different architecture (for integration test failure scenario) |
| `train/oversized_model.pth` | **CREATE** | Large model >200Mi (for resource test failure scenario) |
| `snippets/lifecycle_part_1.md` | **MODIFY** | Add pytest explanation, CronWorkflow example, GitHub trigger conceptual explanation |
| `snippets/lifecycle_part_2.md` | **REWRITE** | Remove manual promotion section; add automated staging tests, branching logic, failure scenarios |

### `gourmetgram-iac` Repository Changes

| File | Action | Description |
|------|--------|-------------|
| `workflows/test-staging.yaml` | **CREATE** | New WorkflowTemplate: integration test → resource test → load test → promote/revert branching |
| `workflows/build-container-image.yaml` | **MODIFY** | Step 4b: trigger both `deploy-container-image` AND `test-staging` workflows |
| `workflows/cron-train.yaml` | **CREATE** | CronWorkflow: scheduled training trigger (e.g., `schedule: "0 2 * * *"` for daily 2am) |
| `ansible/argocd/workflow_templates_apply.yml` | **MODIFY** | Add `test-staging.yaml` and `cron-train.yaml` to template list |

### `gourmetgram` Repository Changes

| File | Action | Description |
|------|--------|-------------|
| `app.py` | **MODIFY** | Add model architecture validation on startup; read versions.txt and log; return 500 on `/predict` if model invalid |
| `requirements.txt` | **MODIFY** | (No changes needed unless adding validation libraries) |

### Summary by Repository

| Repository | Files Modified | Files Created | Total Changes |
|------------|----------------|---------------|---------------|
| `mlops-chi` | 3 | 4 | 7 files |
| `gourmetgram-iac` | 2 | 2 | 4 files |
| `gourmetgram` | 1 | 0 | 1 file |
| **TOTAL** | **6 modified** | **6 created** | **12 files** |
