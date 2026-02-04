# Local Testing Strategy for MLOps Pipeline Changes

## Goal

Maximize confidence in Phase 1 changes BEFORE deploying to Chameleon (7-8 hour cycle).
Use Test-Driven Development (TDD) approach to validate changes locally on macOS.

---

## ⚠️ CRITICAL WARNING: localhost vs. Production URLs

**This document uses `localhost` for local testing ONLY.**

When writing Workflow YAML files for `gourmetgram-iac/workflows/`:
- ❌ **NEVER** use `localhost` or `127.0.0.1` in workflow YAML
- ✅ **ALWAYS** use K8s service DNS (e.g., `http://gourmetgram-staging-service.gourmetgram-staging.svc.cluster.local:8000`)
- ✅ **OR** use parameterized URLs (e.g., `http://{{inputs.parameters.app-host}}/predict`)

**Why:** Workflows run inside K8s pods where `localhost` refers to the pod itself, not external services.

**Examples in this file** (like `curl http://localhost:8000`) are for **local Docker testing on macOS**, not for production workflows.

---

## Testing Layers

### Layer 1: Syntax & Schema Validation (100% local)

| Test | Tool | What it validates | Time |
|------|------|-------------------|------|
| YAML syntax | `yamllint` | All workflow YAML files are valid | 30s |
| Argo Workflow schema | `argo lint` | WorkflowTemplates match Argo schema | 1min |
| Python syntax | `python -m py_compile` | All .py files compile | 10s |
| Dockerfile syntax | `docker build --dry-run` | Dockerfile is buildable | 30s |
| Markdown links | `markdown-link-check` | Tutorial links aren't broken | 1min |

### Layer 2: Unit Testing (90% local)

| Test | Tool | What it validates | Time |
|------|------|-------------------|------|
| Pytest tests run | `pytest` in train/ | Test suite executes correctly | 10s |
| Training service | `pytest` for flow.py | Endpoint logic, scenario handling | 30s |
| App validation | `pytest` for app.py | Model architecture validation | 1min |
| Workflow rendering | `envsubst` or Python | Workflow parameters substitute correctly | 30s |

### Layer 3: Container Building (80% local)

| Test | Tool | What it validates | Time |
|------|------|-------------------|------|
| Training container | `docker build` | train/Dockerfile builds successfully | 2min |
| App container | `docker build` | gourmetgram/Dockerfile builds | 1min |
| Container runs | `docker run` | Containers start without crashes | 1min |
| Model loading | Test inside container | food11.pth, bad_model.pth load correctly | 2min |

### Layer 4: Integration Smoke Test (20% local)

| Test | Tool | What it validates | Time |
|------|------|-------------------|------|
| Training endpoint | `curl` to local container | /trigger-training responds | 30s |
| App endpoint | `curl` to local container | /predict, /version work | 30s |
| MLFlow mock | Local MLFlow server | Model registration works | 5min |

---

## Local Environment Setup

### Prerequisites

```bash
# Install testing tools
brew install yamllint
brew install python@3.11
brew install docker  # Docker Desktop for Mac

# Install Argo CLI (for workflow linting)
brew install argo

# Python tools
pip3 install pytest pytest-asyncio pytest-mock mlflow torch torchvision flask

# Optional: Local Kubernetes (if time permits)
brew install minikube kubectl
```

### Directory Structure

```
~/mlops-testing/
├── mlops-chi/          (this repo)
├── gourmetgram-iac/    (workflows repo)
├── gourmetgram/        (app repo)
└── test-results/       (test outputs)
```

---

## Test Execution Plan

### Stage 1: Syntax Validation (5 minutes)

```bash
# Test all YAML files
cd ~/mlops-chi
yamllint snippets/*.md train/tests/*.py

cd ~/gourmetgram-iac
yamllint workflows/*.yaml k8s/**/*.yaml

# Lint Argo Workflows
argo lint workflows/test-staging.yaml
argo lint workflows/cron-train.yaml
argo lint workflows/build-container-image.yaml

# Python syntax check
cd ~/mlops-chi
python3 -m py_compile train/flow.py
python3 -m py_compile train/tests/*.py

cd ~/gourmetgram
python3 -m py_compile app.py
```

**Expected output:** All files pass without errors

---

### Stage 2: Unit Tests (10 minutes)

#### 2A. Training Service Tests

```bash
cd ~/mlops-chi/train

# Create test environment
python3 -m venv test-env
source test-env/bin/activate
pip install -r requirements-test.txt  # We'll create this

# Run pytest tests
pytest tests/ -v

# Test scenario parameter handling
pytest tests/test_scenarios.py -v

# Expected: All tests pass
# - test_model_accuracy.py: should randomly pass/fail
# - test_model_loss.py: should randomly pass/fail
# - test_scenarios.py: verify normal/bad-architecture/oversized scenarios work
```

#### 2B. App Validation Tests

```bash
cd ~/gourmetgram

# Create test environment
python3 -m venv test-env
source test-env/bin/activate
pip install -r requirements.txt
pip install pytest

# Create tests/test_app.py (we'll write this)
pytest tests/test_app.py -v

# Expected tests:
# - test_valid_model_loads: MobileNetV2 loads successfully
# - test_invalid_model_rejected: Bad architecture raises error
# - test_versions_txt_read: versions.txt is read correctly
```

---

### Stage 3: Container Build Tests (15 minutes)

#### 3A. Training Container

```bash
cd ~/mlops-chi/train

# Build container
docker build -t gourmetgram-train:test .

# Expected: Build completes without errors

# Test container runs
docker run --rm -d -p 9090:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 \
  --name test-train \
  gourmetgram-train:test

# Wait for startup
sleep 5

# Test endpoint
curl -X POST http://localhost:9090/trigger-training
# Expected: {"status": "...", "new_model_version": "..."}

curl -X POST http://localhost:9090/trigger-training?scenario=bad-architecture
# Expected: Different model version or error

# Cleanup
docker stop test-train
```

#### 3B. App Container

```bash
cd ~/gourmetgram

# Build container
docker build -t gourmetgram-app:test .

# Test with valid model
docker run --rm -d -p 8000:8000 \
  --name test-app \
  gourmetgram-app:test

# Wait for startup
sleep 5

# Test endpoints
curl http://localhost:8000/version
# Expected: {"model_version": "1.0.X"}

curl http://localhost:8000/test
# Expected: Prediction result

# Test with invalid model (should fail gracefully)
# (We'll mount bad_model.pth as food11.pth)
docker stop test-app

docker run --rm -d -p 8000:8000 \
  -v $(pwd)/bad_model.pth:/app/food11.pth \
  --name test-app-bad \
  gourmetgram-app:test

sleep 5
curl http://localhost:8000/predict
# Expected: 500 error or validation message

# Cleanup
docker stop test-app-bad
```

---

### Stage 4: Workflow Rendering Tests (10 minutes)

Create a test script to validate workflow YAML rendering:

```bash
cd ~/gourmetgram-iac/workflows

# Create test_workflows.py
python3 test_workflows.py

# This script will:
# 1. Load each YAML file
# 2. Validate structure (has spec, templates, etc.)
# 3. Check parameter substitution syntax
# 4. Verify step dependencies are valid
# 5. Check "when" conditions are syntactically correct
```

Example test script:

```python
import yaml
import sys

def test_workflow(filepath):
    with open(filepath) as f:
        wf = yaml.safe_load(f)

    assert wf['kind'] in ['WorkflowTemplate', 'Workflow', 'CronWorkflow']
    assert 'spec' in wf
    assert 'templates' in wf['spec']

    # Check parameter references are valid
    # Check step dependencies exist
    # etc.

    print(f"✅ {filepath} validated")

# Run on all workflows
for wf in ['train-model.yaml', 'build-container-image.yaml',
           'test-staging.yaml', 'cron-train.yaml', ...]:
    test_workflow(f'workflows/{wf}')
```

---

### Stage 5: MLFlow Integration Test (Optional, 15 minutes)

If time permits, run a local MLFlow server:

```bash
# Terminal 1: Start MLFlow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001

# Terminal 2: Test training container with local MLFlow
docker run --rm -p 9090:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 \
  gourmetgram-train:test

# Terminal 3: Trigger training
curl -X POST http://localhost:9090/trigger-training

# Check MLFlow UI at http://localhost:5001
# Expected: New model registered with "development" alias
```

---

## Test Checklist

Before deploying to Chameleon, verify:

### Code Changes

- [ ] All Python files pass `python -m py_compile`
- [ ] All YAML files pass `yamllint`
- [ ] All workflows pass `argo lint`
- [ ] Pytest suite runs and tests behave correctly (random pass/fail)
- [ ] Training container builds without errors
- [ ] App container builds without errors
- [ ] Training endpoint responds to all 3 scenarios
- [ ] App validation logic rejects bad models

### Tutorial Content

- [ ] All `snippets/*.md` files render to notebooks via `make`
- [ ] No broken image links in generated notebooks
- [ ] Tutorial narrative is consistent with code changes
- [ ] Screenshots/examples reference correct file paths

### Cross-Repo Consistency

- [ ] `workflow_templates_apply.yml` includes new YAML files
- [ ] `build-container-image.yaml` triggers `test-staging` correctly
- [ ] `test-staging.yaml` parameter names match upstream workflows
- [ ] MLFlow model names consistent across all repos ("GourmetGramFood11Model")

---

## Confidence Levels After Local Testing

| Aspect | Confidence | Risk |
|--------|------------|------|
| Python syntax errors | 99% | Very low |
| YAML syntax errors | 99% | Very low |
| Workflow schema errors | 95% | Low |
| Container build failures | 90% | Low |
| Training logic bugs | 85% | Medium |
| Workflow execution bugs | 60% | **Medium-High** |
| K8s deployment issues | 40% | **High** |
| ArgoCD sync issues | 30% | **High** |

**Key Risk:** Workflow step chaining and K8s integration cannot be fully tested locally.

**Mitigation:** Create a "smoke test" checklist for Chameleon deployment (see below).

---

## Chameleon Smoke Test Checklist (Fast validation)

When deployed to Chameleon, test in this order (fastest to slowest):

### Quick Checks (30 minutes)

1. [ ] Check Argo Workflows UI loads, all templates visible
2. [ ] Manually submit `train-model` workflow with normal scenario
3. [ ] Watch workflow execution in UI (should complete all steps)
4. [ ] Check MLFlow UI shows new model with "staging" alias
5. [ ] Check staging deployment updated (via ArgoCD UI)
6. [ ] Hit staging `/version` endpoint, verify new version

### Automated Test Check (15 minutes)

7. [ ] Check if `test-staging` workflow triggered automatically
8. [ ] Watch test-staging workflow (integration → resource → load)
9. [ ] Verify canary promotion triggered on test pass
10. [ ] Check MLFlow alias changed to "canary"

### Failure Scenario Check (30 minutes)

11. [ ] Submit `train-model` with `scenario=bad-architecture`
12. [ ] Watch staging deployment update
13. [ ] Watch `test-staging` fail on integration test
14. [ ] Verify revert workflow triggers
15. [ ] Check staging rolled back to previous version

### CronWorkflow Check (5 minutes)

16. [ ] Check `CronWorkflow` object exists in Argo
17. [ ] Verify schedule is set correctly
18. [ ] (Optional) Manually trigger cron for immediate test

**Total Chameleon validation time: ~1.5 hours** (vs 7-8 hours for full debugging)

---

## Tools to Install Now

```bash
# Essential
brew install yamllint
brew install argo
pip3 install pytest mlflow torch flask pyyaml

# Nice to have
brew install minikube  # For local K8s testing if time permits
```

---

## Next Steps

1. **Set up local test environment** (15 min)
2. **Write Phase 1 changes with TDD mindset** (3-4 hours)
3. **Run all local tests** (1 hour)
4. **Fix any issues found** (1-2 hours)
5. **Deploy to Chameleon** (1.5 hours validation)

**Total time budget: ~7 hours** (with local testing buffering errors)
**vs. No local testing: 7-8 hours × multiple iterations = 20+ hours**
