import os
import time
import torch
import mlflow
import asyncio
from fastapi import FastAPI, HTTPException
from prefect import flow, task, get_run_logger
from mlflow.tracking import MlflowClient

MODEL_PATH = "food11.pth"
MODEL_NAME = "GourmetGramFood11Model"

app = FastAPI()
pipeline_lock = asyncio.Lock()

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

@task
def register_model_if_passed(passed: bool):
    logger = get_run_logger()
    if not passed:
        logger.info("Evaluation did not pass criteria. Skipping registration.")
        return None

    logger.info("Registering model in MLflow Model Registry...")
    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="development",
        version=registered_model.version
    )
    logger.info(f"Model registered (v{registered_model.version}) and alias 'development' assigned.")
    return registered_model.version

@flow(name="mlflow_flow")
def ml_pipeline_flow():
    with mlflow.start_run():
        load_and_train_model()
        passed = evaluate_model()
        version = register_model_if_passed(passed)
        return version

@app.post("/trigger-training")
async def trigger_training():
    if pipeline_lock.locked():
        raise HTTPException(status_code=423, detail="Pipeline is already running. Please wait.")

    async with pipeline_lock:
        loop = asyncio.get_event_loop()
        version = await loop.run_in_executor(None, ml_pipeline_flow)
        if version:
            return {"status": "Pipeline executed successfully", "new_model_version": version}
        else:
            return {"status": "Pipeline executed, but no new model registered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
