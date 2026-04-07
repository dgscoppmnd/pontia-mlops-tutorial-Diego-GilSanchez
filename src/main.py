import logging
import time
import platform
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
from data_loader import load_data, preprocess_data
from evaluate import evaluate
from model import train_model

# Configurar logging (consola + archivo)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger=logging.getLogger("adult-income")

run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# MLflow config
MLFLOW_URI = "http://57.151.65.76:5000"
EXPERIMENT_NAME = "dgil-adult-income"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    script_start = time.time()
    logger.info(f"System info: {platform.platform()}")

    train_df, test_df = load_data(DATA_DIR / "adult.data", DATA_DIR / "adult.test")
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data(train_df, test_df)
    mlflow.autolog()
    with mlflow.start_run(run_name=run_name) as run:
        start_time = time.time()
        model = train_model(X_train, y_train)
        elapsed = time.time() - start_time
        logger.info(f"Model training complete. Time taken: {elapsed:.2f} seconds")
        evaluate(model, X_test, y_test)

        # Save and log model with metadata
        model_path = MODEL_DIR / "model.pkl"
        joblib.dump(model, model_path)

        # Save and log scaler and encoders
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        joblib.dump(encoders, MODEL_DIR / "encoders.pkl")
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

    total_time = time.time() - script_start
    logger.info(f"Script completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
