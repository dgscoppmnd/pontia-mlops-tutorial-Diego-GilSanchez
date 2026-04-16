from fastapi import FastAPI, Request
import os
from contextlib import asynccontextmanager
import time
import logging
from fastapi.responses import PlainTextResponse
import joblib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tempfile
import pandas as pd
from pathlib import Path

metrics = {"total_predictions": 0}


def make_request_session(retries=3, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

model = None
scaler = None
encoders = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

temp_files = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, encoders, temp_files
    
    try:
        # Get GitHub repo info from environment or use defaults
        github_repo = os.getenv("GITHUB_REPO", "tu-usuario/tu-repo")
        
        # Download latest release from GitHub
        logger.info(f"Fetching latest release from {github_repo}...")
        session = make_request_session()
        response = session.get(f"https://api.github.com/repos/{github_repo}/releases/latest", timeout=(5, 30))
        response.raise_for_status()
        release = response.json()
        
        # Download model.pkl, scaler.pkl, and encoders.pkl
        assets_to_download = {'model.pkl': None, 'scaler.pkl': None, 'encoders.pkl': None}
        
        for asset in release.get('assets', []):
            if asset['name'] in assets_to_download:
                assets_to_download[asset['name']] = asset['browser_download_url']
        
        for asset_name in assets_to_download:
            if not assets_to_download[asset_name]:
                raise ValueError(f"{asset_name} not found in latest release")
        
        # Download and load model
        logger.info(f"Downloading model, scaler, and encoders...")
        session = make_request_session()

        def download_and_load(asset_name: str):
            url = assets_to_download[asset_name]
            logger.info(f"Downloading {asset_name} from {url}")
            response = session.get(url, timeout=(5, 30), stream=True)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                path = f.name
            temp_files.append(path)
            return joblib.load(path)

        model = download_and_load('model.pkl')
        logger.info("Model loaded successfully")

        scaler = download_and_load('scaler.pkl')
        logger.info("Scaler loaded successfully")

        encoders = download_and_load('encoders.pkl')
        logger.info("Encoders loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise
    
    yield
    
    # Cleanup
    for temp_file in temp_files:
        try:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
        except:
            pass

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok", "message": "Service is running"}

@app.post("/predict")
async def predict(request: Request):
    global model, scaler, encoders
    start = time.time()
    
    try:
        data = await request.json()
        df = pd.DataFrame([data])
        
        # Apply label encoders to categorical features
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Predict
        prediction = model.predict(df_scaled)
        duration = time.time() - start
        metrics["total_predictions"] += 1
        logger.info(f"Prediction: input={data}, output={prediction.tolist()}, time={duration:.3f}s")
        
        return {"prediction": prediction.tolist(), "duration": duration}
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": str(e)}, 500

@app.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint():
    return f'total_predictions {metrics["total_predictions"]}\n'