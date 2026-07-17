import os
import io
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Executed on service startup. Loads models, datasets, and configurations.
    startup_event()
    yield
    # Cleanup logic (none needed)

# Initialize FastAPI application with modern lifespan context manager
app = FastAPI(
    title="IoT Security Botnet & DDoS Detection Platform",
    description="Real-time Device Threat Scoring & Predictive Security Analytics",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware to allow requests from any origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File system paths for models, telemetry database, output threat scores, and performance metadata
MODEL_PATH = "champion_model.pkl"
DATA_PATH = "it_security_iot_botnet_ddos_detection_80k.csv"
SCORES_PATH = "threat_scores.csv"
IMPORTANCE_PATH = "feature_importance.csv"
RESULTS_PATH = "model_results.csv"
ACTIVE_MODEL_FILE = "active_model.txt"

def get_model_path(model_name: str) -> str:
    """
    Resolves the physical pickle file path corresponding to a chosen model name.
    """
    mapping = {
        'Logistic Regression': 'model_logistic_regression.pkl',
        'Decision Tree': 'model_decision_tree.pkl',
        'Random Forest': 'model_random_forest.pkl',
        'XGBoost': 'model_xgboost.pkl'
    }
    return mapping.get(model_name, 'model_logistic_regression.pkl')

# Global runtime state declarations
active_model_name = "Logistic Regression"
model_pipeline = None
df_full_devices = None
feature_defaults = {}
median_botnet_score = 0.0
unique_categories = {}

# Expected fields for pre-processing mappings
NUMERICAL_COLS = [
    'Outbound_Packets_Sec', 'Avg_Packet_Size_Bytes', 'Unique_Dest_IPs',
    'Device_CPU_Usage_Pct', 'Calculated_Botnet_Score'
]

CATEGORICAL_COLS = [
    'Device_Category', 'Threat_Classification'
]

# Unified feature listing required by ML pipeline column transformer
FEATURE_COLS = [
    'Outbound_Packets_Sec', 'Avg_Packet_Size_Bytes', 'Unique_Dest_IPs',
    'Device_CPU_Usage_Pct', 'Calculated_Botnet_Score', 'Device_Category',
    'Threat_Classification', 'packets_per_ip', 'cpu_intensity_per_packet',
    'network_bandwidth_est', 'high_risk_device'
]

def startup_event():
    """
    Executed on service startup. Loads the designated active model, validates datasets,
    populates caching systems for feature column parameters, and initializes database merges.
    """
    global model_pipeline, active_model_name, df_full_devices, feature_defaults, median_botnet_score, unique_categories
    
    # 1. Read persistent model selection flag if available
    if os.path.exists(ACTIVE_MODEL_FILE):
        try:
            with open(ACTIVE_MODEL_FILE, "r") as f:
                active_model_name = f.read().strip()
        except:
            active_model_name = "Logistic Regression"
    else:
        active_model_name = "Logistic Regression"
        
    model_path = get_model_path(active_model_name)
    # Fallback to general champion_model.pkl if the specific model file is missing
    if not os.path.exists(model_path) and os.path.exists(MODEL_PATH):
        model_path = MODEL_PATH
        
    # 2. Deserialise ML pipeline model checkpoint from file
    if os.path.exists(model_path):
        try:
            model_pipeline = joblib.load(model_path)
            print(f"Model successfully loaded from: {model_path} ({active_model_name})")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
    else:
        print(f"Warning: Model file '{model_path}' not found. Prediction endpoints will fail until the model is trained.")
        
    # 3. Load device telemetry data and cached threat classifications
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            print(f"Base data loaded from {DATA_PATH} with {len(df):,} records")
            
            # Merge existing pre-calculated device threat scores database if present
            if os.path.exists(SCORES_PATH):
                df_scores = pd.read_csv(SCORES_PATH)
                print(f"Scores loaded from {SCORES_PATH}")
                df_full_devices = pd.merge(df, df_scores, on='Device_ID', how='left')
            else:
                print(f"Scores file {SCORES_PATH} not found. Scoring full dataset inline...")
                df_full_devices = df.copy()
                df_full_devices['threat_score'] = 0.0
                df_full_devices['predicted_compromise'] = 0
                df_full_devices['threat_category'] = 'Unscored'
                
            # Fill default values for missing scores or categories
            df_full_devices['threat_score'] = df_full_devices['threat_score'].fillna(0.0)
            df_full_devices['predicted_compromise'] = df_full_devices['predicted_compromise'].fillna(0).astype(int)
            df_full_devices['threat_category'] = df_full_devices['threat_category'].fillna('Low Threat')
            
            # Cache statistical parameters to handle missing data in real-time prediction request structures
            median_botnet_score = float(df['Calculated_Botnet_Score'].median())
            
            for col in NUMERICAL_COLS:
                if col in df.columns:
                    feature_defaults[col] = float(df[col].median())
                else:
                    feature_defaults[col] = 0.0
            
            for col in CATEGORICAL_COLS:
                if col in df.columns:
                    unique_categories[col] = df[col].dropna().unique().tolist()
                    feature_defaults[col] = df[col].mode()[0] if not df[col].mode().empty else ""
                else:
                    unique_categories[col] = []
                    feature_defaults[col] = ""
                    
            print("Feature defaults and category schemas initialized.")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            df_full_devices = pd.DataFrame()
    else:
        print(f"Warning: Base dataset '{DATA_PATH}' not found. App analytics and DB view will be empty.")
        df_full_devices = pd.DataFrame()

class DeviceTelemetryInput(BaseModel):
    """
    Pydantic request body mapping for single real-time telemetry inputs.
    """
    Device_Category: Optional[str] = None
    Outbound_Packets_Sec: Optional[float] = None
    Avg_Packet_Size_Bytes: Optional[float] = None
    Unique_Dest_IPs: Optional[int] = None
    Device_CPU_Usage_Pct: Optional[float] = None
    Calculated_Botnet_Score: Optional[float] = None
    Threat_Classification: Optional[str] = None

@app.post("/api/predict")
def predict_single(device_input: DeviceTelemetryInput):
    """
    Accepts single telemetry metric logs, runs feature engineering, feeds data
    through selected pipeline, and outputs risk categories along with security mitigation recommendations.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="ML Model not loaded on server. Please train the model.")
        
    input_dict = device_input.dict()
    
    # 1. Fill missing input fields with baseline defaults
    final_dict = {}
    for col in NUMERICAL_COLS:
        final_dict[col] = input_dict[col] if input_dict[col] is not None else feature_defaults.get(col, 0.0)
    for col in CATEGORICAL_COLS:
        final_dict[col] = input_dict[col] if input_dict[col] is not None else feature_defaults.get(col, "")
        
    # 2. Construct computed features matching trained pipeline transformations
    final_dict['packets_per_ip'] = final_dict['Outbound_Packets_Sec'] / (final_dict['Unique_Dest_IPs'] + 1)
    final_dict['cpu_intensity_per_packet'] = final_dict['Device_CPU_Usage_Pct'] / (final_dict['Outbound_Packets_Sec'] + 1)
    final_dict['network_bandwidth_est'] = final_dict['Outbound_Packets_Sec'] * final_dict['Avg_Packet_Size_Bytes']
    final_dict['high_risk_device'] = int(final_dict['Calculated_Botnet_Score'] > 50)
    
    # Render dataframe input for the classifier pipeline
    df_input = pd.DataFrame([final_dict])[FEATURE_COLS]
    
    try:
        # Run classification and threat score probability estimation
        prob = float(model_pipeline.predict_proba(df_input)[0][1])
        pred = int(model_pipeline.predict(df_input)[0])
        
        # 3. Categorize device risk thresholds & recommendations
        if prob >= 0.70:
            category = 'High Threat'
            color = 'red'
            rec = "Quarantine device immediately. Terminate outbound traffic, close ports, and trigger syslog inspection."
        elif prob >= 0.30:
            category = 'Medium Threat'
            color = 'yellow'
            rec = "Flag device for deep packet inspection. Monitor outbound behavior closely and perform firmware compliance audit."
        else:
            category = 'Low Threat'
            color = 'blue'
            rec = "No action required. Keep monitoring under baseline security policies."
            
        # 4. Resolve key contextual factors contributing to output classifications
        key_factors = []
        if final_dict['Calculated_Botnet_Score'] > 75:
            key_factors.append("Calculated botnet signature index is critically high.")
        elif final_dict['Calculated_Botnet_Score'] < 20:
            key_factors.append("Low calculated botnet signature matches standard device behavior.")
            
        if final_dict['Outbound_Packets_Sec'] > 150:
            key_factors.append("Extreme outbound packet transmission rate indicates active DDoS threat.")
        elif final_dict['Outbound_Packets_Sec'] < 10:
            key_factors.append("Low packet rate suggests normal network telemetry footprint.")
            
        if final_dict['Device_CPU_Usage_Pct'] > 85:
            key_factors.append("Critical CPU load (>85%) suggests background malware or command execution.")
        elif final_dict['Device_CPU_Usage_Pct'] < 15:
            key_factors.append("CPU usage is within normal idle limits.")
            
        if len(key_factors) == 0:
            key_factors.append("Traffic characteristics are consistent with standard baseline operations.")
            
        return {
            "success": True,
            "threat_score": prob,
            "predicted_compromise": pred,
            "threat_category": category,
            "color": color,
            "recommendation": rec,
            "key_factors": key_factors,
            "input_used": final_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/devices")
def get_devices(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = None,
    threat_level: Optional[str] = None,
    compromised: Optional[int] = None,
    sort_by: Optional[str] = 'threat_score',
    sort_desc: bool = True
):
    """
    Returns lists of scored devices containing search query tags, category filters, and sorting parameters.
    Handles data pagination constraints to optimize frontend dashboard responsiveness.
    """
    global df_full_devices
    if df_full_devices is None or df_full_devices.empty:
        return {"devices": [], "total": 0, "page": page, "limit": limit, "pages": 0}
        
    filtered_df = df_full_devices.copy()
    
    # Apply text filter for Device ID
    if search:
        filtered_df = filtered_df[filtered_df['Device_ID'].str.contains(search, case=False, na=False)]
        
    # Apply threat category filter
    if threat_level and threat_level != 'All':
        filtered_df = filtered_df[filtered_df['threat_category'] == threat_level]
        
    # Apply ground truth compromise label filter
    if compromised is not None:
        filtered_df = filtered_df[filtered_df['Is_Compromised_Label'] == compromised]
        
    # Apply sorting routines
    if sort_by and sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=not sort_desc)
        
    # Paginate filtered subset list
    total_count = len(filtered_df)
    total_pages = (total_count + limit - 1) // limit
    
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_df = filtered_df.iloc[start_idx:end_idx]
    
    devices_list = paginated_df.to_dict(orient='records')
    
    # Handle NaN values to prevent raw JSON serialization failure
    for device in devices_list:
        for key, val in device.items():
            if isinstance(val, float) and np.isnan(val):
                device[key] = None
                
    return {
        "devices": devices_list,
        "total": total_count,
        "page": page,
        "limit": limit,
        "pages": total_pages
    }

@app.get("/api/stats")
def get_stats():
    """
    Compiles summary aggregate stats for rendering the analytics dashboards:
    Calculates total volume, compromise rates, risk level ratios, cumulative bandwidth
    contributions, model results, and device metadata groupings.
    """
    global df_full_devices
    
    feature_importance = []
    # 1. Fetch feature relative importance stats from cache file
    if os.path.exists(IMPORTANCE_PATH):
        try:
            df_imp = pd.read_csv(IMPORTANCE_PATH)
            feature_importance = df_imp.head(10).to_dict(orient='records')
        except Exception as e:
            print(f"Error loading feature importance: {e}")
            
    model_results = []
    # 2. Fetch baseline pipeline results metrics
    if os.path.exists(RESULTS_PATH):
        try:
            df_res = pd.read_csv(RESULTS_PATH)
            model_results = df_res.to_dict(orient='records')
        except Exception as e:
            print(f"Error loading model results: {e}")
            
    if df_full_devices is None or df_full_devices.empty:
        return {
            "summary": {
                "total_devices": 0,
                "compromise_rate": 0.0,
                "high_threat_pct": 0.0,
                "avg_threat_score": 0.0,
                "total_bandwidth": 0.0,
                "compromised_bandwidth_risk": 0.0
            },
            "device_categories": [],
            "threat_classifications": [],
            "feature_importance": feature_importance,
            "model_results": model_results,
            "category_schemas": unique_categories
        }
        
    # 3. Calculate statistics indicators
    total_devices = len(df_full_devices)
    compromise_rate = float(df_full_devices['Is_Compromised_Label'].mean() * 100) if 'Is_Compromised_Label' in df_full_devices.columns else 0.0
    
    high_threat_count = len(df_full_devices[df_full_devices['threat_category'] == 'High Threat'])
    high_threat_pct = float(high_threat_count / total_devices * 100) if total_devices > 0 else 0.0
    
    avg_score = float(df_full_devices['threat_score'].mean() * 100)
    
    total_bandwidth = 0.0
    compromised_bandwidth_risk = 0.0
    if 'Outbound_Packets_Sec' in df_full_devices.columns:
        total_bandwidth = float(df_full_devices['Outbound_Packets_Sec'].sum())
        if 'threat_score' in df_full_devices.columns:
            # Bandwidth hazard estimator: packets rate * threat probability score
            compromised_bandwidth_risk = float((df_full_devices['Outbound_Packets_Sec'] * df_full_devices['threat_score']).sum())
            
    # 4. Group data records by device category schemas
    device_categories = []
    if 'Device_Category' in df_full_devices.columns and 'Is_Compromised_Label' in df_full_devices.columns:
        cat_grouped = df_full_devices.groupby('Device_Category')['Is_Compromised_Label'].agg(['count', 'mean']).reset_index()
        cat_grouped['mean'] = cat_grouped['mean'] * 100
        cat_grouped = cat_grouped.sort_values(by='mean', ascending=False)
        device_categories = cat_grouped.to_dict(orient='records')
        
    # 5. Group data records by threat classification schemas
    threat_classifications = []
    if 'Threat_Classification' in df_full_devices.columns and 'Is_Compromised_Label' in df_full_devices.columns:
        threat_grouped = df_full_devices.groupby('Threat_Classification')['Is_Compromised_Label'].agg(['count', 'mean']).reset_index()
        threat_grouped['mean'] = threat_grouped['mean'] * 100
        threat_grouped = threat_grouped.sort_values(by='mean', ascending=False)
        threat_classifications = threat_grouped.to_dict(orient='records')
        
    return {
        "summary": {
            "total_devices": total_devices,
            "compromise_rate": round(compromise_rate, 2),
            "high_threat_pct": round(high_threat_pct, 2),
            "avg_threat_score": round(avg_score, 2),
            "total_bandwidth": round(total_bandwidth, 2),
            "compromised_bandwidth_risk": round(compromised_bandwidth_risk, 2)
        },
        "device_categories": device_categories,
        "threat_classifications": threat_classifications,
        "feature_importance": feature_importance,
        "model_results": model_results,
        "active_model": active_model_name,
        "category_schemas": {
            'Device_Category': unique_categories.get('Device_Category', []),
            'Threat_Classification': unique_categories.get('Threat_Classification', [])
        }
    }

@app.post("/api/bulk-predict")
async def bulk_predict(file: UploadFile = File(...)):
    """
    Accepts telemetry raw CSV file uploads, processes feature engineering inline,
    makes batch threat score assertions using active classifier model,
    saves the output configurations, and returns download payloads.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="ML Model not loaded on server. Please train the model.")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = await file.read()
        df_uploaded = pd.read_csv(io.BytesIO(contents))
        
        # 1. Fill missing columns with statistical base defaults
        for col in NUMERICAL_COLS + CATEGORICAL_COLS:
            if col not in df_uploaded.columns:
                df_uploaded[col] = feature_defaults.get(col)
                
        # 2. Run batch custom feature engineering formulas
        df_uploaded['packets_per_ip'] = df_uploaded['Outbound_Packets_Sec'] / (df_uploaded['Unique_Dest_IPs'] + 1)
        df_uploaded['cpu_intensity_per_packet'] = df_uploaded['Device_CPU_Usage_Pct'] / (df_uploaded['Outbound_Packets_Sec'] + 1)
        df_uploaded['network_bandwidth_est'] = df_uploaded['Outbound_Packets_Sec'] * df_uploaded['Avg_Packet_Size_Bytes']
        df_uploaded['high_risk_device'] = (df_uploaded['Calculated_Botnet_Score'] > 50).astype(int)
        
        X_pred = df_uploaded[FEATURE_COLS]
        
        # 3. Model batch scoring predictions
        threat_scores = model_pipeline.predict_proba(X_pred)[:, 1]
        predictions = model_pipeline.predict(X_pred)
        
        df_uploaded['threat_score'] = threat_scores
        df_uploaded['predicted_compromise'] = predictions
        
        # Bin threat scores into hazard risk classes
        df_uploaded['threat_category'] = pd.cut(
            df_uploaded['threat_score'],
            bins=[0.0, 0.3, 0.7, 1.0],
            labels=['Low Threat', 'Medium Threat', 'High Threat'],
            include_lowest=True
        )
        
        # 4. Compile batch summary stats and result preview (top 20 rows)
        preview_cols = ['Device_ID', 'Device_Category', 'Outbound_Packets_Sec', 'threat_score', 'predicted_compromise', 'threat_category']
        available_preview_cols = [c for c in preview_cols if c in df_uploaded.columns]
        preview_data = df_uploaded[available_preview_cols].head(20).to_dict(orient='records')
        
        # Safe format float parameters
        for row in preview_data:
            for k, v in row.items():
                if isinstance(v, float) and np.isnan(v):
                    row[k] = None
                    
        # 5. Format results database stream
        out_buf = io.StringIO()
        df_uploaded.to_csv(out_buf, index=False)
        out_buf.seek(0)
        
        csv_bytes = out_buf.getvalue().encode('utf-8')
        import base64
        csv_b64 = base64.b64encode(csv_bytes).decode('utf-8')
        
        return {
            "success": True,
            "summary": {
                "total_scored": len(df_uploaded),
                "predicted_compromises": int(df_uploaded['predicted_compromise'].sum()),
                "compromise_rate_pct": float(df_uploaded['predicted_compromise'].mean() * 100),
                "high_threat_pct": float((df_uploaded['threat_category'] == 'High Threat').mean() * 100)
            },
            "preview": preview_data,
            "csv_data": csv_b64,
            "filename": f"scored_{file.filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk processing error: {str(e)}")

@app.post("/api/select-model")
def select_model(model_name: str = Query(...)):
    """
    Switches active classification model. Loads model parameters dynamically,
    updates persistent state flag file, and alters endpoint prediction behaviors.
    """
    global active_model_name, model_pipeline
    path = get_model_path(model_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Model checkpoint for {model_name} not found. Please train first.")
    
    try:
        model_pipeline = joblib.load(path)
        active_model_name = model_name
        # Persist selection to state file
        with open(ACTIVE_MODEL_FILE, "w") as f:
            f.write(model_name)
        return {"success": True, "active_model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {str(e)}")

@app.post("/api/retrain")
def retrain_model(model_name: Optional[str] = Query(None)):
    """
    Triggers backend ML training pipeline logic. Refreshes application memory
    checkpoints and datasets on successful completion.
    """
    try:
        from run_ml_pipeline import run_pipeline
        run_pipeline(model_to_train=model_name)
        
        # Update active model selections if single target model trained
        if model_name:
            global active_model_name
            active_model_name = model_name
            with open(ACTIVE_MODEL_FILE, "w") as f:
                f.write(model_name)
                
        # Trigger startup routines to load new model/scores
        startup_event()
        return {"success": True, "message": "Model retrained successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Serves front-end single page dashboard application file index.html.
    """
    html_path = "index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        <html>
            <head><title>App Error</title></head>
            <body style="font-family: sans-serif; padding: 50px; text-align: center; background-color: #0f172a; color: #cbd5e1;">
                <h2>index.html is missing!</h2>
                <p>Please place index.html in the same directory as main.py.</p>
            </body>
        </html>
        """

if __name__ == "__main__":
    print("Starting FastAPI Application Server...")
    # Bind to loopback interface and trigger automatic reloading routines for dev environment
    uvicorn.run("main:app", host="127.0.0.2", port=8000, reload=True)
