from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import io
import sys
import traceback

# Reuse helpers from existing ML script
from machine_learning import (
    find_label_columns,
    remove_tagalog_columns,
    prepare_features,
    generate_strand_breakdown,
    simple_rule_prediction,
    generate_structured_recommendation,
    MODEL_OUTPUT, SCALER_OUTPUT, FEATURES_OUTPUT
)

import os
import json
import numpy as np
import joblib
import datetime

app = Flask(__name__)
CORS(app)


def load_model_bundle():
    model = scaler = feature_data = None
    if os.path.exists(MODEL_OUTPUT) and os.path.exists(SCALER_OUTPUT) and os.path.exists(FEATURES_OUTPUT):
        try:
            model = joblib.load(MODEL_OUTPUT)
            scaler = joblib.load(SCALER_OUTPUT)
            with open(FEATURES_OUTPUT, "r") as f:
                feature_data = json.load(f)
        except Exception:
            model = scaler = feature_data = None
    return model, scaler, feature_data


def predict_dataframe(df_in: pd.DataFrame):
    """
    Core prediction routine that mirrors machine_learning.run_prediction_mode,
    returning a list of prediction dicts instead of printing to stdout.
    """
    model, scaler, feature_data = load_model_bundle()
    training_features = None
    training_medians = None
    if feature_data:
        training_features = feature_data.get("feature_names", [])
        training_medians = pd.Series(feature_data.get("medians", {}))

    acad_candidates, tvl_candidates = find_label_columns(df_in)
    acad_col = acad_candidates[0] if acad_candidates else None
    tvl_col = tvl_candidates[0] if tvl_candidates else None

    results = []

    for _, row in df_in.iterrows():
        predicted = None
        confidence = None
        all_probabilities = {}
        proba_array = None

        # Try model path
        if model is not None and training_features is not None:
            try:
                row_df = pd.DataFrame([row])
                preserve_set = set([c for c in [acad_col, tvl_col] if c])
                row_df = remove_tagalog_columns(row_df, preserve_set)

                drop_cols = []
                if acad_col and acad_col in row_df.columns:
                    drop_cols.append(acad_col)
                if tvl_col and tvl_col in row_df.columns:
                    drop_cols.append(tvl_col)

                X_row, _, _, _, _, _ = prepare_features(
                    row_df,
                    drop_cols=drop_cols,
                    is_training=False,
                    scaler=scaler,
                    training_medians=training_medians
                )
                X_row = X_row.reindex(columns=training_features, fill_value=0).fillna(0)

                if X_row.shape[1] == len(training_features):
                    proba = model.predict_proba(X_row)[0]
                    labels = list(getattr(model, 'classes_', ['Academic', 'TechPro', 'Undecided']))
                    for i, label in enumerate(labels):
                        all_probabilities[label] = float(proba[i])
                    idx_max = int(np.argmax(proba))
                    predicted = labels[idx_max]
                    confidence = float(np.max(proba))
                    proba_array = proba
            except Exception:
                predicted = None

        # Fallback rule
        if predicted is None:
            predicted, confidence = simple_rule_prediction(row, acad_col, tvl_col)
            if predicted == "Academic":
                all_probabilities = {'Academic': confidence, 'TechPro': 1 - confidence, 'Undecided': 0.0}
            elif predicted == "TechPro":
                all_probabilities = {'Academic': 1 - confidence, 'TechPro': confidence, 'Undecided': 0.0}
            else:
                all_probabilities = {'Academic': 0.33, 'TechPro': 0.33, 'Undecided': 0.34}
            proba_array = [all_probabilities.get('Academic', 0), all_probabilities.get('TechPro', 0), all_probabilities.get('Undecided', 0)]

        strand_breakdown = generate_strand_breakdown(predicted)

        # Minimal student fields (best-effort)
        def pick_ci(row_like, names):
            for c in row_like.index if hasattr(row_like, 'index') else row_like.keys():
                n = str(c).strip().lower()
                for cand in names:
                    if n == cand.lower() or n.replace('_', ' ').replace('-', ' ') == cand.lower().replace('_', ' ').replace('-', ' '):
                        return row_like.get(c)
            return None

        student_id = str(pick_ci(row, ['student_id', 'student id', 'id', 'lrn', 'studentid']) or '').strip()
        student_name = str(pick_ci(row, ['student_name', 'student name', 'name', 'full_name', 'full name']) or '').strip()
        grade_section = str(pick_ci(row, ['grade_section', 'grade section', 'section', 'grade & section', 'grade and section']) or '').strip()
        email = pick_ci(row, ['gmail', 'email', 'email address'])
        gender = pick_ci(row, ['gender', 'sex'])
        age = pick_ci(row, ['age'])

        try:
            structured_rec = generate_structured_recommendation(
                row, predicted, confidence, all_probabilities if all_probabilities else proba_array, list(df_in.columns)
            )
        except Exception:
            structured_rec = None

        item = {
            'student_id': student_id,
            'student_name': student_name,
            'gmail': str(email).strip() if email is not None else '',
            'gender': str(gender).strip() if gender is not None else '',
            'age': int(age) if pd.notnull(age) and str(age).strip() != '' else '',
            'grade_section': grade_section,
            'predicted_track': predicted,
            'confidence': round(float(confidence), 2) if confidence is not None else 0.75,
            'strand_breakdown': strand_breakdown
        }
        if structured_rec:
            item['recommendation'] = structured_rec
        results.append(item)
    return results


@app.get("/health")
def health():
    """Enhanced health check with model status"""
    health_data = {
        "status": "ok",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "service": "StrandAlign ML API",
        "version": "1.0"
    }
    
    # Check model files
    model_status = {
        "model_loaded": False,
        "scaler_loaded": False,
        "features_loaded": False,
        "model_file_exists": os.path.exists(MODEL_OUTPUT),
        "scaler_file_exists": os.path.exists(SCALER_OUTPUT),
        "features_file_exists": os.path.exists(FEATURES_OUTPUT)
    }
    
    # Try to load model bundle
    try:
        model, scaler, feature_data = load_model_bundle()
        model_status["model_loaded"] = model is not None
        model_status["scaler_loaded"] = scaler is not None
        model_status["features_loaded"] = feature_data is not None
        
        if feature_data:
            model_status["num_features"] = len(feature_data.get("feature_names", []))
            
        # Check if model can predict
        if model is not None:
            try:
                # Get model classes if available
                model_status["model_classes"] = list(getattr(model, 'classes_', ['Academic', 'TechPro', 'Undecided']))
                model_status["ready_for_predictions"] = True
            except Exception:
                model_status["ready_for_predictions"] = False
        else:
            model_status["ready_for_predictions"] = False
            model_status["fallback_mode"] = "Using simple rule prediction"
    except Exception as e:
        model_status["error"] = str(e)
        model_status["ready_for_predictions"] = False
    
    health_data["model_status"] = model_status
    
    # Overall health determination
    if model_status.get("ready_for_predictions") or (
        model_status.get("model_file_exists") and 
        model_status.get("scaler_file_exists") and 
        model_status.get("features_file_exists")
    ):
        health_data["overall_status"] = "healthy"
        return jsonify(health_data), 200
    else:
        health_data["overall_status"] = "degraded"
        health_data["warning"] = "Model files missing, using fallback prediction"
        return jsonify(health_data), 200
    
@app.get("/model/info")
def model_info():
    """Get detailed model information"""
    try:
        model, scaler, feature_data = load_model_bundle()
        
        info = {
            "model_available": model is not None,
            "scaler_available": scaler is not None,
            "features_available": feature_data is not None,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        if model is not None:
            info["model_type"] = type(model).__name__
            info["model_classes"] = list(getattr(model, 'classes_', ['Academic', 'TechPro', 'Undecided']))
        
        if feature_data:
            info["num_features"] = len(feature_data.get("feature_names", []))
            info["feature_names_sample"] = feature_data.get("feature_names", [])[:10]
        
        return jsonify(info), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/predict-csv")
def predict_csv():
    """
    Multipart form with file field 'file' containing a CSV.
    Returns JSON list with predictions.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded with field 'file'"}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        data = f.read()
        try:
            df = pd.read_csv(io.BytesIO(data), encoding='utf-8')
        except Exception:
            df = pd.read_csv(io.BytesIO(data), encoding='latin1')
        results = predict_dataframe(df)
        return jsonify({"count": len(results), "results": results})
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500


@app.post("/predict-rows")
def predict_rows():
    """
    JSON body: { "rows": [ {col: value, ...}, ... ] }
    """
    try:
        payload = request.get_json(silent=True) or {}
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            return jsonify({"error": "Body must be JSON with non-empty 'rows' array"}), 400
        df = pd.DataFrame(rows)
        results = predict_dataframe(df)
        return jsonify({"count": len(results), "results": results})
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Environment variables PORT and HOST can be used to configure
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)


