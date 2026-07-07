"""
app.py
======
Flask backend for Smart Lender — Applicant Creditworthiness Prediction.

Routes
------
GET  /           -> home / prediction form (index.html)
POST /predict    -> run inference, render result.html
GET  /about      -> project about page
GET  /eda        -> univariate / bivariate / multivariate EDA analysis page
GET  /retrain    -> re-train the model pipeline and display results live

Run
---
    cd Flask
    python app.py
    # open http://127.0.0.1:5000
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

import preprocessing as pp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("smart_lender")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "smart-lender-dev-key")

# Track training status
training_in_progress = False
training_output_cache = []


# ---------------------------------------------------------------------------
# Load model + scaler once at startup
# ---------------------------------------------------------------------------
def _load_pickle(name: str):
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


model = _load_pickle("best_model.pkl") or _load_pickle("rdf.pkl")
scaler = _load_pickle("scale.pkl")

if model is None:
    log.error("No model found. Expected Flask/best_model.pkl or Flask/rdf.pkl.")
    log.error("Run `python Training/train_model.py` from the project root first.")
if scaler is None:
    log.error("Scaler not found. Expected Flask/scale.pkl.")

log.info("Model loaded: %s", type(model).__name__ if model else "NONE")
log.info("Scaler loaded: %s", type(scaler).__name__ if scaler else "NONE")


# ---------------------------------------------------------------------------
# Allowed option values (used for input validation)
# ---------------------------------------------------------------------------
ALLOWED = {
    "Gender": {"Male", "Female"},
    "Married": {"Yes", "No"},
    "Education": {"Graduate", "Not Graduate"},
    "Self_Employed": {"Yes", "No"},
    "Dependents": {"0", "1", "2", "3+"},
    "Property_Area": {"Urban", "Rural", "Semiurban"},
}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_form(form: dict) -> list[str]:
    """Return a list of human-readable error strings (empty == valid)."""
    errors: list[str] = []

    # Categorical option checks.
    for field, allowed in ALLOWED.items():
        val = form.get(field, "").strip()
        if val not in allowed:
            errors.append(f"{field}: choose one of {sorted(allowed)}.")

    # Numeric fields must parse as non-negative numbers.
    numeric = {
        "ApplicantIncome": "Applicant Income",
        "CoapplicantIncome": "Co-applicant Income",
        "LoanAmount": "Loan Amount",
    }
    for field, label in numeric.items():
        raw = form.get(field, "").strip()
        try:
            v = float(raw)
            if v < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{label}: enter a valid non-negative number.")

    # Loan_Amount_Term: positive number, typically whole months.
    raw = form.get("Loan_Amount_Term", "").strip()
    if not raw:
        errors.append("Loan Amount Term: select a term duration.")
    else:
        try:
            v = float(raw)
            if v <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Loan Amount Term: enter a positive number (months).")

    # Credit_History: must be 0 or 1.
    raw = form.get("Credit_History", "").strip()
    if raw not in {"0", "1"}:
        errors.append("Credit History: must be 0 (no record) or 1 (good record).")

    return errors


# ---------------------------------------------------------------------------
# Recommendation logic (rule-based, displayed alongside the prediction)
# ---------------------------------------------------------------------------
def build_recommendation(approved: bool, probability: float, form: dict) -> str:
    """Return a short, human-readable recommendation for the lender."""
    if approved:
        return (
            "Applicant shows a strong creditworthiness profile. "
            "Loan approval is recommended. Consider standard due-diligence "
            "(income verification, collateral assessment) before disbursement."
        )

    # Rejection reasons (highest-weighted signals for this dataset).
    reasons = []
    if str(form.get("Credit_History", "")) == "0":
        reasons.append("no/unclear credit history (the single strongest risk factor)")
    try:
        inc = float(form.get("ApplicantIncome", 0) or 0)
        loan = float(form.get("LoanAmount", 0) or 0)
        if loan > 0 and inc > 0:
            ratio = (loan * 1000) / inc  # LoanAmount is in thousands
            if ratio > 8:
                reasons.append(f"high loan-to-income ratio (~{ratio:.1f}x)")
    except (TypeError, ValueError):
        pass
    property_area = form.get("Property_Area", "")
    if property_area == "Rural":
        reasons.append("property in a lower-approval-rate area")
    if reasons:
        return (
            "Loan approval is NOT recommended. Key risk signals: "
            + "; ".join(reasons) + ". "
            "Suggest gathering more documentation or offering a smaller / secured loan."
        )
    return (
        "Applicant does not meet the approval threshold. "
        "Recommend manual review by a credit analyst."
    )


# ---------------------------------------------------------------------------
# EDA helper — generate plots if not already cached
# ---------------------------------------------------------------------------
EDA_IMAGES_DIR = os.path.join(BASE_DIR, "static", "images", "eda")
EDA_PLOTS = {
    "univariate_numeric.png": "Univariate Analysis — Numeric Features",
    "univariate_categorical.png": "Univariate Analysis — Categorical Features",
    "univariate_target.png": "Target Variable Distribution",
    "bivariate_correlation.png": "Bivariate Analysis — Correlation Matrix",
    "bivariate_categorical.png": "Loan Status vs Categorical Features",
    "bivariate_income_vs_loan.png": "Income vs Loan Amount by Status",
    "multivariate_pairplot.png": "Multivariate — Pairplot",
    "multivariate_boxplots.png": "Multivariate — Box Plots",
    "multivariate_violin.png": "Multivariate — Violin Plots",
}


def _generate_eda_plots():
    """Generate EDA plots by importing and running the eda_plots module."""
    log.info("Generating EDA plots...")
    try:
        # Add Flask dir to path
        sys.path.insert(0, BASE_DIR)
        import eda_plots
        eda_plots.generate_all_plots()
        log.info("EDA plots generated successfully.")
        return True
    except Exception as exc:
        log.exception("Failed to generate EDA plots: %s", exc)
        return False


def _check_eda_plots_exist() -> bool:
    """Check if all EDA plot images exist."""
    if not os.path.isdir(EDA_IMAGES_DIR):
        return False
    for plot_file in EDA_PLOTS:
        if not os.path.exists(os.path.join(EDA_IMAGES_DIR, plot_file)):
            return False
    return True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form.to_dict()
    log.info("Predict request: %s", {k: form.get(k) for k in sorted(form)})

    errors = validate_form(form)
    if errors:
        for e in errors:
            flash(e)
        return render_template("index.html", form=form), 400

    if model is None or scaler is None:
        flash("Server is missing trained model artifacts. Please train first.")
        return render_template("index.html", form=form), 500

    try:
        features = pp.transform_input(form, scaler)  # (1, 12)
        pred = int(model.predict(features)[0])
        proba_approved = float(np.asarray(model.predict_proba(features))[0, 1])
    except Exception as exc:
        log.exception("Prediction failed")
        flash(f"Prediction error: {exc}")
        return render_template("index.html", form=form), 500

    approved = pred == 1
    confidence = max(proba_approved, 1 - proba_approved)
    recommendation = build_recommendation(approved, proba_approved, form)

    # Get model name
    model_name = type(model).__name__

    result = {
        "status": "Approved" if approved else "Rejected",
        "probability": round(proba_approved * 100, 1),
        "confidence": round(confidence * 100, 1),
        "recommendation": recommendation,
        "model_name": model_name,
        "income": f"{float(form.get('ApplicantIncome', 0) or 0):,.0f}",
        "loan": f"{float(form.get('LoanAmount', 0) or 0):,.0f}",
        "credit_history": form.get("Credit_History"),
    }
    log.info("Prediction: %s (p=%.3f)", result["status"], proba_approved)
    return render_template("result.html", result=result)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/eda")
def eda():
    """Display EDA (univariate, bivariate, multivariate) analysis page."""
    # Check if plots need generation
    plots_ready = _check_eda_plots_exist()
    if not plots_ready:
        success = _generate_eda_plots()
        if not success:
            flash("Failed to generate EDA plots. Check server logs.", "error")
            return render_template("eda.html", plots=[], plots_ready=False)

    # Build plot list for template
    plots = []
    for filename, caption in EDA_PLOTS.items():
        img_path = url_for("static", filename=f"images/eda/{filename}")
        plots.append({"src": img_path, "caption": caption})

    return render_template("eda.html", plots=plots, plots_ready=True)


@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    """Run the training pipeline and display results."""
    global training_in_progress, training_output_cache

    if request.method == "POST":
        if training_in_progress:
            return jsonify({"status": "error", "message": "Training already in progress."}), 400

        training_in_progress = True
        training_output_cache.clear()

        try:
            # Run training script as subprocess
            train_script = os.path.join(PROJECT_ROOT, "Training", "train_model.py")
            log.info("Starting model retraining...")

            # Capture output line by line
            process = subprocess.Popen(
                [sys.executable, train_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
            )

            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                line = line.rstrip()
                training_output_cache.append(line)
                log.info("[TRAIN] %s", line)

            process.wait()
            success = process.returncode == 0

            # Reload models after training
            global model, scaler
            model = _load_pickle("best_model.pkl") or _load_pickle("rdf.pkl")
            scaler = _load_pickle("scale.pkl")

            if success:
                training_output_cache.append("\n✓ Training completed successfully!")
                training_output_cache.append(f"  Model: {type(model).__name__ if model else 'N/A'}")
                log.info("Model retraining completed successfully.")
            else:
                training_output_cache.append(f"\n✗ Training failed with return code {process.returncode}")
                log.error("Model retraining failed.")

            return jsonify({
                "status": "success" if success else "error",
                "message": "Training completed." if success else "Training failed.",
            })

        except Exception as exc:
            training_output_cache.append(f"\n✗ Error: {exc}")
            log.exception("Training failed with exception")
            return jsonify({"status": "error", "message": str(exc)}), 500
        finally:
            training_in_progress = False

    return render_template("train.html")


@app.route("/retrain/status")
def retrain_status():
    """Return current training output for polling."""
    return jsonify({
        "running": training_in_progress,
        "output": training_output_cache,
    })


@app.errorhandler(404)
def not_found(_):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_):
    return render_template("500.html"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
