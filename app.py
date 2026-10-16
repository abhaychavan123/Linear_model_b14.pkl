import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Model loading logic
model = None
status_msg = "Initializing..."

def init_model():
    global model, status_msg
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = ["Linear_model_b14.pkl", "linear_model_b14.pkl", "model.pkl"]
    
    target_path = None
    for name in candidates:
        full_path = os.path.join(base_dir, name)
        if os.path.exists(full_path):
            target_path = full_path
            break

    if not target_path:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower() in ["linear_model_b14.pkl", "linear_model.pkl"]:
                    target_path = os.path.join(root, f)
                    break
            if target_path:
                break

    if not target_path:
        status_msg = "Model file 'Linear_model_b14.pkl' not found."
        print(f"[ERROR] {status_msg}", file=sys.stderr)
        return

    try:
        with open(target_path, "rb") as f:
            model = pickle.load(f)
        status_msg = f"Model loaded: {os.path.basename(target_path)}"
        print(f"[INFO] {status_msg}", file=sys.stdout)
    except Exception as e:
        status_msg = f"Error loading model: {str(e)}"
        print(f"[ERROR] {status_msg}", file=sys.stderr)

init_model()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insurance Cost Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --surface: #121929;
            --surface-hover: #1b243b;
            --border: #23314e;
            --accent: #2563eb;
            --accent-hover: #1d4ed8;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --card-radius: 16px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2.5rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 680px;
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--card-radius);
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }

        .warning-pill {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            color: #fca5a5;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        .field-group {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }

        label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }

        input, select {
            width: 100%;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            background-color: var(--bg-color);
            border: 1px solid var(--border);
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        input:focus, select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.25);
        }

        select option {
            background-color: var(--bg-color);
            color: var(--text-primary);
        }

        .btn-submit {
            margin-top: 1.75rem;
            width: 100%;
            padding: 0.85rem;
            background-color: var(--accent);
            color: #ffffff;
            font-weight: 600;
            font-size: 1rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
        }

        .btn-submit:active {
            transform: scale(0.99);
        }

        .result-card {
            margin-top: 1.75rem;
            padding: 1.5rem;
            border-radius: 12px;
            background: rgba(37, 99, 235, 0.1);
            border: 1px solid rgba(37, 99, 235, 0.3);
            text-align: center;
        }

        .result-card .label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }

        .result-card .amount {
            display: block;
            margin-top: 0.35rem;
            font-size: 2rem;
            font-weight: 700;
            color: #60a5fa;
        }

        @media (max-width: 600px) {
            .grid {
                grid-template-columns: 1fr;
            }
            .container {
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Medical Insurance Cost Predictor</h1>
            <p>Linear Regression Model Service</p>
        </div>

        {% if not is_ready %}
            <div class="warning-pill">{{ status_msg }}</div>
        {% endif %}

        <form method="POST" action="/">
            <div class="grid">
                <div class="field-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" min="18" max="100" value="30" required>
                </div>

                <div class="field-group">
                    <label for="sex">Sex</label>
                    <select id="sex" name="sex" required>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <div class="field-group">
                    <label for="bmi">BMI (Body Mass Index)</label>
                    <input type="number" step="0.1" id="bmi" name="bmi" min="10.0" max="60.0" value="27.5" required>
                </div>

                <div class="field-group">
                    <label for="children">Number of Children</label>
                    <input type="number" id="children" name="children" min="0" max="10" value="0" required>
                </div>

                <div class="field-group">
                    <label for="smoker">Smoker</label>
                    <select id="smoker" name="smoker" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <div class="field-group">
                    <label for="region">Region</label>
                    <select id="region" name="region" required>
                        <option value="0">Southwest</option>
                        <option value="1">Southeast</option>
                        <option value="2">Northwest</option>
                        <option value="3">Northeast</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="btn-submit" {% if not is_ready %}disabled style="opacity: 0.5;"{% endif %}>
                Calculate Cost
            </button>
        </form>

        {% if prediction is not none %}
            <div class="result-card">
                <div class="label">Estimated Annual Insurance Cost</div>
                <div class="amount">${{ "{:,.2f}".format(prediction) }}</div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST" and model is not None:
        feature_order = ["age", "sex", "bmi", "children", "smoker", "region"]
        try:
            inputs = [float(request.form.get(f, 0)) for f in feature_order]
            features_raw = np.array([inputs])

            # Pass features matching scikit-learn input expectations
            if hasattr(model, "feature_names_in_"):
                features = pd.DataFrame(features_raw, columns=model.feature_names_in_)
            else:
                features = features_raw

            pred = model.predict(features)[0]
            # Ensure price does not predict negative values
            prediction = max(0.0, float(pred))
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}", file=sys.stderr)

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        is_ready=(model is not None),
        status_msg=status_msg
    )

@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
