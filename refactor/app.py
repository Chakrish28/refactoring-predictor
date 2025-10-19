from flask import Flask, request, render_template, jsonify
import joblib
from radon.raw import analyze
from radon.complexity import cc_visit

app = Flask(__name__)
model = joblib.load("model.pkl")

# --- NEW: Define our custom threshold ---
# We'll flag any code if the model is at least 30% sure it needs refactoring.
PREDICTION_THRESHOLD = 0.30

def get_features_from_code(code_string):
    # This function remains the same as the one from my last response
    try:
        if not code_string.strip(): return None
        raw_metrics = analyze(code_string)
        cc_results = cc_visit(code_string)
        lloc = raw_metrics.lloc
        comments = raw_metrics.comments
        num_functions = len(cc_results)
        if num_functions > 0:
            total_cc = sum(item.complexity for item in cc_results)
            avg_cc = total_cc / num_functions
            max_cc = max(item.complexity for item in cc_results)
        else:
            avg_cc = 0
            max_cc = 0
        return [lloc, comments, avg_cc, max_cc, num_functions]
    except Exception:
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        code = data.get('code')
        features = get_features_from_code(code)
        
        if features is None:
            return jsonify({"error": "Could not analyze code."}), 400

        # --- MODIFIED PREDICTION LOGIC ---
        # 1. Get the raw probabilities: [prob_of_0, prob_of_1]
        probabilities = model.predict_proba([features])[0]
        
        # 2. Get the probability of class '1' (Needs Refactor)
        prob_needs_refactor = probabilities[1]
        
        # 3. Apply our custom threshold
        if prob_needs_refactor >= PREDICTION_THRESHOLD:
            needs_refactor = 1
            confidence = prob_needs_refactor * 100
        else:
            needs_refactor = 0
            confidence = probabilities[0] * 100
        # --- END OF MODIFICATION ---
        
        return jsonify({
            "needs_refactor": needs_refactor,
            "confidence": f"{confidence:.2f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

