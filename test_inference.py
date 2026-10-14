import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd

# Test sample as defined in the experiment guide
sample = pd.DataFrame(
    [
        {
            "Age": 30,
            "Sex": "F",
            "BP": "NORMAL",
            "Cholesterol": "NORMAL",
            "Na_to_K": 15.0,
        }
    ]
)

# 1. Original scikit-learn pipeline inference
model = joblib.load("model/drug_model.joblib")
sklearn_pred = model.predict(sample)[0]
sklearn_proba = model.predict_proba(sample)[0]
print(f"sklearn: {sklearn_pred} (probabilities: {sklearn_proba})")

# 2. ONNX Runtime inference (same inputs and data types the web browser will send)
sess = rt.InferenceSession("model/drug_model.onnx")
feeds = {
    "Age": np.array([[30]], dtype=np.int64),
    "Sex": np.array([["F"]], dtype=object),
    "BP": np.array([["NORMAL"]], dtype=object),
    "Cholesterol": np.array([["NORMAL"]], dtype=object),
    "Na_to_K": np.array([[15.0]], dtype=np.float64),
}

output = sess.run(None, feeds)
onnx_label = output[0][0]
onnx_probs = output[1]
print(f"onnx:    {onnx_label} (probabilities: {onnx_probs})")

# Verify equality
assert (
    sklearn_pred == onnx_label
), f"Prediction mismatch: {sklearn_pred} vs {onnx_label}"
print("\nVerification Passed: Scikit-learn and ONNX Runtime predictions match perfectly!")
