import joblib
import os

OLD_MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\Phase 5\phase5_artifacts\vero_model_calibrated.pkl"
NEW_MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\Phase 5\phase5_artifacts\vero_model_calibrated.joblib"

print("Loading model from:", OLD_MODEL_PATH)
model = joblib.load(OLD_MODEL_PATH)

print("Saving model to:", NEW_MODEL_PATH)
joblib.dump(model, NEW_MODEL_PATH)

print("Done. Joblib model created.")
