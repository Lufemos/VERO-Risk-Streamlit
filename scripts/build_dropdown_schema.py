import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]  # project root
BASE_MODEL_PATH = ROOT / "app" / "vero_base_model_prefit.joblib"
IN_SCHEMA_PATH = ROOT / "app" / "vero_feature_schema.json"
OUT_SCHEMA_PATH = ROOT / "app" / "vero_feature_schema_enriched.json"

def extract_ohe_categories(pipeline):
    """
    Extract OneHotEncoder categories from:
      Pipeline(preprocess=ColumnTransformer(num, cat), model=...)
    """
    pre = pipeline.named_steps["preprocess"]
    cat_transformer = pre.named_transformers_["cat"]
    ohe = cat_transformer.named_steps["onehot"]

    cat_cols = pre.transformers_[1][2]  # ("cat", pipeline, cols)
    categories = ohe.categories_

    out = {}
    for col, cats in zip(cat_cols, categories):
        # Convert numpy types to plain python strings
        out[col] = [str(x) for x in cats.tolist()]
    return out

def main():
    schema = json.loads(IN_SCHEMA_PATH.read_text(encoding="utf-8"))
    base_model = joblib.load(BASE_MODEL_PATH)

    categorical_levels = extract_ohe_categories(base_model)

    # attach to schema
    schema["categorical_levels"] = categorical_levels

    OUT_SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Saved enriched schema -> {OUT_SCHEMA_PATH}")

if __name__ == "__main__":
    main()
