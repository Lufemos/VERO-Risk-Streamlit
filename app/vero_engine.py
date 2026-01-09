from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

import joblib
import numpy as np
import pandas as pd


# ----------------------------
# Result object (backward compatible)
# ----------------------------
@dataclass(frozen=True)
class VEROResult:
    """
    Backward-compatible result object.

    Canonical fields expected by the app:
      - vero_probability  (calibrated probability, for display)
      - vero_score
      - risk_stratum

    Transparency fields:
      - base_probability
      - calibrated_probability
      - probability_used_for_score
      - used_base_for_score
    """
    base_probability: float
    calibrated_probability: float
    probability_used_for_score: float
    vero_score: int
    risk_stratum: str
    used_base_for_score: bool = False

    # ---- Backward compatible aliases
    @property
    def vero_probability(self) -> float:
        return float(self.calibrated_probability)

    @property
    def score_probability(self) -> float:
        return float(self.probability_used_for_score)

    @property
    def probability(self) -> float:
        return float(self.calibrated_probability)

    @property
    def score(self) -> int:
        return int(self.vero_score)

    @property
    def stratum(self) -> str:
        return str(self.risk_stratum)


# ----------------------------
# Engine
# ----------------------------
class VEROEngine:
    """
    Loads:
      - base_model: Pipeline(preprocess -> LogisticRegression or similar)
      - calibrator: CalibratedClassifierCV(cv="prefit", method="isotonic" or "sigmoid")
      - metadata: frozen feature list + scoring strata cutpoints

    Provides:
      - predict_single(): returns VEROResult (backward compatible)
      - top_contributors_single(): linear contribution with base-feature mapping
    """

    def __init__(
        self,
        base_model_path: str | Path,
        calibrator_path: str | Path,
        metadata_path: str | Path,
    ):
        base_model_path = Path(base_model_path)
        calibrator_path = Path(calibrator_path)
        metadata_path = Path(metadata_path)

        self.base_model = joblib.load(base_model_path)
        self.calibrator = joblib.load(calibrator_path)
        self.meta = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.feature_cols: List[str] = list(self.meta["feature_cols"])
        self.strata = self.meta["strata"]

        # If isotonic calibration collapses to 0/1, fall back to base probability for scoring.
        self._collapse_eps = float(self.meta.get("calibration_collapse_eps", 1e-12))

    # ----------------------------
    # Internals
    # ----------------------------
    def _risk_stratum(self, score: int) -> str:
        if score <= int(self.strata["low_max"]):
            return "Low"
        if score <= int(self.strata["medium_max"]):
            return "Medium"
        return "High"

    def _to_frame(self, patient: Dict[str, Any]) -> pd.DataFrame:
        # strict ordering; missing keys become NaN and are handled by imputers in the pipeline
        return pd.DataFrame([patient], columns=self.feature_cols)

    @staticmethod
    def _safe_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _is_collapsed(self, p: float) -> bool:
        if np.isnan(p):
            return True
        return (p <= self._collapse_eps) or (p >= 1.0 - self._collapse_eps)

    def _get_proba(self, model: Any, X: pd.DataFrame) -> float:
        if not hasattr(model, "predict_proba"):
            raise TypeError("Model must support predict_proba.")
        proba = model.predict_proba(X)
        return self._safe_float(proba[0, 1])

    # ----------------------------
    # Public API
    # ----------------------------
    def predict_single(self, patient: Dict[str, Any]) -> VEROResult:
        """
        Compute base + calibrated probability.
        If calibrated output collapses to ~0 or ~1 (common with isotonic), compute score from base probability.

        Returns VEROResult with:
          - calibrated_probability (transparent)
          - probability_used_for_score (what actually produced the score)
          - used_base_for_score flag
        """
        X = self._to_frame(patient)

        p_base = self._get_proba(self.base_model, X)

        if not hasattr(self.calibrator, "predict_proba"):
            raise TypeError("Calibrator must support predict_proba.")

        p_cal = self._get_proba(self.calibrator, X)

        used_base = self._is_collapsed(p_cal)
        p_used = p_base if used_base else p_cal

        score = int(np.rint(100 * p_used))
        score = max(0, min(100, score))

        return VEROResult(
            base_probability=float(p_base),
            calibrated_probability=float(p_cal),
            probability_used_for_score=float(p_used),
            vero_score=int(score),
            risk_stratum=self._risk_stratum(score),
            used_base_for_score=bool(used_base),
        )

    def predict_batch(
        self,
        patients: Iterable[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Optional convenience: predict many patient dicts and return a dataframe.
        """
        rows = []
        for p in patients:
            r = self.predict_single(p)
            rows.append(
                {
                    "base_probability": r.base_probability,
                    "calibrated_probability": r.calibrated_probability,
                    "probability_used_for_score": r.probability_used_for_score,
                    "vero_score": r.vero_score,
                    "risk_stratum": r.risk_stratum,
                    "used_base_for_score": r.used_base_for_score,
                }
            )
        return pd.DataFrame(rows)

    def top_contributors_single(self, patient: Dict[str, Any], top_k: int = 3) -> pd.DataFrame:
        """
        For linear models inside a Pipeline with:
          - named_steps["preprocess"]
          - named_steps["model"] having coef_

        contribution_j = transformed_value_j * coef_j

        Then map transformed feature names back to original frozen features.
        """
        X = self._to_frame(patient)

        if not hasattr(self.base_model, "named_steps"):
            raise TypeError("Base model must be a sklearn Pipeline with named_steps for contributors.")

        if "preprocess" not in self.base_model.named_steps or "model" not in self.base_model.named_steps:
            raise KeyError("Expected Pipeline steps named 'preprocess' and 'model' for contributors.")

        preprocess = self.base_model.named_steps["preprocess"]
        model = self.base_model.named_steps["model"]

        if not hasattr(model, "coef_"):
            raise TypeError("Top contributors implemented for linear models with coef_ only.")

        # transformed data
        Z = preprocess.transform(X)
        if hasattr(Z, "toarray"):  # sparse
            Z = Z.toarray()
        Z = np.asarray(Z).ravel()

        coefs = np.asarray(model.coef_).ravel()
        feat_names = preprocess.get_feature_names_out()

        if len(Z) != len(coefs) or len(coefs) != len(feat_names):
            raise ValueError("Mismatch among transformed features, coefficients, and feature names.")

        contrib = Z * coefs
        df = pd.DataFrame({"feature": feat_names, "contribution": contrib})

        frozen = list(self.feature_cols)

        def map_to_original(feat: str) -> str:
            # e.g. "num__age" -> "age"; "cat__gender_Female" -> "gender"
            core = feat.split("__", 1)[-1]

            if core in frozen:
                return core

            # for one-hot columns: find the longest original col that is a prefix
            matches = [c for c in frozen if core.startswith(c + "_")]
            if matches:
                return max(matches, key=len)

            return core

        df["base_feature"] = df["feature"].apply(map_to_original)

        agg = (
            df.groupby("base_feature", as_index=False)
              .agg(total_contribution=("contribution", "sum"))
        )
        agg["abs_contribution"] = agg["total_contribution"].abs()
        agg = agg.sort_values("abs_contribution", ascending=False).head(int(top_k)).reset_index(drop=True)

        return agg[["base_feature", "total_contribution"]]
