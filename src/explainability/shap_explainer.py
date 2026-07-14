"""
SHAP explanations for XGBoost nowcasting decisions.

Provides feature importance rankings to build human trust and feed
context into the DSPy reporter module.
"""
import numpy as np

try:
    import shap
except ImportError:
    shap = None


class SHAPExplainer:
    def __init__(self, xgb_model, feature_names: list[str]):
        """
        Initialize the SHAP explainer for an XGBoost model.
        """
        if shap is None:
            raise ImportError("shap is not installed. Run: pip install shap")
            
        self.feature_names = feature_names
        # Extract base estimator if model is a CalibratedClassifierCV
        base_model = getattr(xgb_model, 'estimator', xgb_model)
        self.explainer = shap.TreeExplainer(base_model)

    def explain(self, combined_features: np.ndarray) -> dict:
        """
        Generate SHAP explanations for a single instance.
        
        Args:
            combined_features: 2D array of shape (1, n_features).
            
        Returns:
            Dictionary with explanation components:
            - top_features: list of (feature_name, shap_value) for top 5
            - class_importances: per-class mean absolute SHAP over features
            - dominant_class: integer index of the class with highest total |SHAP|
            - raw_shap: raw SHAP values list
        """
        # SHAP values for multiclass returns a list of arrays (one per class)
        # Each array is of shape (n_instances, n_features)
        shap_values = self.explainer.shap_values(combined_features)
        
        # Calculate mean absolute SHAP for each class
        # shap_values is a list of length n_classes.
        # For a single instance, each item is shape (1, n_features) or (n_features,)
        class_importances = []
        for class_shap in shap_values:
            class_importances.append(np.sum(np.abs(class_shap)))
            
        # The class with the highest total |SHAP| is the dominant explanatory class
        dominant_class = int(np.argmax(class_importances))
        
        # Get the SHAP values for the dominant class for this single instance
        dominant_shap = shap_values[dominant_class]
        if dominant_shap.ndim == 2:
            dominant_shap = dominant_shap[0]
            
        # Get top 5 features by absolute SHAP value
        feature_indices = np.argsort(np.abs(dominant_shap))[::-1][:5]
        
        top_features = [
            (self.feature_names[i], float(dominant_shap[i])) 
            for i in feature_indices
        ]
        
        return {
            "top_features": top_features,
            "class_importances": class_importances,
            "dominant_class": dominant_class,
            "raw_shap": shap_values
        }

    def generate_report_text(self, explain_result: dict, predicted_class: int | str) -> str:
        """
        Generate a human-readable summary of the SHAP explanation.
        This is fed as input context for the DSPy reporter.
        """
        top_feats = explain_result["top_features"]
        
        lines = [f"The model predicted class {predicted_class}."]
        lines.append("The top 5 most important features driving this prediction are:")
        
        for idx, (feat_name, val) in enumerate(top_feats, 1):
            direction = "increasing" if val > 0 else "decreasing"
            lines.append(f"  {idx}. {feat_name} ({direction} probability, impact: {abs(val):.4f})")
            
        return "\n".join(lines)
