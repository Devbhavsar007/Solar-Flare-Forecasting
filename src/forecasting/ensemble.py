import torch
import numpy as np

def _flux_to_probs(peak_flux: float) -> np.ndarray:
    """Helper to convert a point forecast flux to pseudo-probabilities [N, C, M, X]."""
    # Simple threshold-based one-hot mapping for demonstration
    if peak_flux >= 1e-4:
        idx = 3 # X
    elif peak_flux >= 1e-5:
        idx = 2 # M
    elif peak_flux >= 1e-6:
        idx = 1 # C
    else:
        idx = 0 # N
        
    prob = np.zeros(4)
    prob[idx] = 1.0
    return prob

class ThreeModelEnsemble:
    """
    Ensemble of CausalLSTM and MultiHorizon TCN (TimesFM dropped per D11).
    Kept the class name 'ThreeModelEnsemble' for backwards compatibility,
    but it now strictly operates as a Two-Model ensemble.
    """
    def __init__(self, lstm_model, tcn_model, timesfm_model=None, weights=None):
        self.lstm_model = lstm_model
        self.tcn_model = tcn_model
        
        # We completely ignore timesfm_model if passed
        if weights is None or len(weights) != 2:
            self.weights = (0.50, 0.50)
        else:
            self.weights = weights

    def predict_single(self, X_tensor: torch.Tensor, flux_np: np.ndarray = None, horizon: int = 15) -> np.ndarray:
        """
        Weighted average of model probabilities.
        
        Args:
            X_tensor: (1, T, F) tensor for LSTM and TCN.
            flux_np: (T,) array of flux values (ignored, kept for signature compatibility).
            horizon: Target horizon in minutes.
            
        Returns:
            (4,) array of ensembled class probabilities.
        """
        self.lstm_model.eval()
        self.tcn_model.eval()
        
        with torch.no_grad():
            lstm_prob = self.lstm_model(X_tensor).cpu().numpy()[0]
            
            tcn_out = self.tcn_model(X_tensor)
            tcn_prob = tcn_out[f"h{horizon}"].cpu().numpy()[0]
            
        w1, w2 = self.weights
        ensemble_prob = w1 * lstm_prob + w2 * tcn_prob
            
        return ensemble_prob / np.sum(ensemble_prob)
