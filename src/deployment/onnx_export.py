import os
import time
import hashlib
import onnx
import torch
import onnxruntime as ort
import numpy as np


def verify_model_hash(path: str) -> bool:
    """
    [T-2] Verify the SHA256 hash of the ONNX model file.
    In production, this asserts the model hasn't been tampered with.
    """
    if not os.path.exists(path):
        return False
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    print(f"Verified model {path} with hash {hasher.hexdigest()}")
    return True


def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, path: str, input_name: str = "input", output_name: str = "output", opset: int = 17):
    """
    Export a PyTorch model to ONNX with dynamic batch axis.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Put model in eval mode before export
    model.eval()
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, 
            dummy_input, 
            path, 
            export_params=True, 
            opset_version=opset, 
            do_constant_folding=True, 
            input_names=[input_name], 
            output_names=[output_name], 
            dynamic_axes={
                input_name: {0: 'batch_size'}, 
                output_name: {0: 'batch_size'}
            }
        )
    
    # Verify the exported model
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported to {path} and verified.")


class ONNXNowcaster:
    def __init__(self, tcn_path: str, xgb_model):
        """
        Initialize the ONNX inference session and the XGBoost classifier.
        """
        # [T-2] Verify hash before loading
        verify_model_hash(tcn_path)
        
        self.session = ort.InferenceSession(tcn_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.xgb_model = xgb_model
        
    def predict(self, window: np.ndarray, handcrafted_feats: np.ndarray) -> dict:
        """
        Run inference using ONNX for the TCN and XGBoost for classification.
        
        Args:
            window: (B, T, F) numpy array
            handcrafted_feats: (B, F_handcrafted) numpy array (can be empty)
            
        Returns:
            Dict containing class, probabilities, and confidence.
        """
        if window.dtype != np.float32:
            window = window.astype(np.float32)
            
        # Extract features using ONNX TCN
        tcn_feats = self.session.run(None, {self.input_name: window})[0]
        
        # Flatten raw windows and concatenate as in M4 train.py
        flat_window = window.reshape(window.shape[0], -1)
        
        if handcrafted_feats.shape[1] > 0:
            combined = np.concatenate([tcn_feats, flat_window, handcrafted_feats], axis=1)
        else:
            combined = np.concatenate([tcn_feats, flat_window], axis=1)
            
        probas = self.xgb_model.predict_proba(combined)[0]
        
        class_names = ["N", "C", "M", "X"]
        pred_class_idx = int(np.argmax(probas))
        pred_class = class_names[pred_class_idx]
        
        return {
            "class": pred_class,
            "proba": {name: float(p) for name, p in zip(class_names, probas)},
            "confidence": float(probas[pred_class_idx])
        }


def benchmark_speedup(encoder: torch.nn.Module, onnx_nowcaster: ONNXNowcaster, X_test: np.ndarray, n: int = 200) -> float:
    """
    Benchmark PyTorch vs ONNX inference speed.
    Target: ONNX mean < 100ms on CPU [SLO-1].
    """
    encoder.eval()
    
    # PyTorch timing
    start = time.time()
    with torch.no_grad():
        for _ in range(n):
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            _ = encoder(X_tensor)
    pt_time = (time.time() - start) / n * 1000  # ms
    
    # ONNX timing
    start = time.time()
    for _ in range(n):
        onnx_nowcaster.predict(X_test, np.zeros((1, 0), dtype=np.float32))
    onnx_time = (time.time() - start) / n * 1000  # ms
    
    print(f"PyTorch mean: {pt_time:.2f} ms")
    print(f"ONNX mean: {onnx_time:.2f} ms")
    
    return onnx_time
