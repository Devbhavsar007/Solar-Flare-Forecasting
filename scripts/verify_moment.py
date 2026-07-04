import time
import torch

def verify_moment():
    try:
        from momentfm import MOMENTPipeline
        
        print("Loading AutonLab/MOMENT-1-large...")
        t0 = time.time()
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={"task_name": "reconstruction"}
        )
        model.init()
        t1 = time.time()
        print(f"MOMENT load time: {t1 - t0:.2f}s")
        
        input_tensor = torch.rand(2, 1, 512)
        print("Passing input_tensor to MOMENT...")
        output = model(x_enc=input_tensor)
        
        attrs = [attr for attr in dir(output) if not attr.startswith("__")]
        print(f"Output attributes: {attrs}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_moment()
