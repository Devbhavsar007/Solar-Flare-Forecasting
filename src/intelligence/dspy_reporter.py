"""
DSPy LLM alert reporter using local Ollama (Phi-3-mini).

Generates natural language alerts for significant solar flare detections.
Adheres to [RULE-16] for a module-level singleton LM loader, and
[RULE-10] by using `dspy.context()` context managers per-call rather than
global configuration, ensuring thread-safe concurrency in the pipeline.
"""
import warnings

try:
    import dspy
except ImportError:
    dspy = None


# Module-level singleton [RULE-16]
_LM = None

def get_lm():
    """
    Return the singleton Ollama LM instance.
    Initializes on first call to prevent blocking module import.
    Uses syntax verified by CHECK 5.
    """
    global _LM
    if _LM is None and dspy is not None:
        # CHECK 5: DSPy syntax for Ollama
        _LM = dspy.LM(
            model='ollama/phi3:mini',
            api_base='http://localhost:11434',
            api_key=''
        )
    return _LM


class FlareAlertSignature(dspy.Signature):
    """Generate a structured, natural language solar flare alert report."""
    
    flare_class = dspy.InputField(desc="The predicted GOES class (e.g., M, X)")
    peak_flux = dspy.InputField(desc="The predicted peak flux in W/m^2")
    lead_time_min = dspy.InputField(desc="Estimated time until peak flux in minutes")
    confidence = dspy.InputField(desc="Model confidence score [0.0, 1.0]")
    graphrag_context = dspy.InputField(desc="Relevant scientific context from the GraphRAG knowledge base")
    uncertainty_q10 = dspy.InputField(desc="Lower bound (10th percentile) flux estimate")
    uncertainty_q90 = dspy.InputField(desc="Upper bound (90th percentile) flux estimate")
    
    alert_summary = dspy.OutputField(desc="A concise, actionable, and scientific natural language alert report")


class SolarFlareReporter(dspy.Module):
    """
    DSPy module for generating natural language flare alerts.
    """
    def __init__(self):
        super().__init__()
        # Use simple Predict for zero-shot, or ChainOfThought for more reasoning
        if dspy is not None:
            self.predict = dspy.Predict(FlareAlertSignature)
    
    def forward(
        self,
        flare_class: str,
        peak_flux: float,
        lead_time_min: int,
        confidence: float,
        graphrag_context: str,
        uncertainty_q10: float,
        uncertainty_q90: float,
    ) -> str:
        """
        Generate the alert report.
        
        Enforces [RULE-10] by using `with dspy.context(lm=get_lm()):`.
        """
        if dspy is None:
            raise ImportError("dspy is not installed.")
            
        lm = get_lm()
        if lm is None:
            raise RuntimeError("Failed to initialize dspy LM.")

        with dspy.context(lm=lm):  # [RULE-10] Per-call context manager
            prediction = self.predict(
                flare_class=str(flare_class),
                peak_flux=str(peak_flux),
                lead_time_min=str(lead_time_min),
                confidence=str(confidence),
                graphrag_context=graphrag_context,
                uncertainty_q10=str(uncertainty_q10),
                uncertainty_q90=str(uncertainty_q90),
            )
            return prediction.alert_summary


def optimize_reporter(training_examples: list, metric_fn, num_threads: int = 4):
    """
    Optimize the reporter using DSPy MIPROv2.
    """
    if dspy is None:
        raise ImportError("dspy is not installed.")
        
    lm = get_lm()
    reporter = SolarFlareReporter()
    
    with dspy.context(lm=lm):
        optimizer = dspy.MIPROv2(metric=metric_fn)
        compiled_reporter = optimizer.compile(
            reporter,
            trainset=training_examples,
            num_threads=num_threads
        )
        
    # Save the optimized module
    import os
    os.makedirs("models", exist_ok=True)
    compiled_reporter.save("models/dspy_reporter_optimised.json")
    
    return compiled_reporter
