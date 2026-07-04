import dspy

def verify_dspy():
    print("Testing DSPy + Ollama syntax...")
    syntaxes = [
        ("dspy.LM('ollama/phi3:mini', api_base='http://localhost:11434')", lambda: dspy.LM("ollama/phi3:mini", api_base="http://localhost:11434")),
        ("dspy.LM('ollama_chat/phi3:mini', api_base='http://localhost:11434')", lambda: dspy.LM("ollama_chat/phi3:mini", api_base="http://localhost:11434")),
        ("dspy.OllamaLocal('phi3:mini')", lambda: dspy.OllamaLocal("phi3:mini"))
    ]
    
    working_lm = None
    working_syntax = ""
    for name, func in syntaxes:
        try:
            lm = func()
            print(f"SUCCESS: {name}")
            working_lm = lm
            working_syntax = name
            break
        except Exception as e:
            print(f"FAILED: {name} - {e}")
            
    if working_lm:
        print("\nTesting per-call context manager:")
        try:
            with dspy.context(lm=working_lm):
                result = dspy.Predict("q->a")(q="test")
                print("Context manager works!")
        except Exception as e:
            print(f"Context manager failed: {e}")
    else:
        print("No working DSPy syntax found for Ollama.")

if __name__ == "__main__":
    verify_dspy()
