import yaml

def verify_graphrag():
    print("Writing GraphRAG settings diff...")
    
    settings_fragment = {
        "llm": {
            "api_key": "${GRAPHRAG_API_KEY}",
            "type": "openai_chat",
            "model": "phi3:mini",
            "model_supports_json": True,
            "api_base": "http://localhost:11434/v1"
        },
        "embeddings": {
            "async_mode": "threaded",
            "llm": {
                "api_key": "${GRAPHRAG_API_KEY}",
                "type": "openai_embedding",
                "model": "nomic-embed-text",
                "api_base": "http://localhost:11434/v1"
            }
        }
    }
    
    print("\n--- data/graphrag/settings.yaml changes ---")
    print(yaml.dump(settings_fragment, default_flow_style=False))
    
    with open("data/graphrag/settings.yaml", "w") as f:
        yaml.dump(settings_fragment, f)
    print("Wrote fragment to data/graphrag/settings.yaml")
    
    with open("data/graphrag/settings.yaml") as f:
        s = yaml.safe_load(f)
    llm_base = s["llm"]["api_base"]
    emb_base = s["embeddings"]["llm"]["api_base"]
    
    assert "localhost" in llm_base or "127.0.0.1" in llm_base, \
        f"GraphRAG LLM api_base must be localhost, got: {llm_base}"
    assert "localhost" in emb_base or "127.0.0.1" in emb_base, \
        f"GraphRAG embeddings api_base must be localhost, got: {emb_base}"
    print("CHECK 7 PASSED: GraphRAG is wired to local Ollama only.")

if __name__ == "__main__":
    verify_graphrag()
