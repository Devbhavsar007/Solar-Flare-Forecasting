"""
GraphRAG context retriever for DSPy reporter.

Executes the GraphRAG CLI via subprocess to retrieve relevant RAG context
for the LLM. If the CLI fails or times out, it silently returns an empty
string, allowing the LLM reporter to proceed without context rather than
crashing the pipeline.
"""
import subprocess


def retrieve_flare_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve domain context using GraphRAG CLI.

    Args:
        query: Natural language query to search the knowledge graph.
        top_k: Number of contexts to retrieve (not strictly enforced by the CLI here,
               but used conceptually).

    Returns:
        String containing the retrieved context, or an empty string if retrieval fails.
    """
    cmd = [
        "graphrag",
        "query",
        "--root",
        "data/graphrag",
        "--query",
        query,
    ]
    try:
        # Run with a 30-second timeout to prevent stalling the real-time pipeline.
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"GraphRAG error: {result.stderr}")
            return ""
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"GraphRAG retrieval failed: {e}")
        return ""
