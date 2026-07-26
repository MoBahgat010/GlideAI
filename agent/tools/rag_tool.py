from langchain_core.tools import tool
import sys
from pathlib import Path


@tool
def query_knowledge_base(query: str) -> str:
    """Use this tool to search the uploaded documents and knowledge base for answers."""
    # Ensure app can be imported
    app_dir = Path(__file__).resolve().parent.parent.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
        
    from app import get_retrieval_pipeline, _format_source
        
    pipeline = get_retrieval_pipeline()
    result = pipeline.retrieve(query, progress=lambda stage, msg: print(f"[RAG] {stage}: {msg}"))
    
    if not result.get("results"):
        return "No relevant information found in the knowledge base."
    
    context_blocks = "\n\n".join(
        f"**[{i+1}]** (score {r.get('rerank_score', 0):.3f}, "
        f"{_format_source(r)})\n{r.get('chunk_text', '')}"
        for i, r in enumerate(result["results"])
    )
    
    return f"Retrieved Context:\n{context_blocks}"

