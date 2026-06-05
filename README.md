Electronics AI Vision & Datasheet Assistant


1. The Problem :
      Engineers waste hours hunting through dense, unstructured PDF component datasheets to manually find critical parameters like forward voltages, pinouts, and maximum current limits. This manual process slows down hardware development cycles and increases the risk of component selection errors.

2. The Solution :
      I built a localized, full-stack edge application that uses real-time computer vision to identify physical PCB components while running an integrated Retrieval-Augmented Generation (RAG) pipeline. By combining object detection with local text intelligence, engineers can point a camera at a component and instantly query its specific technical datasheet using natural language with zero data latency or cloud dependencies.


3. Tech Stack & Architecture :
    * Frontend UI: Streamlit (engineered with state-preserving fragments to isolate fast-refresh camera loops from LLM invocation bottlenecks).
    
    *  AI Orchestration Framework: LangChain (utilizing advanced text splitting and custom prompt layouts).
    
    *  Vector Database: FAISS (Facebook AI Similarity Search) for high-performance, in-memory structural semantic search.
    
    *  Local Models: Meta's Llama 3 (via Ollama) for intelligent technical analysis, paired with dedicated text embeddings (nomic-embed-text).
    
    *  Computer Vision: Ultralytics YOLOv11 for lightweight, high-frame-rate real-time object tracking.
