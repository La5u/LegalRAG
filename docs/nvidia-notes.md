# NVIDIA Notes

The current MVP uses Ollama, sentence-transformers, Neo4j, pdfium, and
pdfplumber. That stack is simpler and easier to run during prototype work.

A future production deployment could evaluate NVIDIA tooling for higher
throughput:

- Triton or TensorRT-LLM for batched LLM serving.
- TensorRT-optimized embedding models.
- GPU vector search through cuVS or a managed vector service.
- Optional RAPIDS tools for batch preprocessing.

This is not part of the current MVP implementation. Revisit only after the
retrieval quality and legal reasoning behavior are stable.
