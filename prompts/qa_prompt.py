"""Prompt template for context-based question answering."""

QA_PROMPT = (
    "Answer the question using only the context below. If the answer is not "
    "available in the context, say: The answer is not available in the provided "
    "context.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)
