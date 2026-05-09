def build_prompt(user_query: str, context: str) -> str:
    return f"""
You are an information extraction assistant for a university RAG system.

Your task is to answer the user's rule or policy question using ONLY the provided context.

Rules:
- Answer in Indonesian.
- Summarize only rules, requirements, or provisions found in the context.
- Do not use outside knowledge.
- Do not invent requirements.
- If the context contains relevant rules, answer them in concise bullet points.
- Do not include source reference. Source will be added automatically.
- Do not put source references inside bullet points.
- Maximum 8 bullet points.
- If no relevant rule is present, answer:
  "Informasi tersebut tidak ditemukan secara jelas dalam dokumen yang tersedia."

Answer format:
- First sentence: direct answer.
- Then concise bullet points if needed.

User question:
{user_query}

Context:
{context}

Answer:
"""