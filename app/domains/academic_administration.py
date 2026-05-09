import json
import re
from app.llm import ollama_generate 


VALID_DOCUMENT_TYPES = {
    "academic_calendar",
    "academic_regulation",
    "academic_guide",
    "general_document"
}

VALID_ACADEMIC_YEARS = {
    "2026/2027",
    "2024/2025",
    "general"
}

VALID_TOPICS = {
    "academic_calendar",
    "administrasi_akademik",
    "kurikulum",
    "pedoman_pendidikan",
    "penilaian_hasil_belajar",
    "general"
}

VALID_QUERY_INTENTS = {
    "date_schedule",
    "rule_policy",
    "procedure",
    "general_info"
}


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    return json.loads(match.group(0))


def analyze_query(user_query: str) -> dict:
    prompt = f"""
You are a query analyzer for the academic_administration domain in a university RAG system.

Your task:
1. Detect safe metadata filters.
2. Detect rerank keywords.
3. Detect query intent.

Available metadata fields:
- document_type
- academic_year
- topic

Allowed document_type values:
- academic_calendar
- academic_regulation
- academic_guide
- general_document

Allowed academic_year values:
- 2026/2027
- 2024/2025
- general

Allowed topic values:
- academic_calendar
- administrasi_akademik
- kurikulum
- pedoman_pendidikan
- penilaian_hasil_belajar
- general

Rules:
- For schedule/date questions, use document_type "academic_calendar".
- For rules/policy questions, use document_type "academic_regulation" or "academic_guide".
- If the query mentions 2026/2027 or 2026-2027, use academic_year "2026/2027".
- If the query mentions 2024/2025 or 2024-2025, use academic_year "2024/2025".
- If the query is about general rules, use academic_year "general".
- For specific schedule activities such as KRS, UKT, KHS, yudisium, masa perkuliahan, or ujian, use topic "academic_calendar".
- Specific activity words must go into rerank_keywords, not topic.
- Do not invent new metadata values.
- Output must be valid JSON only.

Output schema:
{{
  "domain": "academic_administration",
  "query_intent": "date_schedule | rule_policy | procedure | general_info",
  "metadata_filters": {{
    "document_type": "value_or_empty_string",
    "academic_year": "value_or_empty_string",
    "topic": "value_or_empty_string"
  }},
  "rerank_keywords": ["keyword_1", "keyword_2"],
  "reason": "short reason"
}}

Examples:

User question:
"Kapan registrasi akademik KRS Online semester gasal 2026/2027?"

Output:
{{
  "domain": "academic_administration",
  "query_intent": "date_schedule",
  "metadata_filters": {{
    "document_type": "academic_calendar",
    "academic_year": "2026/2027",
    "topic": "academic_calendar"
  }},
  "rerank_keywords": ["registrasi akademik", "KRS Online", "semester gasal", "2026/2027"],
  "reason": "The question asks for an academic calendar schedule."
}}

User question:
"Apa aturan penilaian hasil belajar mahasiswa?"

Output:
{{
  "domain": "academic_administration",
  "query_intent": "rule_policy",
  "metadata_filters": {{
    "document_type": "academic_regulation",
    "academic_year": "general",
    "topic": "penilaian_hasil_belajar"
  }},
  "rerank_keywords": ["penilaian", "hasil belajar", "nilai mahasiswa"],
  "reason": "The question asks about assessment rules."
}}

User question:
"Bagaimana aturan cuti kuliah mahasiswa?"

Output:
{{
  "domain": "academic_administration",
  "query_intent": "rule_policy",
  "metadata_filters": {{
    "document_type": "academic_regulation",
    "academic_year": "general",
    "topic": "administrasi_akademik"
  }},
  "rerank_keywords": ["cuti kuliah", "permohonan cuti", "SKCK"],
  "reason": "The question asks about academic leave rules."
}}

Now analyze the question.

User question:
{user_query}

Output:
"""

    response = ollama_generate(prompt)

    try:
        result = extract_json(response)
    except json.JSONDecodeError:
        return {
            "domain": "academic_administration",
            "query_intent": "general_info",
            "metadata_filters": {},
            "rerank_keywords": [],
            "reason": "Failed to parse analyzer output.",
            "raw_response": response
        }

    query_intent = result.get("query_intent", "general_info")
    if query_intent not in VALID_QUERY_INTENTS:
        query_intent = "general_info"

    metadata_filters = result.get("metadata_filters", {})
    if not isinstance(metadata_filters, dict):
        metadata_filters = {}

    clean_metadata_filters = {}

    document_type = metadata_filters.get("document_type", "")
    academic_year = metadata_filters.get("academic_year", "")
    topic = metadata_filters.get("topic", "")

    if document_type in VALID_DOCUMENT_TYPES:
        clean_metadata_filters["document_type"] = document_type

    if academic_year in VALID_ACADEMIC_YEARS:
        clean_metadata_filters["academic_year"] = academic_year

    if topic in VALID_TOPICS:
        clean_metadata_filters["topic"] = topic

    rerank_keywords = result.get("rerank_keywords", [])
    if not isinstance(rerank_keywords, list):
        rerank_keywords = []

    rerank_keywords = [
        str(keyword).strip()
        for keyword in rerank_keywords
        if str(keyword).strip()
    ]

    return {
        "domain": "academic_administration",
        "query_intent": query_intent,
        "metadata_filters": clean_metadata_filters,
        "rerank_keywords": rerank_keywords,
        "reason": result.get("reason", "")
    }


def build_filter(analysis: dict) -> dict:
    filters = [
        {"domain": "academic_administration"}
    ]

    metadata_filters = analysis.get("metadata_filters", {})

    for key in ["document_type", "academic_year", "topic"]:
        value = metadata_filters.get(key)

        if value:
            filters.append({key: value})

    if len(filters) == 1:
        return filters[0]

    return {
        "$and": filters
    }


def rerank(results, rerank_keywords: list, query: str = ""):
    query_lower = query.lower()

    strong_phrases = [
        "registrasi akademik (krs online)",
        "registrasi akademik krs online",
        "krs online",
        "pembayaran ukt",
        "registrasi administrasi/pembayaran ukt",
        "masa perkuliahan",
        "kartu hasil studi",
        "khs online",
        "batas akhir yudisium",
        "yudisium di fakultas",
    ]

    negative_phrases = [
        "krs oleh operator fakultas",
        "cakra widya",
        "semester antara",
        "kartu rencana ekstrakurikuler",
        "kre online",
    ]

    reranked = []

    for doc, distance in results:
        text = doc.page_content.lower()
        bonus = 0
        penalty = 0

        for keyword in rerank_keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower and keyword_lower in text:
                bonus += 1

        for phrase in strong_phrases:
            if phrase in query_lower and phrase in text:
                bonus += 5

        if "krs online" in query_lower:
            if "registrasi akademik (krs online)" in text:
                bonus += 10
            if "registrasi akademik" in text and "krs online" in text:
                bonus += 5

        for phrase in negative_phrases:
            if phrase in text:
                penalty += 2

        reranked.append({
            "doc": doc,
            "distance": distance,
            "keyword_bonus": bonus,
            "penalty": penalty,
            "final_score": bonus - penalty,
            "sort_key": (-(bonus - penalty), distance)
        })

    return sorted(reranked, key=lambda item: item["sort_key"])
