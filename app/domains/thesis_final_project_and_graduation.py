import json
import re
from app.llm import ollama_generate


DOMAIN = "thesis_final_project_and_graduation"


VALID_DOCUMENT_TYPES = {
    "academic_guide_revision",
    "graduation_announcement",
    "graduation_letter",
    "graduation_regulation",
    "thesis_form",
    "thesis_guide",
    "thesis_sop",
}

VALID_ACADEMIC_YEARS = {
    "general"
}

VALID_DOCUMENT_YEARS = {
    "2018",
    "2022",
    "2025",
    "general"
}

VALID_TOPICS = {
    "graduation_ceremony",
    "graduation_requirement",
    "thesis_guidance_card",
    "thesis_supervision",
    "thesis_topic_approval",
    "thesis_writing_guideline",
    "yudisium_and_graduation",
}

VALID_QUERY_INTENTS = {
    "date_schedule",
    "rule_policy",
    "procedure",
    "form_requirement",
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
You are a query analyzer for the thesis_final_project_and_graduation domain in a university RAG system.

Your task:
1. Detect safe metadata filters.
2. Detect rerank keywords.
3. Detect query intent.

This domain contains documents about:
- thesis / skripsi
- final project / tugas akhir
- thesis supervision / pembimbingan skripsi
- thesis writing guideline / pedoman penulisan skripsi
- thesis topic approval / persetujuan tema
- thesis guidance card / kartu bimbingan
- yudisium
- graduation / wisuda
- graduation announcement / surat edaran wisuda
- graduation requirements

Available metadata fields:
- document_type
- academic_year
- document_year
- topic

Allowed document_type values:
- academic_guide_revision
- graduation_announcement
- graduation_letter
- graduation_regulation
- thesis_form
- thesis_guide
- thesis_sop

Allowed academic_year values:
- general

Allowed document_year values:
- 2018
- 2022
- 2025
- general

Allowed topic values:
- graduation_ceremony
- graduation_requirement
- thesis_guidance_card
- thesis_supervision
- thesis_topic_approval
- thesis_writing_guideline
- yudisium_and_graduation

Query intent values:
- date_schedule
- rule_policy
- procedure
- form_requirement
- general_info

Rules:
- This domain has relatively few chunks, so do not over-filter.
- Use metadata filters only when the user's question clearly implies them.
- If unsure, leave metadata value as an empty string.
- For questions about skripsi writing format, citation style, chapters, or writing rules, use topic "thesis_writing_guideline".
- For questions about thesis supervision, supervisor, bimbingan, or SOP pembimbingan, use topic "thesis_supervision".
- For questions about kartu bimbingan skripsi, use topic "thesis_guidance_card".
- For questions about lembar persetujuan tema or approval of thesis topic, use topic "thesis_topic_approval".
- For questions about yudisium and graduation rules, use topic "yudisium_and_graduation".
- For questions about graduation requirements, use topic "graduation_requirement".
- For questions about wisuda schedule, ceremony, announcement, or surat edaran wisuda 2025, use topic "graduation_ceremony" and document_year "2025".
- If the user mentions 2025, use document_year "2025".
- If the user mentions 2022, use document_year "2022".
- If the user mentions 2018, use document_year "2018".
- Do not invent new metadata values.
- Output must be valid JSON only.
- Do not include markdown.
- Do not explain outside JSON.

Output schema:
{{
  "domain": "thesis_final_project_and_graduation",
  "query_intent": "date_schedule | rule_policy | procedure | form_requirement | general_info",
  "metadata_filters": {{
    "document_type": "value_or_empty_string",
    "academic_year": "value_or_empty_string",
    "document_year": "value_or_empty_string",
    "topic": "value_or_empty_string"
  }},
  "rerank_keywords": ["keyword_1", "keyword_2"],
  "reason": "short reason"
}}

Examples:

User question:
"Bagaimana format penulisan skripsi?"

Output:
{{
  "domain": "thesis_final_project_and_graduation",
  "query_intent": "rule_policy",
  "metadata_filters": {{
    "document_type": "thesis_guide",
    "academic_year": "general",
    "document_year": "",
    "topic": "thesis_writing_guideline"
  }},
  "rerank_keywords": ["format penulisan skripsi", "pedoman penulisan skripsi", "skripsi"],
  "reason": "The question asks about thesis writing guidelines."
}}

User question:
"Bagaimana prosedur pembimbingan skripsi?"

Output:
{{
  "domain": "thesis_final_project_and_graduation",
  "query_intent": "procedure",
  "metadata_filters": {{
    "document_type": "thesis_sop",
    "academic_year": "general",
    "document_year": "",
    "topic": "thesis_supervision"
  }},
  "rerank_keywords": ["pembimbingan skripsi", "dosen pembimbing", "SOP pembimbingan"],
  "reason": "The question asks about thesis supervision procedure."
}}

User question:
"Apa fungsi kartu bimbingan skripsi?"

Output:
{{
  "domain": "thesis_final_project_and_graduation",
  "query_intent": "form_requirement",
  "metadata_filters": {{
    "document_type": "thesis_form",
    "academic_year": "general",
    "document_year": "",
    "topic": "thesis_guidance_card"
  }},
  "rerank_keywords": ["kartu bimbingan skripsi", "bimbingan skripsi"],
  "reason": "The question asks about the thesis guidance card form."
}}

User question:
"Apa syarat mengikuti wisuda tahun 2025?"

Output:
{{
  "domain": "thesis_final_project_and_graduation",
  "query_intent": "rule_policy",
  "metadata_filters": {{
    "document_type": "",
    "academic_year": "general",
    "document_year": "2025",
    "topic": "graduation_ceremony"
  }},
  "rerank_keywords": ["wisuda 2025", "syarat wisuda", "peserta wisuda"],
  "reason": "The question asks about graduation information in 2025."
}}

User question:
"Bagaimana aturan yudisium dan wisuda?"

Output:
{{
  "domain": "thesis_final_project_and_graduation",
  "query_intent": "rule_policy",
  "metadata_filters": {{
    "document_type": "graduation_regulation",
    "academic_year": "general",
    "document_year": "",
    "topic": "yudisium_and_graduation"
  }},
  "rerank_keywords": ["yudisium", "wisuda", "aturan yudisium", "aturan wisuda"],
  "reason": "The question asks about yudisium and graduation regulation."
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
            "domain": DOMAIN,
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
    document_year = metadata_filters.get("document_year", "")
    topic = metadata_filters.get("topic", "")

    if document_type in VALID_DOCUMENT_TYPES:
        clean_metadata_filters["document_type"] = document_type

    if academic_year in VALID_ACADEMIC_YEARS:
        clean_metadata_filters["academic_year"] = academic_year

    if document_year in VALID_DOCUMENT_YEARS:
        clean_metadata_filters["document_year"] = document_year

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
        "domain": DOMAIN,
        "query_intent": query_intent,
        "metadata_filters": clean_metadata_filters,
        "rerank_keywords": rerank_keywords,
        "reason": result.get("reason", "")
    }


def build_filter(analysis: dict, strict: bool = False) -> dict:
    """
    Filter dibuat ringan karena collection domain ini hanya sekitar 117 chunk.

    strict=False:
    - Selalu filter domain.
    - Filter topic kalau ada.
    - Filter document_year kalau ada.
    - Tidak wajib filter document_type supaya retrieval tidak terlalu sempit.

    strict=True:
    - Filter domain.
    - Filter document_type, academic_year, document_year, topic jika ada.
    """
    filters = [
        {"domain": DOMAIN}
    ]

    metadata_filters = analysis.get("metadata_filters", {})

    if strict:
        for key in ["document_type", "academic_year", "document_year", "topic"]:
            value = metadata_filters.get(key)
            if value:
                filters.append({key: value})
    else:
        # Light filtering: cukup domain + topic/year yang jelas
        for key in ["topic", "document_year"]:
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
        "pedoman penulisan skripsi",
        "penulisan skripsi",
        "format skripsi",
        "pembimbingan skripsi",
        "sop pembimbingan",
        "dosen pembimbing",
        "kartu bimbingan skripsi",
        "lembar persetujuan tema",
        "persetujuan tema",
        "tugas akhir",
        "ujian skripsi",
        "seminar proposal",
        "yudisium",
        "wisuda",
        "syarat wisuda",
        "surat edaran wisuda",
        "graduation",
    ]

    negative_phrases = [
        # sengaja ringan; domain ini kecil, jangan terlalu banyak penalti
        "lampiran",
        "daftar isi",
    ]

    reranked = []

    for doc, distance in results:
        text = doc.page_content.lower()
        metadata = doc.metadata

        bonus = 0
        penalty = 0

        # Bonus keyword dari LLM
        for keyword in rerank_keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower and keyword_lower in text:
                bonus += 2

        # Bonus frasa kuat
        for phrase in strong_phrases:
            if phrase in query_lower and phrase in text:
                bonus += 5

        # Bonus metadata topic kalau cocok dengan query
        topic = metadata.get("topic", "")

        if "skripsi" in query_lower:
            if topic in {
                "thesis_writing_guideline",
                "thesis_supervision",
                "thesis_guidance_card",
                "thesis_topic_approval",
            }:
                bonus += 2

        if "bimbing" in query_lower or "pembimbing" in query_lower:
            if topic == "thesis_supervision":
                bonus += 5

        if "kartu bimbingan" in query_lower:
            if topic == "thesis_guidance_card":
                bonus += 8

        if "tema" in query_lower or "persetujuan tema" in query_lower:
            if topic == "thesis_topic_approval":
                bonus += 8

        if "yudisium" in query_lower:
            if topic == "yudisium_and_graduation":
                bonus += 8

        if "wisuda" in query_lower:
            if topic in {"graduation_ceremony", "yudisium_and_graduation", "graduation_requirement"}:
                bonus += 5

        if "2025" in query_lower:
            if metadata.get("document_year") == "2025":
                bonus += 5

        # Penalti kecil untuk halaman yang biasanya tidak substantif
        for phrase in negative_phrases:
            if phrase in text:
                penalty += 1

        final_score = bonus - penalty

        reranked.append({
            "doc": doc,
            "distance": distance,
            "keyword_bonus": bonus,
            "penalty": penalty,
            "final_score": final_score,
            "sort_key": (-final_score, distance)
        })

    return sorted(reranked, key=lambda item: item["sort_key"])