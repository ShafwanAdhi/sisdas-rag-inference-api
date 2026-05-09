import json
import re
from app.llm import ollama_generate


DOMAIN = "finance_tuition_and_scholarship"


VALID_DOCUMENT_TYPES = {
    "finance_announcement",
    "finance_guide",
    "finance_policy",
    "registration_announcement",
    "scholarship_announcement",
}

VALID_ACADEMIC_YEARS = {
    "2020/2021",
    "2021/2022",
    "2024/2025",
    "2025/2026",
    "general",
}

VALID_DOCUMENT_YEARS = {
    "2020",
    "2021",
    "2024",
    "2025",
    "general",
}

VALID_TOPICS = {
    "bank_indonesia_scholarship",
    "kip_registration",
    "new_student_registration",
    "semester_registration",
    "ukt_and_ipi_recommendation",
    "ukt_application_guide",
    "ukt_assistance",
}

VALID_QUERY_INTENTS = {
    "payment_schedule",
    "scholarship_info",
    "registration_info",
    "procedure",
    "rule_policy",
    "general_info",
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
You are a query analyzer for the finance_tuition_and_scholarship domain in a university RAG system.

Your task:
1. Detect safe metadata filters.
2. Detect rerank keywords.
3. Detect query intent.

This domain contains documents about:
- tuition fees / UKT
- IPI
- UKT payment
- student registration related to payment
- new student registration payment
- scholarship / beasiswa
- Bank Indonesia scholarship
- KIP registration
- UKT assistance / bantuan UKT
- UKT application guide
- semester registration announcements

Available metadata fields:
- document_type
- academic_year
- document_year
- topic

Allowed document_type values:
- finance_announcement
- finance_guide
- finance_policy
- registration_announcement
- scholarship_announcement

Allowed academic_year values:
- 2020/2021
- 2021/2022
- 2024/2025
- 2025/2026
- general

Allowed document_year values:
- 2020
- 2021
- 2024
- 2025
- general

Allowed topic values:
- bank_indonesia_scholarship
- kip_registration
- new_student_registration
- semester_registration
- ukt_and_ipi_recommendation
- ukt_application_guide
- ukt_assistance

Query intent values:
- payment_schedule
- scholarship_info
- registration_info
- procedure
- rule_policy
- general_info

Rules:
- This domain has relatively few chunks, so do not over-filter.
- Use metadata filters only when clearly implied by the question.
- If unsure, leave metadata value as an empty string.
- For questions about UKT payment deadline, registration payment, or semester registration, use topic "semester_registration" when the query refers to semester registration.
- For questions about new student registration, SNBT, UTBK, or maba, use topic "new_student_registration".
- For questions about UKT assistance or bantuan UKT, use topic "ukt_assistance".
- For questions about UKT application guide, application submission, or pengajuan UKT, use topic "ukt_application_guide".
- For questions about UKT and IPI recommendation, use topic "ukt_and_ipi_recommendation".
- For questions about Bank Indonesia scholarship or Beasiswa BI, use topic "bank_indonesia_scholarship".
- For questions about KIP or KIP-K, use topic "kip_registration".
- If the user mentions 2020/2021 or 2020-2021, use academic_year "2020/2021" and document_year "2020".
- If the user mentions 2021/2022 or 2021-2022, use academic_year "2021/2022" and document_year "2021".
- If the user mentions 2024/2025 or 2024-2025, use academic_year "2024/2025" and document_year "2024".
- If the user mentions 2025/2026 or 2025-2026, use academic_year "2025/2026" and document_year "2025".
- If the user mentions 2024 but not an academic year, use document_year "2024".
- If the user mentions 2025 but not an academic year, use document_year "2025".
- Specific words such as UKT, IPI, Beasiswa BI, KIP, registrasi, pembayaran, jadwal, and batas akhir must go into rerank_keywords.
- Do not invent new metadata values.
- Output must be valid JSON only.
- Do not include markdown.
- Do not explain outside JSON.

Output schema:
{{
  "domain": "finance_tuition_and_scholarship",
  "query_intent": "payment_schedule | scholarship_info | registration_info | procedure | rule_policy | general_info",
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
"Kapan batas pembayaran UKT semester gasal 2025/2026?"

Output:
{{
  "domain": "finance_tuition_and_scholarship",
  "query_intent": "payment_schedule",
  "metadata_filters": {{
    "document_type": "registration_announcement",
    "academic_year": "2025/2026",
    "document_year": "2025",
    "topic": "semester_registration"
  }},
  "rerank_keywords": ["pembayaran UKT", "semester gasal", "2025/2026", "registrasi mahasiswa"],
  "reason": "The question asks about the UKT payment or registration schedule for semester gasal 2025/2026."
}}

User question:
"Bagaimana cara mengajukan UKT?"

Output:
{{
  "domain": "finance_tuition_and_scholarship",
  "query_intent": "procedure",
  "metadata_filters": {{
    "document_type": "finance_guide",
    "academic_year": "",
    "document_year": "",
    "topic": "ukt_application_guide"
  }},
  "rerank_keywords": ["pengajuan UKT", "aplikasi pengajuan UKT", "panduan UKT"],
  "reason": "The question asks about the procedure for submitting a UKT application."
}}

User question:
"Apa informasi Beasiswa BI tahun 2024?"

Output:
{{
  "domain": "finance_tuition_and_scholarship",
  "query_intent": "scholarship_info",
  "metadata_filters": {{
    "document_type": "scholarship_announcement",
    "academic_year": "",
    "document_year": "2024",
    "topic": "bank_indonesia_scholarship"
  }},
  "rerank_keywords": ["Beasiswa BI", "Bank Indonesia", "tahun 2024", "beasiswa"],
  "reason": "The question asks about Bank Indonesia scholarship information in 2024."
}}

User question:
"Bagaimana informasi perpanjangan registrasi KIP-K?"

Output:
{{
  "domain": "finance_tuition_and_scholarship",
  "query_intent": "registration_info",
  "metadata_filters": {{
    "document_type": "scholarship_announcement",
    "academic_year": "",
    "document_year": "",
    "topic": "kip_registration"
  }},
  "rerank_keywords": ["KIP", "KIP-K", "perpanjangan registrasi", "registrasi KIP"],
  "reason": "The question asks about KIP registration extension."
}}

User question:
"Apa rekomendasi UKT dan IPI di UM?"

Output:
{{
  "domain": "finance_tuition_and_scholarship",
  "query_intent": "rule_policy",
  "metadata_filters": {{
    "document_type": "finance_policy",
    "academic_year": "general",
    "document_year": "",
    "topic": "ukt_and_ipi_recommendation"
  }},
  "rerank_keywords": ["rekomendasi UKT", "IPI", "UKT dan IPI"],
  "reason": "The question asks about UKT and IPI recommendation policy."
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
            "raw_response": response,
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
        "reason": result.get("reason", ""),
    }


def build_filter(analysis: dict, strict: bool = False) -> dict:
    """
    Filter dibuat ringan karena collection finance hanya sekitar 64 chunk.

    strict=False:
    - Selalu filter domain.
    - Filter topic kalau ada.
    - Filter academic_year/document_year kalau ada.
    - Tidak wajib filter document_type agar retrieval tidak terlalu sempit.

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
        for key in ["topic", "academic_year", "document_year"]:
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
        "pembayaran ukt",
        "batas pembayaran ukt",
        "registrasi administrasi",
        "registrasi mahasiswa",
        "semester gasal",
        "semester genap",
        "ukt",
        "ipi",
        "ukt dan ipi",
        "rekomendasi ukt",
        "pengajuan ukt",
        "aplikasi pengajuan ukt",
        "bantuan ukt",
        "beasiswa",
        "beasiswa bi",
        "bank indonesia",
        "kip",
        "kip-k",
        "registrasi kip",
        "perpanjangan registrasi",
        "maba",
        "mahasiswa baru",
        "utbk",
        "snbt",
    ]

    negative_phrases = [
        "daftar isi",
        "lampiran",
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

        # Bonus frasa kuat jika frasa itu muncul di query dan chunk
        for phrase in strong_phrases:
            if phrase in query_lower and phrase in text:
                bonus += 5

        topic = metadata.get("topic", "")
        document_type = metadata.get("document_type", "")
        academic_year = metadata.get("academic_year", "")
        document_year = metadata.get("document_year", "")

        # Bonus metadata berdasarkan query
        if "ukt" in query_lower:
            if topic in {
                "ukt_and_ipi_recommendation",
                "ukt_application_guide",
                "ukt_assistance",
                "semester_registration",
                "new_student_registration",
            }:
                bonus += 5

        if "ipi" in query_lower:
            if topic == "ukt_and_ipi_recommendation":
                bonus += 8

        if "pengajuan" in query_lower and "ukt" in query_lower:
            if topic == "ukt_application_guide":
                bonus += 10

        if "bantuan" in query_lower and "ukt" in query_lower:
            if topic == "ukt_assistance":
                bonus += 10

        if "beasiswa" in query_lower:
            if document_type == "scholarship_announcement":
                bonus += 6

        if "bi" in query_lower or "bank indonesia" in query_lower:
            if topic == "bank_indonesia_scholarship":
                bonus += 10

        if "kip" in query_lower or "kip-k" in query_lower:
            if topic == "kip_registration":
                bonus += 10

        if "maba" in query_lower or "mahasiswa baru" in query_lower or "snbt" in query_lower or "utbk" in query_lower:
            if topic == "new_student_registration":
                bonus += 10

        if "registrasi" in query_lower:
            if topic in {"semester_registration", "new_student_registration", "kip_registration"}:
                bonus += 5

        if "pembayaran" in query_lower or "bayar" in query_lower:
            if topic in {"semester_registration", "new_student_registration", "ukt_assistance"}:
                bonus += 5

        if "2020/2021" in query_lower or "2020-2021" in query_lower or "2020" in query_lower:
            if academic_year == "2020/2021" or document_year == "2020":
                bonus += 5

        if "2021/2022" in query_lower or "2021-2022" in query_lower or "2021" in query_lower:
            if academic_year == "2021/2022" or document_year == "2021":
                bonus += 5

        if "2024/2025" in query_lower or "2024-2025" in query_lower or "2024" in query_lower:
            if academic_year == "2024/2025" or document_year == "2024":
                bonus += 5

        if "2025/2026" in query_lower or "2025-2026" in query_lower or "2025" in query_lower:
            if academic_year == "2025/2026" or document_year == "2025":
                bonus += 5

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
            "sort_key": (-final_score, distance),
        })

    return sorted(reranked, key=lambda item: item["sort_key"])