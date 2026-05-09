import json
import re
from app.llm import ollama_generate


DOMAIN = "facilities_and_campus_services"


VALID_DOCUMENT_TYPES = {
    "facility_info",
    "facility_sop",
    "facility_fee_info",
}

VALID_ACADEMIC_YEARS = {
    "general"
}

VALID_DOCUMENT_YEARS = {
    "general"
}

VALID_TOPICS = {
    "library_facilities",
    "general_facilities",
    "graphic_studio",
    "sociology_laboratory",
    "laboratory",
    "recording_media",
    "painting_studio",
    "music_studio",
    "dance_studio",
    "sociology_laboratory_fee",
}

VALID_QUERY_INTENTS = {
    "facility_info",
    "procedure",
    "fee_info",
    "rule_policy",
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
You are a query analyzer for the facilities_and_campus_services domain in a university RAG system.

Your task:
1. Detect safe metadata filters.
2. Detect rerank keywords.
3. Detect query intent.

This domain contains documents about:
- campus facilities
- public facilities / sarana umum
- library facilities / perpustakaan
- laboratory SOP
- studio SOP
- graphic studio
- sociology laboratory
- music studio
- dance studio
- painting studio
- recording media
- laboratory equipment usage fees

Available metadata fields:
- document_type
- academic_year
- document_year
- topic

Allowed document_type values:
- facility_info
- facility_sop
- facility_fee_info

Allowed academic_year values:
- general

Allowed document_year values:
- general

Allowed topic values:
- library_facilities
- general_facilities
- graphic_studio
- sociology_laboratory
- laboratory
- recording_media
- painting_studio
- music_studio
- dance_studio
- sociology_laboratory_fee

Query intent values:
- facility_info
- procedure
- fee_info
- rule_policy
- general_info

Rules:
- This domain has relatively few chunks, so do not over-filter.
- Use metadata filters only when clearly implied by the question.
- If unsure, leave metadata value as an empty string.
- For questions about library, perpustakaan, reading room, library facilities, use topic "library_facilities".
- For questions about general campus facilities, public facilities, sarana umum, use topic "general_facilities".
- For questions about SOP, procedure, rules, or usage flow of a facility, use document_type "facility_sop".
- For questions about fees, tariffs, costs, payment, or lab equipment rental, use document_type "facility_fee_info" and topic "sociology_laboratory_fee" if related to sociology lab.
- For questions about sociology laboratory or lab sosio, use topic "sociology_laboratory".
- For questions about general laboratory, use topic "laboratory".
- For questions about graphic studio or grafis, use topic "graphic_studio".
- For questions about recording media or media rekam, use topic "recording_media".
- For questions about painting studio or studio lukis, use topic "painting_studio".
- For questions about music studio or studio musik, use topic "music_studio".
- For questions about dance studio or studio tari, use topic "dance_studio".
- Specific activity words must go into rerank_keywords.
- Do not invent new metadata values.
- Output must be valid JSON only.
- Do not include markdown.
- Do not explain outside JSON.

Output schema:
{{
  "domain": "facilities_and_campus_services",
  "query_intent": "facility_info | procedure | fee_info | rule_policy | general_info",
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
"Apa saja sarana dan prasarana perpustakaan UM?"

Output:
{{
  "domain": "facilities_and_campus_services",
  "query_intent": "facility_info",
  "metadata_filters": {{
    "document_type": "facility_info",
    "academic_year": "general",
    "document_year": "",
    "topic": "library_facilities"
  }},
  "rerank_keywords": ["sarana perpustakaan", "prasarana perpustakaan", "perpustakaan"],
  "reason": "The question asks about library facilities."
}}

User question:
"Apa saja sarana umum yang tersedia di UM?"

Output:
{{
  "domain": "facilities_and_campus_services",
  "query_intent": "facility_info",
  "metadata_filters": {{
    "document_type": "facility_info",
    "academic_year": "general",
    "document_year": "",
    "topic": "general_facilities"
  }},
  "rerank_keywords": ["sarana umum", "fasilitas umum", "sarana"],
  "reason": "The question asks about general campus facilities."
}}

User question:
"Bagaimana SOP penggunaan laboratorium sosiologi?"

Output:
{{
  "domain": "facilities_and_campus_services",
  "query_intent": "procedure",
  "metadata_filters": {{
    "document_type": "facility_sop",
    "academic_year": "general",
    "document_year": "",
    "topic": "sociology_laboratory"
  }},
  "rerank_keywords": ["SOP laboratorium sosiologi", "lab sosio", "penggunaan laboratorium"],
  "reason": "The question asks about the SOP for using the sociology laboratory."
}}

User question:
"Berapa tarif penggunaan alat laboratorium sosiologi?"

Output:
{{
  "domain": "facilities_and_campus_services",
  "query_intent": "fee_info",
  "metadata_filters": {{
    "document_type": "facility_fee_info",
    "academic_year": "general",
    "document_year": "",
    "topic": "sociology_laboratory_fee"
  }},
  "rerank_keywords": ["tarif penggunaan alat", "laboratorium sosiologi", "biaya", "tarif"],
  "reason": "The question asks about usage fees for sociology laboratory equipment."
}}

User question:
"Bagaimana prosedur penggunaan studio musik?"

Output:
{{
  "domain": "facilities_and_campus_services",
  "query_intent": "procedure",
  "metadata_filters": {{
    "document_type": "facility_sop",
    "academic_year": "general",
    "document_year": "",
    "topic": "music_studio"
  }},
  "rerank_keywords": ["SOP studio musik", "prosedur studio musik", "penggunaan studio musik"],
  "reason": "The question asks about the procedure for using the music studio."
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
    Filter dibuat ringan karena collection domain ini hanya sekitar 96 chunk.

    strict=False:
    - Selalu filter domain.
    - Filter topic kalau ada.
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
        for key in ["topic"]:
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
        "sarana perpustakaan",
        "prasarana perpustakaan",
        "perpustakaan",
        "sarana umum",
        "fasilitas umum",
        "sop laboratorium",
        "sop lab",
        "laboratorium sosiologi",
        "lab sosio",
        "tarif penggunaan alat",
        "tarif lab",
        "biaya penggunaan alat",
        "studio musik",
        "studio tari",
        "studio lukis",
        "media rekam",
        "studio grafis",
        "grafis",
    ]

    procedure_phrases = [
        "prosedur",
        "layanan peminjaman",
        "peminjam",
        "mengakses website",
        "menghubungi laboran",
        "laboran memverifikasi",
        "mengisi formulir",
        "melampirkan kartu identitas",
        "melakukan pembayaran",
        "bukti transfer",
        "diberikan kwitansi",
        "mengambil peralatan",
        "mengisi buku tamu",
        "mengecek jadwal",
        "mengisi daftar pinjam",
        "konfirmasi peminjaman",
        "batas peminjaman",
    ]

    weak_intro_phrases = [
        "latar belakang",
        "ruang lingkup",
        "acuan",
        "tugas dan kewajiban",
        "surat pengajuan",
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

        topic = metadata.get("topic", "")
        document_type = metadata.get("document_type", "")

        # Bonus keyword dari LLM
        for keyword in rerank_keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower and keyword_lower in text:
                bonus += 2

        # Bonus frasa kuat jika muncul di query dan chunk
        for phrase in strong_phrases:
            if phrase in query_lower and phrase in text:
                bonus += 5

        # Bonus metadata umum
        if "perpustakaan" in query_lower or "library" in query_lower:
            if topic == "library_facilities":
                bonus += 8

        if "sarana umum" in query_lower or "fasilitas umum" in query_lower:
            if topic == "general_facilities":
                bonus += 8

        if "sosiologi" in query_lower or "sosio" in query_lower:
            if topic in {"sociology_laboratory", "sociology_laboratory_fee"}:
                bonus += 8

        if "tarif" in query_lower or "biaya" in query_lower or "harga" in query_lower:
            if document_type == "facility_fee_info":
                bonus += 8
            if topic == "sociology_laboratory_fee":
                bonus += 10

        if "sop" in query_lower or "prosedur" in query_lower or "penggunaan" in query_lower:
            if document_type == "facility_sop":
                bonus += 5

            # Ini bagian penting: naikkan chunk yang benar-benar berisi prosedur
            for phrase in procedure_phrases:
                if phrase in text:
                    bonus += 3

            # Turunkan chunk pembuka/latar belakang
            for phrase in weak_intro_phrases:
                if phrase in text:
                    penalty += 4

        if "grafis" in query_lower and topic == "graphic_studio":
            bonus += 8

        if ("media rekam" in query_lower or "rekam" in query_lower) and topic == "recording_media":
            bonus += 8

        if "lukis" in query_lower and topic == "painting_studio":
            bonus += 8

        if "musik" in query_lower and topic == "music_studio":
            bonus += 8

        if "tari" in query_lower and topic == "dance_studio":
            bonus += 8

        if "laboratorium" in query_lower or "lab" in query_lower:
            if topic in {"laboratory", "sociology_laboratory"}:
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
            "sort_key": (-final_score, distance)
        })

    return sorted(reranked, key=lambda item: item["sort_key"])