from app.llm import gemini_generate

from app.answer_templates import date_schedule
from app.answer_templates import payment_schedule
from app.answer_templates import procedure
from app.answer_templates import rule_policy
from app.answer_templates import form_requirement
from app.answer_templates import scholarship_info
from app.answer_templates import registration_info
from app.answer_templates import facility_info
from app.answer_templates import fee_info
from app.answer_templates import general_info


INTENT_TEMPLATE_MAP = {
    "date_schedule": date_schedule,
    "payment_schedule": payment_schedule,
    "procedure": procedure,
    "rule_policy": rule_policy,
    "form_requirement": form_requirement,
    "scholarship_info": scholarship_info,
    "registration_info": registration_info,
    "facility_info": facility_info,
    "fee_info": fee_info,
    "general_info": general_info,
}



def choose_top_k_for_answer(query_intent: str) -> int:
    if query_intent in {"date_schedule", "payment_schedule"}:
        return 2

    if query_intent == "procedure":
        return 5

    if query_intent == "rule_policy":
        return 3

    if query_intent == "form_requirement":
        return 2

    if query_intent in {"facility_info", "fee_info", "scholarship_info", "registration_info"}:
        return 3

    return 2


def build_context_from_reranked_results(
    reranked_results,
    top_k: int = 1,
    max_chars_per_doc: int = 2000
) -> str:
    context_blocks = []

    for idx, item in enumerate(reranked_results[:top_k], start=1):
        doc = item["doc"]
        metadata = doc.metadata

        file_name = metadata.get("file_name", "unknown_file")
        page = metadata.get("page", "unknown_page")
        chunk = metadata.get("chunk_index", "unknown_chunk")
        domain = metadata.get("domain", "unknown_domain")
        document_type = metadata.get("document_type", "unknown_document_type")
        academic_year = metadata.get("academic_year", "unknown_academic_year")
        document_year = metadata.get("document_year", "unknown_document_year")
        topic = metadata.get("topic", "unknown_topic")

        content = doc.page_content[:max_chars_per_doc]

        context_block = f"""
[SOURCE {idx}]
File: {file_name}
Page: {page}
Chunk: {chunk}
Domain: {domain}
Document Type: {document_type}
Academic Year: {academic_year}
Document Year: {document_year}
Topic: {topic}

Content:
{content}
"""

        context_blocks.append(context_block)

    return "\n\n".join(context_blocks)


def compress_numbers(values):
    numbers = []

    for value in values:
        try:
            numbers.append(int(value))
        except Exception:
            pass

    numbers = sorted(set(numbers))

    if not numbers:
        return ", ".join(str(value) for value in values)

    ranges = []
    start = numbers[0]
    prev = numbers[0]

    for num in numbers[1:]:
        if num == prev + 1:
            prev = num
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = num
            prev = num

    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")

    return ", ".join(ranges)


def build_source_line(reranked_results, top_k: int) -> str:
    used = reranked_results[:top_k]

    sources = {}

    for item in used:
        doc = item["doc"]
        metadata = doc.metadata

        file_name = metadata.get("file_name", "unknown_file")
        page = metadata.get("page", "unknown_page")
        chunk = metadata.get("chunk_index", "unknown_chunk")

        if file_name not in sources:
            sources[file_name] = {
                "pages": [],
                "chunks": []
            }

        sources[file_name]["pages"].append(page)
        sources[file_name]["chunks"].append(chunk)

    source_parts = []

    for file_name, data in sources.items():
        pages = compress_numbers(data["pages"])
        chunks = compress_numbers(data["chunks"])

        source_parts.append(
            f"{file_name}, halaman {pages}, chunk {chunks}"
        )

    return "Sumber: " + "; ".join(source_parts) + "."


def remove_source_lines(answer: str) -> str:
    lines = answer.strip().splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("sumber:"):
            continue

        if stripped.lower().startswith("source:"):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def generate_answer_from_context(
    user_query: str,
    reranked_results,
    query_intent: str = "general_info"
) -> str:
    if not reranked_results:
        return "Informasi yang relevan tidak ditemukan dalam dokumen yang tersedia."

    top_k = choose_top_k_for_answer(query_intent)

    context = build_context_from_reranked_results(
        reranked_results=reranked_results,
        top_k=top_k,
        max_chars_per_doc=2000
    )

    template_module = INTENT_TEMPLATE_MAP.get(query_intent, general_info)

    prompt = template_module.build_prompt(
        user_query=user_query,
        context=context
    )

    answer = gemini_generate(prompt)
    answer = remove_source_lines(answer)

    source_line = build_source_line(
        reranked_results=reranked_results,
        top_k=top_k
    )

    if not answer:
        answer = "Informasi tersebut tidak ditemukan secara jelas dalam dokumen yang tersedia."

    return f"{answer}\n\n{source_line}"
