import json
import re
from app.llm import ollama_generate


VALID_DOMAINS = {
    "academic_administration",
    "thesis_final_project_and_graduation",
    "finance_tuition_and_scholarship",
    "facilities_and_campus_services"
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


def route_query_domain(user_query: str) -> dict:
    prompt = f"""
You are a domain router for a university campus RAG system.

Classify the user's question into one or more relevant domains.

Available domains:
1. academic_administration
   Questions about KRS, academic calendar, class schedule, grades, transcript,
   attendance, academic rules, study plan, academic status, and academic administration.

2. thesis_final_project_and_graduation
   Questions about thesis, final project, proposal seminar, thesis defense,
   supervisors, graduation requirements, graduation registration, and graduation ceremony.

3. finance_tuition_and_scholarship
   Questions about UKT, tuition fees, payment, invoices, scholarships,
   financial aid, installments, and student financial obligations.

4. facilities_and_campus_services
   Questions about library, laboratories, classrooms, Wi-Fi, parking,
   dormitory, health services, student ID card, and campus services.

Rules:
- The question may belong to more than one domain.
- Return only valid domain names.
- Do not explain outside JSON.
- Output must be valid JSON only.

Output schema:
{{
  "domains": ["domain_name"],
  "reason": "short reason"
}}

User question:
{user_query}

Output:
"""

    response = ollama_generate(prompt)

    try:
        result = extract_json(response)
    except json.JSONDecodeError:
        return {
            "domains": [],
            "reason": "Failed to parse router output.",
            "raw_response": response
        }

    domains = [
        domain for domain in result.get("domains", [])
        if domain in VALID_DOMAINS
    ]

    return {
        "domains": domains,
        "reason": result.get("reason", "")
    }
