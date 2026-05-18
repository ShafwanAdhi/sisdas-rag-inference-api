from typing import Any, Dict, Generator, List, Optional
import json
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.router import route_query_domain
from app.retrieval import get_vectorstore, retrieve
from app.answer_generator import choose_top_k_for_answer, generate_answer_from_context

from app.domains import academic_administration
from app.domains import thesis_final_project_and_graduation
from app.domains import finance_tuition_and_scholarship
from app.domains import facilities_and_campus_services

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Campus Assistant API",
    description="API RAG dengan response biasa dan response streaming status proses.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOMAIN_MODULES = {
    "academic_administration": academic_administration,
    "finance_tuition_and_scholarship": finance_tuition_and_scholarship,
    "thesis_final_project_and_graduation": thesis_final_project_and_graduation,
    "facilities_and_campus_services": facilities_and_campus_services,
}

COLLECTION_BY_DOMAIN = {
    "academic_administration": "academic_administration",
    "thesis_final_project_and_graduation": "thesis_final_project_and_graduation",
    "finance_tuition_and_scholarship": "finance_tuition_and_scholarship",
    "facilities_and_campus_services": "facilities_and_campus_services",
}


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Pertanyaan dari user")


class ContextResponse(BaseModel):
    id: Optional[str] = None
    page_content: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None
    keyword_bonus: Optional[float] = None
    penalty: Optional[float] = None
    final_score: Optional[float] = None
    retrieval_rank: Optional[int] = None
    rerank_rank: Optional[int] = None
    rank_delta: Optional[int] = None
    matched_rerank_keywords: List[str] = Field(default_factory=list)
    contains_rerank_keyword: bool = False


class ChatResponse(BaseModel):
    user_query: str
    answer: str
    route: Dict[str, Any]
    analyses: Dict[str, Any] = Field(default_factory=dict)
    contexts: List[ContextResponse] = Field(default_factory=list)
    reranking_keyword: Dict[str, Any] = Field(default_factory=dict)
    query_intent_adaptive_top_k: Dict[str, Any] = Field(default_factory=dict)


def serialize_context(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mengubah hasil retrieval/rerank yang berisi object Document
    menjadi dict biasa agar aman dikirim sebagai JSON.
    """
    doc = item.get("doc")
    return {
        "id": getattr(doc, "id", None),
        "page_content": getattr(doc, "page_content", ""),
        "metadata": getattr(doc, "metadata", {}) or {},
        "distance": item.get("distance"),
        "keyword_bonus": item.get("keyword_bonus"),
        "penalty": item.get("penalty"),
        "final_score": item.get("final_score"),
        "retrieval_rank": item.get("retrieval_rank"),
        "rerank_rank": item.get("rerank_rank"),
        "rank_delta": item.get("rank_delta"),
        "matched_rerank_keywords": item.get("matched_rerank_keywords", []),
        "contains_rerank_keyword": item.get("contains_rerank_keyword", False),
    }


def make_sse_event(event: str, data: Dict[str, Any]) -> str:
    """
    Format event untuk Server-Sent Events.
    Frontend bisa membaca event: status, done, atau error.
    """
    json_data = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {json_data}\n\n"


def get_matched_keywords(text: str, rerank_keywords: List[str]) -> List[str]:
    """
    Menandai keyword hasil analyzer yang benar-benar muncul pada chunk.
    Ini dipakai untuk kebutuhan whitebox reranking keyword.
    """
    text_lower = text.lower()
    matched_keywords = []

    for keyword in rerank_keywords:
        keyword_text = str(keyword).strip()
        if keyword_text and keyword_text.lower() in text_lower:
            matched_keywords.append(keyword_text)

    return matched_keywords


def summarize_context_for_whitebox(item: Dict[str, Any]) -> Dict[str, Any]:
    doc = item.get("doc")
    metadata = getattr(doc, "metadata", {}) or {}

    return {
        "file_name": metadata.get("file_name"),
        "page": metadata.get("page"),
        "chunk_index": metadata.get("chunk_index"),
        "domain": metadata.get("domain"),
        "topic": metadata.get("topic"),
        "retrieval_rank": item.get("retrieval_rank"),
        "rerank_rank": item.get("rerank_rank"),
        "rank_delta": item.get("rank_delta"),
        "keyword_bonus": item.get("keyword_bonus"),
        "penalty": item.get("penalty"),
        "final_score": item.get("final_score"),
        "matched_rerank_keywords": item.get("matched_rerank_keywords", []),
    }


def annotate_rerank_results(
    *,
    retrieved_results: List[Any],
    reranked_results: List[Dict[str, Any]],
    rerank_keywords: List[str],
) -> List[Dict[str, Any]]:
    """
    Menambahkan rank sebelum dan sesudah reranking supaya frontend bisa
    mengecek apakah chunk yang memuat keyword penting naik peringkat.
    """
    retrieval_rank_by_doc = {
        id(doc): rank
        for rank, (doc, _distance) in enumerate(retrieved_results, start=1)
    }

    for rerank_rank, item in enumerate(reranked_results, start=1):
        doc = item.get("doc")
        retrieval_rank = retrieval_rank_by_doc.get(id(doc))
        matched_keywords = get_matched_keywords(
            getattr(doc, "page_content", ""),
            rerank_keywords,
        )

        item["retrieval_rank"] = retrieval_rank
        item["rerank_rank"] = rerank_rank
        item["rank_delta"] = (
            retrieval_rank - rerank_rank
            if retrieval_rank is not None
            else None
        )
        item["matched_rerank_keywords"] = matched_keywords
        item["contains_rerank_keyword"] = bool(matched_keywords)

    return reranked_results


def build_keyword_reranking_report(
    *,
    domain: str,
    analysis: Dict[str, Any],
    retrieved_count: int,
    reranked_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    keyword_contexts = [
        item
        for item in reranked_results
        if item.get("contains_rerank_keyword")
    ]
    promoted_contexts = [
        item
        for item in keyword_contexts
        if (item.get("rank_delta") or 0) > 0
    ]

    return {
        "domain": domain,
        "rerank_keywords": analysis.get("rerank_keywords", []),
        "retrieved_count": retrieved_count,
        "reranked_count": len(reranked_results),
        "chunks_with_keyword_count": len(keyword_contexts),
        "promoted_keyword_chunk_count": len(promoted_contexts),
        "chunks_with_keyword_promoted": bool(promoted_contexts),
        "top_keyword_chunks": [
            summarize_context_for_whitebox(item)
            for item in keyword_contexts[:5]
        ],
    }


def build_query_intent_top_k_report(
    *,
    best_domain: Optional[str],
    analyses: Dict[str, Any],
    available_context_count: int,
) -> Dict[str, Any]:
    query_intent = analyses.get(best_domain or "", {}).get("analysis", {}).get(
        "query_intent",
        "general_info",
    )
    adaptive_top_k = choose_top_k_for_answer(query_intent)

    return {
        "best_domain": best_domain,
        "query_intent": query_intent,
        "adaptive_top_k": adaptive_top_k,
        "context_count_used": min(adaptive_top_k, available_context_count),
        "available_context_count": available_context_count,
        "selection_rule": "choose_top_k_for_answer(query_intent)",
    }


def rag_answer(user_query: str) -> Dict[str, Any]:
    """
    Versi response biasa: cocok untuk POST /chat.
    Tidak mengirim status bertahap ke frontend.
    """
    logger.info("Routing query...")
    route = route_query_domain(user_query)
    domains = route.get("domains", [])

    if not domains:
        return {
            "answer": "Domain pertanyaan tidak dapat ditentukan.",
            "route": route,
            "analyses": {},
            "contexts": [],
            "reranking_keyword": {},
            "query_intent_adaptive_top_k": {},
        }

    all_contexts = []
    analyses = {}
    reranking_keyword = {"domains": {}}

    for domain in domains:
        domain_module = DOMAIN_MODULES.get(domain)
        if domain_module is None:
            continue

        analysis = domain_module.analyze_query(user_query)
        metadata_filter = domain_module.build_filter(analysis)
        collection_name = COLLECTION_BY_DOMAIN[domain]
        vectorstore = get_vectorstore(collection_name)

        results = retrieve(
            vectorstore=vectorstore,
            query=user_query,
            metadata_filter=metadata_filter,
            k=10,
        )

        reranked = domain_module.rerank(
            results=results,
            rerank_keywords=analysis.get("rerank_keywords", []),
            query=user_query,
        )
        reranked = annotate_rerank_results(
            retrieved_results=results,
            reranked_results=reranked,
            rerank_keywords=analysis.get("rerank_keywords", []),
        )

        analyses[domain] = {
            "analysis": analysis,
            "metadata_filter": metadata_filter,
        }
        reranking_keyword["domains"][domain] = build_keyword_reranking_report(
            domain=domain,
            analysis=analysis,
            retrieved_count=len(results),
            reranked_results=reranked,
        )
        all_contexts.extend(reranked[:5])

    if not all_contexts:
        return {
            "answer": "Informasi yang relevan tidak ditemukan dalam dokumen yang tersedia.",
            "route": route,
            "analyses": analyses,
            "contexts": [],
            "reranking_keyword": reranking_keyword,
            "query_intent_adaptive_top_k": {},
        }

    all_contexts = sorted(
        all_contexts,
        key=lambda item: (-(item.get("final_score", 0)), item.get("distance", 999)),
    )

    best_domain = all_contexts[0]["doc"].metadata.get("domain")
    query_intent = analyses.get(best_domain, {}).get("analysis", {}).get(
        "query_intent",
        "general_info",
    )
    query_intent_adaptive_top_k = build_query_intent_top_k_report(
        best_domain=best_domain,
        analyses=analyses,
        available_context_count=len(all_contexts),
    )
    reranking_keyword["summary"] = {
        "chunks_with_keyword_promoted": any(
            report.get("chunks_with_keyword_promoted")
            for report in reranking_keyword["domains"].values()
        ),
        "promoted_keyword_chunk_count": sum(
            report.get("promoted_keyword_chunk_count", 0)
            for report in reranking_keyword["domains"].values()
        ),
        "chunks_with_keyword_count": sum(
            report.get("chunks_with_keyword_count", 0)
            for report in reranking_keyword["domains"].values()
        ),
    }

    answer = generate_answer_from_context(
        user_query=user_query,
        reranked_results=all_contexts,
        query_intent=query_intent,
    )

    return {
        "answer": answer,
        "route": route,
        "analyses": analyses,
        "contexts": [serialize_context(item) for item in all_contexts[:3]],
        "reranking_keyword": reranking_keyword,
        "query_intent_adaptive_top_k": query_intent_adaptive_top_k,
    }


def rag_answer_stream(user_query: str) -> Generator[str, None, None]:
    """
    Versi streaming: cocok untuk GET /chat/stream.
    Fungsi ini mengirim status proses ke frontend secara bertahap.
    """
    try:
        yield make_sse_event(
            "status",
            {
                "step": "routing",
                "message": "Menganalisis domain pertanyaan user...",
                "progress": 5,
            },
        )

        route = route_query_domain(user_query)
        domains = route.get("domains", [])

        yield make_sse_event(
            "status",
            {
                "step": "routing_done",
                "message": f"Query diarahkan ke domain: {', '.join(domains) if domains else 'tidak ditemukan'}",
                "progress": 15,
                "route": route,
                "domains": domains,
            },
        )

        if not domains:
            yield make_sse_event(
                "done",
                {
                    "user_query": user_query,
                    "answer": "Domain pertanyaan tidak dapat ditentukan.",
                    "route": route,
                    "analyses": {},
                    "contexts": [],
                    "reranking_keyword": {},
                    "query_intent_adaptive_top_k": {},
                },
            )
            return

        all_contexts = []
        analyses = {}
        reranking_keyword = {"domains": {}}
        total_domains = len(domains)

        for index, domain in enumerate(domains, start=1):
            domain_module = DOMAIN_MODULES.get(domain)
            if domain_module is None:
                yield make_sse_event(
                    "status",
                    {
                        "step": "skip_domain",
                        "message": f"Domain {domain} tidak dikenali, dilewati.",
                        "progress": 20,
                        "domain": domain,
                    },
                )
                continue

            base_progress = 15 + int((index - 1) / max(total_domains, 1) * 55)

            yield make_sse_event(
                "status",
                {
                    "step": "analyzing_domain",
                    "message": f"Menganalisis query untuk domain {domain}...",
                    "progress": base_progress + 5,
                    "domain": domain,
                },
            )

            analysis = domain_module.analyze_query(user_query)
            metadata_filter = domain_module.build_filter(analysis)

            analyses[domain] = {
                "analysis": analysis,
                "metadata_filter": metadata_filter,
            }

            yield make_sse_event(
                "status",
                {
                    "step": "retrieving",
                    "message": f"Mengambil dokumen relevan dari ChromaDB untuk domain {domain}...",
                    "progress": base_progress + 15,
                    "domain": domain,
                    "metadata_filter": metadata_filter,
                },
            )

            collection_name = COLLECTION_BY_DOMAIN[domain]
            vectorstore = get_vectorstore(collection_name)
            results = retrieve(
                vectorstore=vectorstore,
                query=user_query,
                metadata_filter=metadata_filter,
                k=10,
            )

            yield make_sse_event(
                "status",
                {
                    "step": "reranking",
                    "message": f"Melakukan reranking dokumen untuk domain {domain}...",
                    "progress": base_progress + 25,
                    "domain": domain,
                    "retrieved_count": len(results),
                },
            )

            reranked = domain_module.rerank(
                results=results,
                rerank_keywords=analysis.get("rerank_keywords", []),
                query=user_query,
            )
            reranked = annotate_rerank_results(
                retrieved_results=results,
                reranked_results=reranked,
                rerank_keywords=analysis.get("rerank_keywords", []),
            )
            reranking_keyword["domains"][domain] = build_keyword_reranking_report(
                domain=domain,
                analysis=analysis,
                retrieved_count=len(results),
                reranked_results=reranked,
            )

            all_contexts.extend(reranked[:5])

            yield make_sse_event(
                "status",
                {
                    "step": "domain_done",
                    "message": f"Domain {domain} selesai diproses.",
                    "progress": base_progress + 35,
                    "domain": domain,
                    "selected_context_count": len(reranked[:5]),
                    "reranking_keyword": reranking_keyword["domains"][domain],
                },
            )

        if not all_contexts:
            yield make_sse_event(
                "done",
                {
                    "user_query": user_query,
                    "answer": "Informasi yang relevan tidak ditemukan dalam dokumen yang tersedia.",
                    "route": route,
                    "analyses": analyses,
                    "contexts": [],
                    "reranking_keyword": reranking_keyword,
                    "query_intent_adaptive_top_k": {},
                },
            )
            return

        yield make_sse_event(
            "status",
            {
                "step": "global_reranking",
                "message": "Mengurutkan ulang dokumen terbaik dari seluruh domain...",
                "progress": 75,
                "total_context_count": len(all_contexts),
            },
        )

        all_contexts = sorted(
            all_contexts,
            key=lambda item: (-(item.get("final_score", 0)), item.get("distance", 999)),
        )

        best_domain = all_contexts[0]["doc"].metadata.get("domain")
        query_intent = analyses.get(best_domain, {}).get("analysis", {}).get(
            "query_intent",
            "general_info",
        )
        query_intent_adaptive_top_k = build_query_intent_top_k_report(
            best_domain=best_domain,
            analyses=analyses,
            available_context_count=len(all_contexts),
        )
        reranking_keyword["summary"] = {
            "chunks_with_keyword_promoted": any(
                report.get("chunks_with_keyword_promoted")
                for report in reranking_keyword["domains"].values()
            ),
            "promoted_keyword_chunk_count": sum(
                report.get("promoted_keyword_chunk_count", 0)
                for report in reranking_keyword["domains"].values()
            ),
            "chunks_with_keyword_count": sum(
                report.get("chunks_with_keyword_count", 0)
                for report in reranking_keyword["domains"].values()
            ),
        }

        yield make_sse_event(
            "status",
            {
                "step": "generating",
                "message": "Menyusun jawaban akhir menggunakan konteks terbaik...",
                "progress": 90,
                "best_domain": best_domain,
                "query_intent": query_intent,
                "adaptive_top_k": query_intent_adaptive_top_k["adaptive_top_k"],
            },
        )

        answer = generate_answer_from_context(
            user_query=user_query,
            reranked_results=all_contexts,
            query_intent=query_intent,
        )

        yield make_sse_event(
            "done",
            {
                "user_query": user_query,
                "answer": answer,
                "route": route,
                "analyses": analyses,
                "contexts": [serialize_context(item) for item in all_contexts[:3]],
                "reranking_keyword": reranking_keyword,
                "query_intent_adaptive_top_k": query_intent_adaptive_top_k,
            },
        )

    except Exception as exc:
        logger.exception("Failed to stream chat request")
        yield make_sse_event(
            "error",
            {
                "step": "error",
                "message": str(exc),
            },
        )


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "RAG FastAPI server is running",
        "docs": "/docs",
        "chat_endpoint": "POST /chat",
        "stream_endpoint": "GET /chat/stream?query=...",
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    Endpoint biasa:
    frontend menunggu sampai semua proses selesai, lalu menerima satu JSON final.
    """
    try:
        result = await run_in_threadpool(rag_answer, request.query)
        return {
            "user_query": request.query,
            "answer": result["answer"],
            "route": result.get("route", {}),
            "analyses": result.get("analyses", {}),
            "contexts": result.get("contexts", []),
            "reranking_keyword": result.get("reranking_keyword", {}),
            "query_intent_adaptive_top_k": result.get(
                "query_intent_adaptive_top_k",
                {},
            ),
        }
    except Exception as exc:
        logger.exception("Failed to process chat request")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/chat/stream")
def chat_stream(query: str = Query(..., min_length=1)) -> StreamingResponse:
    """
    Endpoint streaming status:
    frontend akan menerima event bertahap lewat Server-Sent Events.
    """
    return StreamingResponse(
        rag_answer_stream(query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Jalankan dari root project:
# uvicorn main_streaming:app --reload
