import os
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional

import chromadb
from fastapi import FastAPI, HTTPException, Request, Depends, Response, Cookie
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from upstash_redis.asyncio import Redis
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

from openai import OpenAI, RateLimitError
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# ─── SETTINGS ────────────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    UPSTASH_REDIS_REST_URL: str
    UPSTASH_REDIS_REST_TOKEN: str
    CHROMA_API_KEY: str
    CHROMA_TENANT: str
    CHROMA_DATABASE: str
    COLLECTION_NAME: str = "legal_docs"
    TOP_K: int = 7
    SESSION_TIMEOUT_MIN: int = 30
    RATE_LIMIT: str = "60/minute"
    RELEVANCE_THRESHOLD: float = 0.75
    RECENCY_KEYWORDS: list = [
        "2024", "2025", "latest", "current", "recent", "new law",
        "updated", "amendment", "now", "today", "changed"
    ]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# ─── LOGGING ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("legal-bot")

# ─── GLOBALS ─────────────────────────────────────────────────────────────────────
redis: Redis = None

# ─── GREETING DETECTION ──────────────────────────────────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "howya", "hiya", "sup", "what's up", "greetings", "yo", "morning", "evening"
}

GREETING_RESPONSES = [
    "Hello! 👋 I'm your Irish Legal Assistant. I can help you understand Irish law, your rights, "
    "legislation, legal processes, and more. What legal question can I help you with today?",
    "Hi there! 👋 Welcome to the Irish Legal AI Assistant. I'm here to help you navigate Irish law "
    "and legal matters. What would you like to know?",
    "Hey! 👋 I'm here to help with any Irish legal questions you have — from employment law to "
    "tenancy rights, criminal law, family law, and beyond. What's on your mind?"
]

NON_LEGAL_PATTERNS = [
    "weather", "sport", "football", "recipe", "cook", "movie", "music",
    "news", "joke", "story", "game", "song", "celebrity", "politics",
    "cryptocurrency", "bitcoin", "stock", "fashion", "travel"
]

def classify_query(query: str) -> str:
    """Returns: 'greeting' | 'non_legal' | 'legal'"""
    q = query.lower().strip()

    if q in GREETINGS or any(q.startswith(g) for g in GREETINGS):
        return "greeting"

    if len(q.split()) <= 2 and "?" not in q and not any(
        kw in q for kw in ["law", "right", "legal", "act", "court", "fine", "penalty", "gdpr", "rent", "tax"]
    ):
        return "greeting"

    if any(pattern in q for pattern in NON_LEGAL_PATTERNS):
        return "non_legal"

    return "legal"

def needs_recency_check(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in settings.RECENCY_KEYWORDS)

# ─── LIFESPAN ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis
    redis = Redis(
        url=settings.UPSTASH_REDIS_REST_URL,
        token=settings.UPSTASH_REDIS_REST_TOKEN
    )
    logger.info("Upstash Redis connection established")
    yield
    await redis.close()
    logger.info("Upstash Redis connection closed")

# ─── APP ─────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Irish Legal AI Bot",
    description="RAG-driven Irish legal assistant",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# ─── OPENAI CLIENT ───────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ─── MODERATION ──────────────────────────────────────────────────────────────────
async def moderate_content(text: str) -> bool:
    try:
        resp = await asyncio.to_thread(
            openai_client.moderations.create, input=text
        )
        return not resp.results[0].flagged
    except RateLimitError:
        logger.warning("Moderation rate limited — allowing through (fail-open)")
        return True  # fail-open: assume safe, let query proceed
    except Exception as e:
        logger.error(f"Moderation error: {e}")
        return True  # fail-open for other transient errors too

# ─── SESSION MANAGEMENT ──────────────────────────────────────────────────────────
class SessionData(BaseModel):
    session_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    history: list

async def get_session(
    session_id: str = Cookie(default=None),
    response: Response = None
) -> SessionData:
    if session_id:
        try:
            raw = await redis.get(session_id)
            if raw:
                data = SessionData.model_validate_json(raw)
                if datetime.utcnow() <= data.expires_at:
                    data.last_activity = datetime.utcnow()
                    remaining = int((data.expires_at - datetime.utcnow()).total_seconds())
                    await redis.setex(session_id, remaining, data.model_dump_json())
                    return data
                else:
                    await redis.delete(session_id)
        except Exception as e:
            logger.error(f"Session fetch error: {e}")

    new_id = str(uuid.uuid4())
    now = datetime.utcnow()
    data = SessionData(
        session_id=new_id,
        created_at=now,
        expires_at=now + timedelta(minutes=settings.SESSION_TIMEOUT_MIN),
        last_activity=now,
        history=[]
    )
    await redis.setex(new_id, settings.SESSION_TIMEOUT_MIN * 60, data.model_dump_json())

    if response:
        response.set_cookie(
            key="session_id",
            value=new_id,
            httponly=True,
            secure=True,
            samesite="None",
            path="/"
        )
    return data

async def save_session(session: SessionData):
    remaining = (session.expires_at - datetime.utcnow()).total_seconds()
    if remaining > 0:
        await redis.setex(session.session_id, int(remaining), session.model_dump_json())

# ─── VECTOR STORE ────────────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.OPENAI_API_KEY
)

cloud_client = chromadb.CloudClient(
    tenant=settings.CHROMA_TENANT,
    database=settings.CHROMA_DATABASE,
    api_key=settings.CHROMA_API_KEY
)

vectordb = Chroma(
    client=cloud_client,
    collection_name=settings.COLLECTION_NAME,
    embedding_function=embeddings
)

class RetrievalResult(BaseModel):
    context: str
    sources: list[str]
    confidence: str
    top_score: float
    has_dated_content: bool
    doc_years: list[str]

def extract_years_from_text(text: str) -> list[str]:
    import re
    return re.findall(r"\b(19[89]\d|20[012]\d)\b", text)

def retrieve_context(query: str) -> RetrievalResult:
    try:
        docs = vectordb.similarity_search_with_score(query, k=settings.TOP_K)
        if not docs:
            return RetrievalResult(
                context="No relevant legal context found.",
                sources=[],
                confidence="none",
                top_score=0.0,
                has_dated_content=False,
                doc_years=[]
            )

        docs.sort(key=lambda x: x[1])
        top_score = docs[0][1]
        all_years = []
        snippets = []
        sources = []

        for i, (doc, score) in enumerate(docs):
            years = extract_years_from_text(doc.page_content)
            all_years.extend(years)
            relevance_label = (
                "High" if score < 0.5 else
                "Medium" if score < 0.75 else
                "Low"
            )
            snippets.append(
                f"[Source {i+1} | Relevance: {relevance_label} | Score: {score:.3f}]\n"
                f"{doc.page_content.strip()}"
            )
            sources.append(doc.metadata.get("source", f"Source {i+1}"))

        if top_score < 0.4:
            confidence = "high"
        elif top_score < 0.65:
            confidence = "medium"
        elif top_score < settings.RELEVANCE_THRESHOLD:
            confidence = "low"
        else:
            confidence = "none"

        unique_years = sorted(set(all_years), reverse=True)

        return RetrievalResult(
            context="\n\n".join(snippets),
            sources=sources,
            confidence=confidence,
            top_score=top_score,
            has_dated_content=bool(unique_years),
            doc_years=unique_years
        )

    except Exception as e:
        logger.error(f"Vector retrieval error: {e}")
        return RetrievalResult(
            context="Context retrieval failed.",
            sources=[],
            confidence="none",
            top_score=1.0,
            has_dated_content=False,
            doc_years=[]
        )

# ─── PROMPTS (IMPROVED – NATURAL & CONVERSATIONAL) ─────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question", "history", "recency_note"],
    template=(
        "You are an experienced Irish legal expert assistant.\n\n"
        "Use the retrieved legal context below as evidence, but do not simply repeat it.\n"
        "Your goal is to answer naturally, conversationally, and helpfully.\n\n"
        "{recency_note}"
        "RETRIEVED LEGAL CONTEXT (use for facts, citations, and support):\n{context}\n\n"
        "CONVERSATION HISTORY:\n{history}\n\n"
        "USER QUESTION:\n{question}\n\n"
        "Write a direct, clear answer. If the user asks for a specific aspect (penalties, exceptions, steps), address it directly.\n"
        "Cite relevant Acts or case law when it adds value. Keep the tone professional yet approachable.\n"
        "Do not use rigid sections like 'DIRECT ANSWER' or 'LEGAL BASIS'. Instead, write flowing prose.\n"
        "If the question is simple, a short answer is fine. If complex, break it into short paragraphs or bullet points.\n"
        "End with a practical next step or a follow‑up question when appropriate.\n"
    )
)

FALLBACK_PROMPT = PromptTemplate(
    input_variables=["question", "history", "partial_context"],
    template=(
        "You are a knowledgeable Irish legal expert assistant.\n\n"
        "The user's question could not be fully answered from your legal database.\n"
        "Use your expert knowledge of Irish law (as of 2024–2025) to provide the best possible answer.\n\n"
        "PARTIAL CONTEXT FROM DATABASE (may be incomplete):\n{partial_context}\n\n"
        "CONVERSATION HISTORY:\n{history}\n\n"
        "USER QUESTION:\n{question}\n\n"
        "Answer clearly and conversationally. Cite specific Irish legislation or EU regulations when known.\n"
        "If the information is uncertain, state that and suggest where the user can verify (e.g., citizensinformation.ie).\n"
        "Do not use forced section headers. Write natural, helpful paragraphs.\n"
    )
)

RECENCY_SUPPLEMENT_NOTE = (
    "⚠️ The user is asking about recent law (2024/2025 or 'latest'). "
    "If the retrieved context contains both old and new provisions, give priority to the most recent. "
    "If only older information is available, clearly note that and advise checking official sources.\n\n"
)

# ─── LLM & LCEL CHAINS ───────────────────────────────────────────────────────────
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=settings.OPENAI_API_KEY,
    model="gpt-4-turbo",
    max_tokens=1800
)

_parser = StrOutputParser()

rag_chain      = RAG_PROMPT      | llm | _parser
fallback_chain = FALLBACK_PROMPT | llm | _parser

# ─── INTELLIGENT ANSWER ENGINE ───────────────────────────────────────────────────
async def generate_answer(
    query: str,
    history_text: str,
    retrieval: RetrievalResult,
    recency_needed: bool
) -> tuple[str, str]:
    """Returns (answer, mode_used)  —  mode: 'rag' | 'fallback' | 'hybrid'"""
    recency_note = RECENCY_SUPPLEMENT_NOTE if recency_needed else ""

    if retrieval.confidence in ("high", "medium"):
        logger.info(f"Mode: RAG | Confidence: {retrieval.confidence} | Score: {retrieval.top_score:.3f}")
        answer = await asyncio.to_thread(
            rag_chain.invoke,
            {
                "context": retrieval.context,
                "question": query,
                "history": history_text,
                "recency_note": recency_note,
            }
        )
        return answer, "rag"

    elif retrieval.confidence == "low":
        logger.info(f"Mode: HYBRID | Confidence: {retrieval.confidence} | Score: {retrieval.top_score:.3f}")
        answer = await asyncio.to_thread(
            fallback_chain.invoke,
            {
                "question": query,
                "history": history_text,
                "partial_context": retrieval.context,
            }
        )
        return answer, "hybrid"

    else:
        logger.info(f"Mode: FALLBACK | Confidence: {retrieval.confidence} | Score: {retrieval.top_score:.3f}")
        answer = await asyncio.to_thread(
            fallback_chain.invoke,
            {
                "question": query,
                "history": history_text,
                "partial_context": "No relevant context found in the legal database.",
            }
        )
        return answer, "fallback"

# ─── PYDANTIC MODELS ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str]
    mode: str
    confidence: str

class SessionStatusResponse(BaseModel):
    status: str
    ttl: int
    session_id: Optional[str]
    created_at: Optional[datetime]
    expires_at: Optional[datetime]
    last_activity: Optional[datetime]
    history_count: Optional[int]

class SessionHistoryResponse(BaseModel):
    history: list
    session_id: str

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    redis: str
    vectordb: str

# ─── ROUTES ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    redis_status = "ok"
    try:
        await redis.ping()
    except Exception:
        redis_status = "error"

    vectordb_status = "ok"
    try:
        await asyncio.to_thread(vectordb.similarity_search, "test", 1)
    except Exception:
        vectordb_status = "error"

    return HealthResponse(
        status="ok" if redis_status == "ok" and vectordb_status == "ok" else "degraded",
        timestamp=datetime.utcnow(),
        redis=redis_status,
        vectordb=vectordb_status
    )

@app.post("/query", response_model=QueryResponse)
@limiter.limit(settings.RATE_LIMIT)
async def handle_query(
    request: Request,
    req: QueryRequest,
    session: SessionData = Depends(get_session),
    response: Response = None
):
    import random

    query = req.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > 2000:
        raise HTTPException(status_code=400, detail="Query too long. Max 2000 characters.")

    if not await moderate_content(query):
        raise HTTPException(status_code=400, detail="Content policy violation.")

    query_type = classify_query(query)

    if query_type == "greeting":
        return QueryResponse(
            answer=random.choice(GREETING_RESPONSES),
            session_id=session.session_id,
            sources=[],
            mode="greeting",
            confidence="n/a"
        )

    if query_type == "non_legal":
        return QueryResponse(
            answer="I'm specifically designed to assist with Irish legal matters only. 🏛️\n\n",
            session_id=session.session_id,
            sources=[],
            mode="deflect",
            confidence="n/a"
        )

    recency_needed = needs_recency_check(query)
    retrieval = await asyncio.to_thread(retrieve_context, query)

    history_text = "No prior conversation."
    if session.history:
        recent = session.history[-3:]
        history_text = "\n".join(
            [f"User: {h['q']}\nAssistant: {h['a']}" for h in recent]
        )

    try:
        answer, mode_used = await generate_answer(
            query=query,
            history_text=history_text,
            retrieval=retrieval,
            recency_needed=recency_needed
        )
    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate a response.")

    if not await moderate_content(answer):
        answer = "I'm unable to provide a response to that query due to content policy restrictions."

    session.history.append({
        "q": query,
        "a": answer,
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode_used
    })
    if len(session.history) > 5:
        session.history.pop(0)

    await save_session(session)

    logger.info(
        f"Query handled | session={session.session_id} | "
        f"mode={mode_used} | confidence={retrieval.confidence} | "
        f"sources={len(retrieval.sources)} | recency={recency_needed}"
    )

    return QueryResponse(
        answer=answer,
        session_id=session.session_id,
        sources=retrieval.sources,
        mode=mode_used,
        confidence=retrieval.confidence
    )

@app.get("/session/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str = Cookie(default=None)):
    if not session_id:
        return SessionStatusResponse(
            status="new", ttl=-2, session_id=None,
            created_at=None, expires_at=None,
            last_activity=None, history_count=None
        )
    try:
        raw = await redis.get(session_id)
    except Exception as e:
        logger.error(f"Redis error on status check: {e}")
        raise HTTPException(status_code=500, detail="Session store unavailable.")

    if not raw:
        return SessionStatusResponse(
            status="expired", ttl=-2, session_id=session_id,
            created_at=None, expires_at=None,
            last_activity=None, history_count=None
        )

    data = SessionData.model_validate_json(raw)
    now = datetime.utcnow()

    if now > data.expires_at:
        return SessionStatusResponse(
            status="expired", ttl=-2, session_id=session_id,
            created_at=data.created_at, expires_at=data.expires_at,
            last_activity=data.last_activity, history_count=len(data.history)
        )

    return SessionStatusResponse(
        status="active",
        ttl=int((data.expires_at - now).total_seconds()),
        session_id=session_id,
        created_at=data.created_at,
        expires_at=data.expires_at,
        last_activity=data.last_activity,
        history_count=len(data.history)
    )

@app.get("/session/history", response_model=SessionHistoryResponse)
async def get_session_history(session: SessionData = Depends(get_session)):
    return SessionHistoryResponse(
        history=session.history,
        session_id=session.session_id
    )

@app.delete("/session/clear")
async def clear_session(session: SessionData = Depends(get_session)):
    try:
        await redis.delete(session.session_id)
        logger.info(f"Session cleared: {session.session_id}")
        return JSONResponse({"message": "Session cleared successfully."})
    except Exception as e:
        logger.error(f"Session clear error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session.")

# ─── EXCEPTION HANDLERS ──────────────────────────────────────────────────────────
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down."}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."}
    )

# ─── LAUNCH ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=4, log_level="info")