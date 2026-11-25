"""Run the MedVQA-Evidence FastAPI and LangGraph application."""

import argparse
from contextlib import asynccontextmanager, closing
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import re
from secrets import token_urlsafe
import sqlite3
from threading import Lock
from time import perf_counter
from typing import Literal, TypedDict
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from PIL import Image
from pydantic import BaseModel, Field

from evidence.perturbation import BiomedAlignment

from .audit import estimate_risk, probe
from .maira import CARDIOMEGALY, CXR, ChestRegions
from .dataset import SPLITS, Slake
from .paths import ARTIFACTS, DATA, default_model
from evidence.rag.retrieval import CORPUS, DEFAULT_RAG_DB, Retriever
from evidence.vqa.model import MODEL_ID, VQAPredictor


Intent = Literal["auto", "visual", "knowledge", "mixed"]
GRAPH_STATE = ARTIFACTS / "data" / "chat.sqlite3"
LOGGER = logging.getLogger(__name__)


class State(TypedDict, total=False):
    question: str
    requested_intent: Intent
    image_id: str | None
    modality: str | None
    organ: str | None
    intent: str
    preference: str
    last_visual_answer: str
    image_answer: str | None
    audit: dict | None
    passages: list[dict]
    knowledge_answer: str | None
    status: list[str]


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1500)
    intent: Intent = "auto"
    image_id: str | None = Field(default=None, max_length=80)
    modality: str | None = Field(default=None, max_length=30)
    organ: str | None = Field(default=None, max_length=50)
    thread_id: UUID | None = None


class Preference(BaseModel):
    style: Literal["concise", "detailed"] = "concise"


def route(state):
    intent = state["requested_intent"]
    if intent == "auto":
        if not state.get("image_id"):
            intent = "knowledge"
        elif re.search(r"why|how does|compare|different|source|principle|解释|出处|区别|原理", state["question"], re.I):
            intent = "mixed"
        else:
            intent = "visual"
    return {"intent": intent}


class Dependencies:
    def __init__(self, *, data, corpus, db, model_mode, model_id, adapter, checkpoint, embedding_model, reranker,
                 risk_model, risk_threshold, maira_results):
        self.data = data
        self.model_mode = model_mode
        self.model_id = model_id
        self.adapter = adapter
        self.checkpoint = checkpoint
        self.chest = ChestRegions(maira_results) if maira_results else None
        self.risk_model = json.loads(risk_model.read_text(encoding="utf-8")) if risk_model else None
        if self.risk_model:
            source = self.risk_model.get("mask_source", "unet")
            if (source == "unet" and not checkpoint) or (source == "maira" and not self.chest):
                raise ValueError("Configure the same region source used by the saved risk model.")
            if source not in {"unet", "maira"}:
                raise ValueError("Unsupported risk model region source.")
        self.risk_threshold = risk_threshold
        self.model_lock = Lock()
        self.retriever = Retriever(corpus, db=db, embedding_model=embedding_model, reranker=reranker)

    @lru_cache(maxsize=1)
    def model(self):
        return VQAPredictor(self.model_id, adapter=self.adapter, device=self.model_mode)

    @lru_cache(maxsize=1)
    def segmenter(self):
        from .segmentation.unet import MaskPredictor

        return MaskPredictor(self.checkpoint, device="cuda" if self.model_mode == "cuda" else "cpu")

    @lru_cache(maxsize=1)
    def modalities(self):
        with Slake(self.data) as slake:
            return {row.image_id: row.modality for split in SPLITS for row in slake.questions(split)}

    def region(self, image_id, image, organ):
        modality = self.modalities().get(image_id)
        if self.chest and (modality, organ) == (CXR, CARDIOMEGALY):
            mask, details = self.chest.region(image_id, image)
            return "maira", mask, details
        if self.checkpoint:
            segmenter = self.segmenter()
            if segmenter.supports(modality, organ):
                return "unet", segmenter.mask(image, organ), None
        return None, None, None

    def can_score(self, source, organ):
        if not self.risk_model or self.risk_model.get("mask_source", "unet") != source:
            return False
        organs = self.risk_model.get("organs")
        if organs:
            return organ in organs
        return source == "unet" and not self.segmenter().multiclass

    @lru_cache(maxsize=1)
    def alignment(self):
        return BiomedAlignment(self.model_mode) if any(key.startswith("alignment_") for key in self.risk_model["features"]) else None


def build_graph(deps, checkpointer):
    def visual(state: State):
        image_id = state.get("image_id")
        if not image_id:
            return {"status": ["Choose a public SLAKE image for visual questions."]}
        if deps.model_mode == "off":
            return {"status": ["Visual inference is disabled in the CPU retrieval demo."]}
        if not (deps.data / "imgs.zip").is_file():
            return {"status": ["SLAKE images are missing. Run python -m slake_vqa data --download."]}
        with Slake(deps.data) as slake:
            try:
                image = slake.image(image_id)
            except KeyError:
                return {"status": ["Image ID not found in SLAKE."]}
            modality = state.get("modality")
            organ = state.get("organ")
            organ = organ.strip().casefold() if organ else None
            with deps.model_lock:
                try:
                    model = deps.model()
                except (ImportError, OSError, RuntimeError):
                    LOGGER.exception("Could not initialize visual inference.")
                    return {"status": ["Visual model unavailable. Check the model path and runtime."]}
                status = []
                if (deps.checkpoint or deps.chest) and modality and organ:
                    source, mask, details = (deps.region(image_id, image, organ)
                                             if modality.strip().upper() == deps.modalities().get(image_id)
                                             else (None, None, None))
                    if source:
                        score_enabled = deps.can_score(source, organ)
                        alignment = deps.alignment() if score_enabled else None
                        summary = probe(model, image, state["question"], mask, align=alignment,
                                        allow_area_control=source == "maira")
                        summary["mask_source"] = source
                        if details is not None:
                            summary["localization"] = details
                        if mask is None:
                            status.append("No MAIRA-2 prediction was exported for this image.")
                        elif not mask.getbbox():
                            status.append("Candidate region is empty.")
                        if summary["target_specific_change"] is None:
                            status.append("Insufficient evidence for a matched-area comparison.")
                        if score_enabled:
                            risk = estimate_risk(summary, deps.risk_model)
                            summary["learned_risk_score"] = risk
                            if risk is not None and deps.risk_threshold is not None and risk >= deps.risk_threshold:
                                status.append("High estimated visual-support risk. Original answer requires review.")
                        return {"image_answer": summary["answer"], "last_visual_answer": summary["answer"],
                                "audit": summary, "status": status}
                    status = ["Requested organ or modality is outside the configured region sources."]
                answer = model.answer(state["question"], image)
                return {"image_answer": answer, "last_visual_answer": answer, "status": status}

    def knowledge(state: State):
        question = state["question"]
        previous = state.get("last_visual_answer")
        if previous and re.search(r"\bit\b|\bthat\b|这种|这个|刚才", question, re.I):
            question += f" {previous}"
        hits = deps.retriever.retrieve(question, limit=3 if state["preference"] == "detailed" else 2)
        if not hits:
            return {"status": [*state["status"], "Insufficient knowledge sources."]}
        answer = "\n\n".join(f"[{i}] {hit['text']}" for i, hit in enumerate(hits, 1))
        status = state["status"]
        if deps.model_mode != "off":
            prompt = ("Answer a general medical imaging education question with the supplied excerpts. "
                      "Cite excerpt numbers as [1] or [2]. If the excerpts do not answer it, say evidence is insufficient. "
                      "Do not infer findings from a patient image.\n" + answer + "\nQuestion: " + question)
            try:
                with deps.model_lock:
                    generated, _ = deps.model().complete(prompt, max_new_tokens=180)
                numbers = re.findall(r"\[(\d+)\]", generated)
                if numbers and all(1 <= int(number) <= len(hits) for number in numbers):
                    answer = generated
            except (ImportError, OSError, RuntimeError):
                LOGGER.exception("Could not generate a cited answer.")
                status = [*status, "Generation unavailable. Showing retrieved passages."]
        return {"knowledge_answer": answer, "passages": [{"citation": i, **hit} for i, hit in enumerate(hits, 1)],
                "status": status}

    graph = StateGraph(State)
    graph.add_node("route", route)
    graph.add_node("visual", visual)
    graph.add_node("knowledge", knowledge)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", lambda state: "knowledge" if state["intent"] == "knowledge" else "visual",
                                {"knowledge": "knowledge", "visual": "visual"})
    graph.add_conditional_edges("visual", lambda state: "knowledge" if state["intent"] == "mixed" else END,
                                {"knowledge": "knowledge", END: END})
    graph.add_edge("knowledge", END)
    return graph.compile(checkpointer=checkpointer)


def create_app(*, data=DATA, corpus=CORPUS, db=DEFAULT_RAG_DB, state_db=GRAPH_STATE,
               model_mode="off", adapter=None, checkpoint=None, embedding_model=None, reranker=None,
               risk_model=None, risk_threshold=None, model_id=MODEL_ID, maira_results=None):
    if risk_model and not (checkpoint or maira_results):
        raise ValueError("A risk model requires a matching region source.")
    if risk_model and risk_threshold is not None and not 0 < risk_threshold < 1:
        raise ValueError("The risk threshold must be in (0,1).")
    @asynccontextmanager
    async def lifespan(app):
        state_db.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(state_db)) as saver, closing(sqlite3.connect(state_db, check_same_thread=False)) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS preferences (token_hash TEXT PRIMARY KEY, style TEXT NOT NULL)")
            conn.commit()
            deps = Dependencies(data=data, corpus=corpus, db=db, model_mode=model_mode, model_id=model_id, adapter=adapter,
                                checkpoint=checkpoint, embedding_model=embedding_model, reranker=reranker,
                                risk_model=risk_model, risk_threshold=risk_threshold, maira_results=maira_results)
            app.state.graph = build_graph(deps, saver)
            app.state.conn = conn
            app.state.deps = deps
            try:
                yield
            finally:
                deps.retriever.close()

    from fastapi import Header

    app = FastAPI(title="MedVQA-Evidence", lifespan=lifespan)
    sql_lock = Lock()
    graph_lock = Lock()

    def lookup(token):
        if not token:
            raise HTTPException(401, "Create a demo profile first.")
        key = sha256(token.encode()).hexdigest()
        with sql_lock:
            row = app.state.conn.execute("SELECT style FROM preferences WHERE token_hash = ?", (key,)).fetchone()
        if row is None:
            raise HTTPException(401, "Unknown demo profile token.")
        return key, row[0]

    @app.get("/health")
    def health():
        return {"status": "ready", "visual_inference": model_mode, "segmentation_configured": checkpoint is not None,
                "maira_configured": maira_results is not None}

    @app.get("/", response_class=HTMLResponse)
    def home():
        return Path(__file__).with_name("demo.html").read_text(encoding="utf-8")

    @app.get("/images/{image_id}")
    def public_image(image_id: str, organ: str | None = None):
        try:
            with Slake(data) as slake:
                image = slake.image(image_id)
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(404, "Public SLAKE image unavailable.") from exc
        if organ:
            deps = app.state.deps
            with deps.model_lock:
                source, mask, _ = deps.region(image_id, image, organ.strip().casefold())
            if source is None:
                raise HTTPException(400, "This target or image modality is unsupported by the configured region sources.")
            if mask is None:
                raise HTTPException(404, "No MAIRA-2 prediction was exported for this image.")
            if not mask.getbbox():
                raise HTTPException(422, "The model predicted an empty audit region.")
            tint = Image.blend(image, Image.new("RGB", image.size, "#31c1a2"), 0.55)
            image = Image.composite(tint, image, mask)
        image.thumbnail((800, 800))
        with BytesIO() as stream:
            image.save(stream, format="JPEG", quality=90)
            return Response(stream.getvalue(), media_type="image/jpeg")

    @app.post("/profiles", status_code=201)
    def create_profile():
        token = token_urlsafe(32)
        with sql_lock:
            app.state.conn.execute("INSERT INTO preferences VALUES (?, ?)", (sha256(token.encode()).hexdigest(), "concise"))
            app.state.conn.commit()
        return {"profile_token": token}

    @app.put("/preferences")
    def set_preference(body: Preference, x_profile_token: str | None = Header(default=None)):
        key, _ = lookup(x_profile_token)
        with sql_lock:
            app.state.conn.execute("UPDATE preferences SET style = ? WHERE token_hash = ?", (body.style, key))
            app.state.conn.commit()
        return body.model_dump()

    @app.post("/ask")
    def ask(body: Ask, x_profile_token: str | None = Header(default=None)):
        key, style = lookup(x_profile_token)
        if not body.question.strip():
            raise HTTPException(422, "The question is empty.")
        thread = str(body.thread_id or uuid4())
        config = {"configurable": {"thread_id": f"{key}:{thread}"}}
        request = {"question": body.question.strip(), "requested_intent": body.intent, "image_id": body.image_id,
                   "organ": body.organ, "modality": body.modality, "preference": style,
                   "audit": None, "status": [], "image_answer": None, "knowledge_answer": None, "passages": []}
        started = perf_counter()
        with graph_lock:
            result = app.state.graph.invoke(request, config)
        return {"thread_id": thread, "style": style, "latency_ms": round((perf_counter() - started) * 1000, 1),
                **{name: result.get(name) for name in ("intent", "image_answer", "knowledge_answer", "passages", "audit", "status")}}

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["off", "cuda", "cpu"], default="off")
    parser.add_argument("--model-id", default=default_model(), help="Hugging Face model ID or a prepared local directory.")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--maira-results", type=Path)
    parser.add_argument("--risk-model", type=Path)
    parser.add_argument("--risk-threshold", type=float)
    parser.add_argument("--embedding-model")
    parser.add_argument("--reranker")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app(data=args.data, corpus=args.corpus, model_mode=args.model, model_id=args.model_id, adapter=args.adapter,
                     checkpoint=args.checkpoint, embedding_model=args.embedding_model, reranker=args.reranker,
                     risk_model=args.risk_model, risk_threshold=args.risk_threshold, maira_results=args.maira_results)
    uvicorn.run(app, host=os.environ.get("EVIDENCE_HOST", "127.0.0.1"), port=args.port)


if __name__ == "__main__":
    main()
