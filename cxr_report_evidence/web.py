"""Serve the local chest X-ray research workflow through FastAPI and LangGraph."""

import argparse
import base64
import io
import logging
import os
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TypedDict

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from langgraph.graph import END, START, StateGraph
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from evidence.rag.retrieval import CORPUS, Retriever
from evidence.storage import digest_image, file_digest, write_json

from . import paths
from .audit import interventions
from .data import VARIANTS, image_at, load_manifest, read_json, samples

LOG = logging.getLogger(__name__)
UPLOAD_BYTES = 12 * 1024 * 1024


class State(TypedDict, total=False):
    intent: str
    study_id: str
    question: str
    image_bytes: bytes
    result: dict


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=800)


def execute(python, *args):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    try:
        subprocess.run([str(python), "-m", "cxr_report_evidence", *map(str, args)],
                       env=env, check=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        LOG.error("Worker for %s failed (%s)", args[0], type(error).__name__)
        raise HTTPException(503, f"{args[0]} worker failed. Check the server terminal.") from error


def jpeg(image, edge):
    buffer = io.BytesIO()
    image = image.copy()
    image.thumbnail((edge, edge))
    image.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def display_result(report, reference, probes):
    claims = [dict(row) for row in (probes["rows"] if probes else []) if row["study_id"] == report["study_id"]]
    return {"study_id": report["study_id"], "reports": report["reports"], "reference": reference,
            "audit_loaded": probes is not None, "claims": claims}


def create_app(args):
    studies = {row["study_id"]: row for row in load_manifest(args.manifest)} if args.manifest else {}
    artifact = read_json(args.reports) if args.reports else None
    probes = read_json(args.probes) if args.probes else None
    if artifact and (not args.manifest or artifact["config"]["manifest_sha256"] != file_digest(args.manifest)):
        raise ValueError("Reports were generated with a different manifest.")
    if probes and (not args.reports or probes["config"]["reports_sha256"] != file_digest(args.reports)):
        raise ValueError("Audit probes were produced from different reports.")
    reports = {row["study_id"]: row for row in artifact["rows"]} if artifact else {}
    claims = {row["claim_id"]: row for row in probes["rows"]} if probes else {}
    live = bool(args.qwen and args.adapter and args.maira and args.maira_python)
    inference_lock = Lock()
    coverage = None
    if artifact:
        config = artifact["config"]
        planned = samples(list(studies.values()), config["split"], config["limit"])
        coverage = {
            "planned": len(planned),
            "complete": sum(set(reports.get(row["study_id"], {}).get("reports", ())) == set(VARIANTS)
                            for row in planned),
            "counts": {name: sum(name in row["reports"] for row in reports.values()) for name in VARIANTS},
            "has_probes": bool(probes),
            "device": config.get("device"),
        }

    def lookup(ident):
        if ident not in reports or ident not in studies:
            raise HTTPException(404, "Study not found in the prepared report results.")
        return studies[ident], reports[ident]

    def study_image(ident):
        study, report = lookup(ident)
        image = image_at(args.images, study["image"])
        if digest_image(image) != report["image_sha256"]:
            raise HTTPException(409, "This image changed after report generation.")
        return image

    def archived(state):
        study, report = lookup(state["study_id"])
        result = display_result(report, study["reference"], probes)
        return {"result": {**result, "view": study["view"], "split": study["split"],
                           "preview_size": list(study_image(study["study_id"]).size)}}

    def uploaded(state):
        with inference_lock, TemporaryDirectory(prefix="cxr-upload-") as directory:
            root = Path(directory)
            try:
                with Image.open(io.BytesIO(state["image_bytes"]), formats=("PNG", "JPEG")) as raw:
                    if raw.width * raw.height > 24_000_000:
                        raise HTTPException(413, "Maximum image size is 24 megapixels.")
                    image = ImageOps.exif_transpose(raw).convert("RGB")
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                raise HTTPException(400, "Choose a readable frontal X-ray JPG or PNG.") from error
            image.save(root / "upload.png")
            manifest, predictions, grounding, review = [root / name for name in ("study.json", "reports.json", "ground.json", "probe.json")]
            write_json(manifest, {"studies": [{"study_id": "upload", "patient_id": "upload", "split": "test",
                                               "image": "upload.png", "reference": ""}]})
            data = ["--manifest", manifest, "--images", root]
            execute(sys.executable, "generate", *data, "--model", args.qwen, "--adapter", args.adapter,
                    "--variants", "lora_image", "lora_text", "--output", predictions)
            execute(args.maira_python, "ground", *data, "--reports", predictions, "--model", args.maira,
                    "--max-findings", args.max_findings, "--output", grounding)
            execute(sys.executable, "probe", *data, "--reports", predictions, "--grounding", grounding,
                    "--model", args.qwen, "--output", review)
            report = read_json(predictions)["rows"][0]
            result = display_result(report, None, read_json(review))
            for row in result["claims"]:
                if row["control_match"]:
                    target, control, _ = interventions(image, row["boxes"], row["radius"])
                    row["previews"] = {name: "data:image/jpeg;base64," + base64.b64encode(jpeg(variant, 700)).decode("ascii")
                                       for name, variant in (("target", target), ("control", control))}
            return {"result": {**result, "preview_size": list(image.size)}}

    def knowledge(state):
        with closing(Retriever(args.corpus, db=None)) as retriever:
            hits = retriever.retrieve(state["question"], limit=3)
        if not hits:
            return {"result": {"answer": None, "sources": [], "mode": "unavailable"}}
        answer = None
        if args.rag_generate:
            with inference_lock, TemporaryDirectory(prefix="cxr-rag-") as directory:
                request, output = Path(directory) / "request.json", Path(directory) / "answer.json"
                write_json(request, {"question": state["question"], "passages": hits})
                try:
                    execute(sys.executable, "knowledge", "--request", request, "--model", args.qwen, "--output", output)
                    answer = read_json(output)["answer"]
                except HTTPException:
                    LOG.warning("Using local extracts after generation failed.")
        return {"result": {"answer": answer or "\n\n".join(f"[{i}] {hit['text']}" for i, hit in enumerate(hits, 1)),
                           "sources": hits, "mode": "generated" if answer else "extracts"}}

    workflow = StateGraph(State)
    workflow.add_node("study", archived)
    workflow.add_node("upload", uploaded)
    workflow.add_node("knowledge", knowledge)
    workflow.add_conditional_edges(START, lambda state: state["intent"], {name: name for name in ("study", "upload", "knowledge")})
    for name in ("study", "upload", "knowledge"):
        workflow.add_edge(name, END)
    graph = workflow.compile()
    app = FastAPI(title="CXR Report Evidence")

    @app.middleware("http")
    async def local_responses(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/", response_class=HTMLResponse)
    def page():
        return Path(__file__).with_name("demo.html").read_text(encoding="utf-8")

    @app.get("/api/studies")
    def choices():
        return {"live": live, "coverage": coverage,
                "studies": [{"id": ident, "split": studies[ident]["split"], "view": studies[ident]["view"]}
                            for ident in reports]}

    @app.get("/api/study/{ident}")
    def inspect(ident: str):
        return graph.invoke({"intent": "study", "study_id": ident})["result"]

    @app.get("/api/image/{ident}")
    def preview(ident: str, claim: str | None = None, variant: str = "original"):
        image = study_image(ident)
        if variant != "original":
            row = claims.get(claim)
            if variant not in {"target", "control"} or not row or row["study_id"] != ident or not row["control_match"]:
                raise HTTPException(404, "A matched perturbation is unavailable for this finding.")
            target, control, _ = interventions(image, row["boxes"], row["radius"])
            image = target if variant == "target" else control
        return Response(jpeg(image, 1200), media_type="image/jpeg")

    @app.post("/api/knowledge")
    def ask(item: Question):
        return graph.invoke({"intent": "knowledge", "question": item.question})["result"]

    @app.post("/api/live")
    async def upload(request: Request):
        if not live:
            raise HTTPException(503, "Live GPU processing is not configured.")
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > UPLOAD_BYTES:
                raise HTTPException(413, "Maximum upload size is 12 MiB.")
            payload.extend(chunk)
        result = await run_in_threadpool(graph.invoke, {"intent": "upload", "image_bytes": bytes(payload)})
        return result["result"]

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "images", "reports", "probes", "qwen", "adapter", "maira", "maira-python"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--local-results", action="store_true", help="Open the local report results and available audit.")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--max-findings", type=int, default=4)
    parser.add_argument("--rag-generate", action="store_true")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    if args.local_results:
        args.manifest = args.manifest or paths.MANIFEST
        args.images = args.images or paths.IMAGES
        args.reports = args.reports or paths.results_run() / "reports.json"
        if not args.reports.is_file():
            parser.error(f"No reports found at {args.reports}. Run python -m cxr_report_evidence generate first.")
        probes = args.reports.with_name("probes.json")
        if args.probes is None and probes.is_file():
            args.probes = probes
    if args.reports and not (args.manifest and args.images):
        parser.error("Archive mode requires --manifest and --images with --reports.")
    if args.rag_generate and not args.qwen:
        parser.error("--rag-generate requires --qwen.")
    if args.max_findings < 1:
        parser.error("--max-findings must be positive.")
    uvicorn.run(create_app(args), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
