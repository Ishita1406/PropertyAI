"""FastAPI backend for the Property Investment Advisor."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

# Get the project root (where src/ is located)
PROJECT_ROOT = Path(__file__).parent

# Add src to Python path if not already there
src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

# Now import from property_advisor (which is in src/)
from property_advisor.config import API_HOST, API_PORT
from property_advisor.graph import build_graph
from property_advisor.report_generator import generate_reports

# Set up directories
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
FRONTEND_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Frontend directory: {FRONTEND_DIR}")
print(f"📁 Reports directory: {REPORTS_DIR}")

app = FastAPI(title="Property Alpha AI", description="Indian Property Investment Advisor API")

# CORS policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = build_graph()

FRIENDLY_AI_ERROR = "AI recommendation service temporarily unavailable."


class AnalyzeRequest(BaseModel):
    address: str = Field(..., min_length=1)
    budget: float = Field(..., gt=0)
    horizon: int = Field(default=5, gt=0, le=50)
    strategy: str = Field(default="rental")


class ApproveRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)


class RejectRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    feedback: str = Field(default="")


def _invoke(config: dict, command_or_state) -> dict:
    try:
        return _graph.invoke(command_or_state, config)
    except Exception as exc:
        print(f"❌ Error in graph invoke: {exc}")
        raise HTTPException(status_code=503, detail=FRIENDLY_AI_ERROR) from exc


def _pending_approval_payload(thread_id: str, interrupt_value: dict) -> dict:
    return {
        "approval_required": True,
        "thread_id": thread_id,
        "property_address": interrupt_value.get("property_address", ""),
        "recommendation": interrupt_value.get("recommendation", {}),
        "investment_metrics": interrupt_value.get("investment_metrics", {}),
        "risk_assessment": interrupt_value.get("risk_assessment", {}),
        "guardrail_result": interrupt_value.get("guardrail_result", {}),
        # Added in the two-agent refactor (additive; existing keys unchanged).
        "research_findings": interrupt_value.get("research_findings", {}),
        "data_provenance": interrupt_value.get("data_provenance", {}),
        "data_disclaimer": interrupt_value.get("data_disclaimer", ""),
    }


def _report_download_paths(paths: dict) -> dict:
    return {
        "json": f"/reports/{os.path.basename(paths['json'])}",
        "markdown": f"/reports/{os.path.basename(paths['md'])}",
        "pdf": f"/reports/{os.path.basename(paths['pdf'])}",
    }


# ADD THIS HEALTH ENDPOINT
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "project_root": str(PROJECT_ROOT),
        "frontend_exists": FRONTEND_DIR.exists(),
        "frontend_index": (FRONTEND_DIR / "index.html").exists(),
        "reports_exists": REPORTS_DIR.exists()
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> dict:
    try:
        if req.strategy not in ("rental", "flip", "long_term_appreciation"):
            raise HTTPException(
                status_code=400,
                detail="strategy must be one of: rental, flip, long_term_appreciation"
            )

        thread_id = f"api-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "property_address": req.address,
            "budget": req.budget,
            "investment_horizon_years": req.horizon,
            "investment_strategy": req.strategy,
        }

        result = _invoke(config, initial_state)

        if "__interrupt__" in result:
            return _pending_approval_payload(thread_id, result["__interrupt__"][0].value)

        return {"approval_required": False, "final_report": result.get("final_report", {})}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in /analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/approve")
async def approve(req: ApproveRequest) -> dict:
    try:
        config = {"configurable": {"thread_id": req.thread_id}}
        result = _invoke(config, Command(resume={"approved": True, "feedback": ""}))

        while "__interrupt__" in result:
            result = _invoke(config, Command(resume={"approved": True, "feedback": ""}))

        final_report = result.get("final_report", {})
        response: dict = {"final_report": final_report}

        if final_report.get("status") == "approved":
            try:
                paths = generate_reports(final_report, output_dir=str(REPORTS_DIR))
                response["report_paths"] = _report_download_paths(paths)
            except Exception as e:
                print(f"❌ Report generation error: {str(e)}")
                response["report_error"] = "Analysis completed but report generation failed."

        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in /approve: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/reject")
async def reject(req: RejectRequest) -> dict:
    try:
        config = {"configurable": {"thread_id": req.thread_id}}
        result = _invoke(config, Command(resume={"approved": False, "feedback": req.feedback}))

        if "__interrupt__" in result:
            return _pending_approval_payload(req.thread_id, result["__interrupt__"][0].value)

        return {"approval_required": False, "final_report": result.get("final_report", {})}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in /reject: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return {
            "error": "index.html not found",
            "frontend_dir": str(FRONTEND_DIR),
            "path": str(index_path)
        }
    return FileResponse(str(index_path))


# Mount static directories
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


if __name__ == "__main__":
    import threading
    import webbrowser

    import uvicorn

    # NOTE: reload=True cannot be combined with passing the app object —
    # uvicorn prints a warning and exits without serving anything.
    url = f"http://localhost:{API_PORT}/"
    print(f"🌐 Open the app at {url}")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")