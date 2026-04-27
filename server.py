"""
Video QA API Server

FastAPI backend that connects the React frontend to the Moment-DETR pipeline.

Usage:
    pip install fastapi uvicorn python-multipart
    python3 server.py --ckpt results/hl-video_tef-exp-xxx/model_best.ckpt

Then open the frontend at the URL shown in the terminal.
"""
import os
import sys
import shutil
import tempfile
import argparse
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import VideoQAPipeline

# ============================================================
# App
# ============================================================

app = FastAPI(title="Video QA API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = None  # initialized in startup


@app.post("/query")
async def query_video(
    video: UploadFile = File(...),
    query: str = Form(...),
    threshold: float = Form(0.5),
):
    """
    Upload a video and ask a question.
    Returns retrieved moments and an LLM-generated answer.
    """
    # Save uploaded video to temp file
    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    try:
        result = pipe.run(tmp_path, query, threshold=threshold)
        return JSONResponse({
            "query": result["query"],
            "answer": result["answer"],
            "moments": result["moments"],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipe is not None}


# ============================================================
# Main
# ============================================================

def main():
    global pipe

    parser = argparse.ArgumentParser(description="Video QA API Server")
    parser.add_argument("--ckpt", required=True, help="Moment-DETR checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openai_key", default=None, help="OpenAI API key")
    parser.add_argument("--llm", default="gpt-4o-mini")
    parser.add_argument("--no_slowfast", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("Initializing pipeline...")
    pipe = VideoQAPipeline(
        ckpt_path=args.ckpt,
        device=args.device,
        openai_key=args.openai_key,
        llm_model=args.llm,
        use_slowfast=not args.no_slowfast,
    )

    print(f"\nServer starting at http://localhost:{args.port}")
    print(f"API docs at http://localhost:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()