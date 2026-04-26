"""
Video QA Pipeline: Raw Video → Feature Extraction → Moment Retrieval → LLM Answer

End-to-end pipeline:
  Stage 1: Feature Extraction - raw video → CLIP + SlowFast features
  Stage 2: Moment Retrieval   - features + query → top-k relevant moments
  Stage 3: Keyframe Extraction- raw video + moments → keyframes for VLM
  Stage 4: Answer Generation  - query + moments + keyframes → natural language answer

Usage:
    # Basic (template answer, no API key needed)
    python3 pipeline.py \
        --ckpt results/hl-video_tef-exp-xxx/model_best.ckpt \
        --query "What is the person doing?" \
        --video /path/to/video.mp4

    # With LLM answer generation
    python3 pipeline.py \
        --ckpt results/hl-video_tef-exp-xxx/model_best.ckpt \
        --query "What is the person doing?" \
        --video /path/to/video.mp4 \
        --openai_key sk-xxx

    # Interactive mode
    python3 pipeline.py \
        --ckpt results/hl-video_tef-exp-xxx/model_best.ckpt \
        --interactive
"""
import os
import sys
import base64
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moment_detr.inference import setup_model
from utils.basic_utils import l2_normalize_np_array
from extract_features import CLIPExtractor, SlowFastExtractor


# ============================================================
# Stage 1: Feature Extraction (raw video → CLIP + SlowFast)
# ============================================================

class FeatureExtractor:
    """Extract CLIP + SlowFast features from a raw video."""

    def __init__(self, device="cuda", use_slowfast=True):
        self.device = device
        self.clip_extractor = CLIPExtractor(device=device)

        self.sf_extractor = None
        if use_slowfast:
            try:
                self.sf_extractor = SlowFastExtractor(device=device)
            except Exception as e:
                print(f"  Warning: SlowFast not available ({e}), using CLIP only")

    def extract(self, video_path, clip_length=2):
        """
        Extract features from a raw video.

        Returns:
            video_feat: [num_clips, feat_dim] (2816 if CLIP+SF, 512 if CLIP only)
            duration: video duration in seconds
        """
        clip_feat, num_clips, duration = self.clip_extractor.extract(video_path, clip_length)

        if self.sf_extractor is not None:
            sf_feat, _, _ = self.sf_extractor.extract(video_path, clip_length)
            min_len = min(len(clip_feat), len(sf_feat))
            video_feat = np.concatenate([sf_feat[:min_len], clip_feat[:min_len]], axis=1)
        else:
            video_feat = clip_feat

        return video_feat, duration

    def extract_text(self, query):
        """Encode a text query with CLIP. Returns [1, 512]."""
        return self.clip_extractor.extract_text_features(query)


# ============================================================
# Stage 2: Moment Retrieval (Moment-DETR)
# ============================================================

def build_model_inputs(video_feat, text_feat, device="cuda"):
    """Build Moment-DETR model inputs with TEF."""
    num_clips = video_feat.shape[0]

    tef_start = np.arange(0, num_clips, 1.0) / num_clips
    tef_end = tef_start + 1.0 / num_clips
    tef = np.stack([tef_start, tef_end], axis=1)
    video_feat_tef = np.concatenate([video_feat, tef], axis=1)

    return {
        "src_vid": torch.from_numpy(video_feat_tef).float().unsqueeze(0).to(device),
        "src_vid_mask": torch.ones(1, num_clips).bool().to(device),
        "src_txt": torch.from_numpy(text_feat).float().unsqueeze(0).to(device),
        "src_txt_mask": torch.ones(1, text_feat.shape[0]).bool().to(device),
    }, num_clips


def retrieve_moments(model, video_feat, text_feat, clip_length=2, device="cuda", top_k=5):
    """Run Moment-DETR to retrieve relevant video moments."""
    model.eval()
    model_inputs, num_clips = build_model_inputs(video_feat, text_feat, device)

    with torch.no_grad():
        outputs = model(**model_inputs)

    pred_spans = outputs["pred_spans"][0]
    pred_logits = outputs["pred_logits"][0]
    scores = torch.softmax(pred_logits, dim=-1)[:, 0]

    duration = num_clips * clip_length
    moments = []
    for i in range(len(pred_spans)):
        center = pred_spans[i][0].item() * duration
        width = pred_spans[i][1].item() * duration
        moments.append({
            "start": round(max(0, center - width / 2), 1),
            "end": round(min(duration, center + width / 2), 1),
            "score": round(scores[i].item(), 4),
        })

    moments.sort(key=lambda x: x["score"], reverse=True)

    saliency = []
    if "saliency_scores" in outputs:
        saliency = outputs["saliency_scores"][0].cpu().numpy().tolist()

    return moments[:top_k], saliency


def load_moment_detr(ckpt_path, device="cuda"):
    """Load trained Moment-DETR from checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    opt = checkpoint["opt"]
    opt.device = torch.device(device)

    model, _, _, _ = setup_model(opt)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


# ============================================================
# Stage 3: Keyframe Extraction
# ============================================================

def extract_keyframes(video_path, moments, frames_per_moment=2, max_frames=8):
    """Extract keyframes from retrieved moments as base64 strings."""
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    total_extracted = 0

    for moment in moments:
        if total_extracted >= max_frames:
            break

        frames_b64 = []
        n = min(frames_per_moment, max_frames - total_extracted)
        timestamps = np.linspace(moment["start"], moment["end"], n + 2)[1:-1]

        for t in timestamps:
            frame_idx = min(int(t * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                if max(h, w) > 512:
                    scale = 512 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames_b64.append(base64.b64encode(buf).decode("utf-8"))
                total_extracted += 1

        if frames_b64:
            results.append({"moment": moment, "frames": frames_b64})

    cap.release()
    return results


# ============================================================
# Stage 4: LLM Answer Generation
# ============================================================

def generate_answer(query, moments, frame_data=None, api_key=None, model_name="gpt-4o-mini"):
    """Generate answer using LLM, or fall back to template."""
    if not api_key:
        return _template_answer(query, moments)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        return _template_answer(query, moments)

    system_prompt = """You are a video analysis assistant. The user asks a question about a video, 
and the system has retrieved the most relevant video segments.

Your task:
1. Answer the question based on the retrieved segments and frame images (if provided)
2. Reference specific timestamps so the user knows where to look
3. If the evidence is insufficient, honestly say so
4. Be concise and direct"""

    evidence = "Retrieved video segments:\n"
    for i, m in enumerate(moments, 1):
        evidence += f"  Segment {i}: {m['start']:.1f}s - {m['end']:.1f}s (relevance: {m['score']:.2%})\n"

    if frame_data and "4o" in model_name:
        content = [{"type": "text", "text": f"Question: {query}\n\n{evidence}\nKeyframes:"}]
        for item in frame_data[:5]:
            m = item["moment"]
            for b64 in item["frames"]:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                })
            content.append({
                "type": "text",
                "text": f"[{m['start']:.1f}s - {m['end']:.1f}s, relevance: {m['score']:.2%}]"
            })
        content.append({"type": "text", "text": "\nAnswer the question based on the above."})
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}\n\n{evidence}\nPlease answer."},
        ]

    response = client.chat.completions.create(
        model=model_name, messages=messages, max_tokens=1024, temperature=0.3,
    )
    return response.choices[0].message.content


def _template_answer(query, moments):
    if not moments:
        return f"No relevant moments found for: '{query}'"
    answer = f"For the query '{query}', the following relevant moments were found:\n\n"
    for i, m in enumerate(moments, 1):
        answer += f"  {i}. {m['start']:.1f}s - {m['end']:.1f}s (relevance: {m['score']:.2%})\n"
    answer += f"\nMost relevant content is at {moments[0]['start']:.1f}s - {moments[0]['end']:.1f}s."
    return answer


# ============================================================
# Pipeline
# ============================================================

class VideoQAPipeline:
    """
    End-to-end: raw video + query → answer.

    Usage:
        pipe = VideoQAPipeline("results/.../model_best.ckpt")
        result = pipe.run("video.mp4", "What is the person doing?")
        print(result["answer"])
    """

    def __init__(self, ckpt_path, device="cuda", openai_key=None,
                 llm_model="gpt-4o-mini", use_slowfast=True):
        self.device = device
        self.openai_key = openai_key or os.environ.get("OPENAI_API_KEY")
        self.llm_model = llm_model

        print("Initializing pipeline...")
        print(f"  Loading Moment-DETR from {ckpt_path}")
        self.model = load_moment_detr(ckpt_path, device)

        print(f"  Loading feature extractors...")
        self.feat = FeatureExtractor(device=device, use_slowfast=use_slowfast)

        print("Pipeline ready!\n")

    def run(self, video_path, query, top_k=5):
        """
        Run the full pipeline on a raw video.

        Args:
            video_path: path to video file
            query: question about the video
            top_k: number of moments to retrieve

        Returns:
            dict with "query", "answer", "moments", "saliency"
        """
        # Stage 1: Extract features
        print(f"[Stage 1] Extracting features from {os.path.basename(video_path)}...")
        video_feat, duration = self.feat.extract(video_path)
        text_feat = self.feat.extract_text(query)
        print(f"  Video: {duration:.1f}s, {video_feat.shape[0]} clips, feature dim {video_feat.shape[1]}")

        # Stage 2: Retrieve moments
        print(f"[Stage 2] Retrieving moments for: '{query}'")
        moments, saliency = retrieve_moments(
            self.model, video_feat, text_feat, device=self.device, top_k=top_k
        )
        for i, m in enumerate(moments, 1):
            print(f"  {i}. {m['start']:.1f}s - {m['end']:.1f}s (score: {m['score']:.4f})")

        # Stage 3: Extract keyframes
        print(f"[Stage 3] Extracting keyframes...")
        frame_data = extract_keyframes(video_path, moments)
        if frame_data:
            n = sum(len(item["frames"]) for item in frame_data)
            print(f"  Extracted {n} frames from {len(frame_data)} moments")

        # Stage 4: Generate answer
        print(f"[Stage 4] Generating answer...")
        answer = generate_answer(
            query, moments, frame_data,
            api_key=self.openai_key, model_name=self.llm_model,
        )

        return {
            "query": query,
            "answer": answer,
            "moments": moments,
            "saliency": saliency,
        }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Video QA Pipeline")
    parser.add_argument("--ckpt", required=True, help="Moment-DETR checkpoint")
    parser.add_argument("--video", default=None, help="Video file path")
    parser.add_argument("--query", default=None, help="Question about the video")
    parser.add_argument("--openai_key", default=None, help="OpenAI API key")
    parser.add_argument("--llm", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_slowfast", action="store_true", help="Skip SlowFast, use CLIP only")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    pipe = VideoQAPipeline(
        ckpt_path=args.ckpt,
        device=args.device,
        openai_key=args.openai_key,
        llm_model=args.llm,
        use_slowfast=not args.no_slowfast,
    )

    if args.interactive:
        print("=" * 50)
        print("  Video QA - Interactive Mode")
        print("  Type 'quit' to exit, 'video' to change video")
        print("=" * 50)

        video_path = None
        while True:
            if video_path is None:
                video_path = input("\nVideo path: ").strip()
                if video_path.lower() == "quit":
                    break
                if not os.path.exists(video_path):
                    print(f"File not found: {video_path}")
                    video_path = None
                    continue

            query = input("Question: ").strip()
            if query.lower() == "quit":
                break
            if query.lower() == "video":
                video_path = None
                continue

            try:
                result = pipe.run(video_path, query, top_k=args.top_k)
                print(f"\n{'='*50}")
                print(f"Answer:\n{result['answer']}")
                print(f"{'='*50}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        if not args.video or not args.query:
            parser.print_help()
            print("\nError: --video and --query are required")
            return

        result = pipe.run(args.video, args.query, top_k=args.top_k)

        print(f"\n{'='*50}")
        print(f"Question: {result['query']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nRetrieved Moments:")
        for i, m in enumerate(result["moments"], 1):
            print(f"  {i}. {m['start']:.1f}s - {m['end']:.1f}s (score: {m['score']:.4f})")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
