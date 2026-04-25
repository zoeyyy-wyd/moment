"""
Moment-DETR Inference Script (CLIP + SlowFast)

用训练好的 model_best.ckpt 对视频做推理。

Usage:
    # 对已有特征的视频推理
    python3 inference_sf.py \
        --ckpt results/hl-video_tef-exp-xxx/model_best.ckpt \
        --query "a person riding a bike" \
        --clip_feat features/clip_features/VIDEO_ID.npz \
        --slowfast_feat features/slowfast_features/VIDEO_ID.npz

    # 批量推理 (交互模式)
    python3 inference_sf.py \
        --ckpt results/hl-video_tef-exp-xxx/model_best.ckpt \
        --interactive
"""
import os
import sys
import argparse
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moment_detr.config import BaseOptions
from moment_detr.inference import setup_model
from utils.basic_utils import l2_normalize_np_array


def load_features(clip_feat_path, slowfast_feat_path=None, normalize=True):
    """
    加载并拼接 CLIP + SlowFast 特征。

    Returns:
        video_features: [1, num_clips, feat_dim] tensor
    """
    # Load CLIP features
    clip_data = np.load(clip_feat_path, allow_pickle=True)
    clip_keys = list(clip_data.keys())
    clip_feat = clip_data[clip_keys[0]]  # [N, 512]

    if normalize:
        clip_feat = l2_normalize_np_array(clip_feat)

    # Load SlowFast features (if provided)
    if slowfast_feat_path and os.path.exists(slowfast_feat_path):
        sf_data = np.load(slowfast_feat_path, allow_pickle=True)
        sf_keys = list(sf_data.keys())
        sf_feat = sf_data[sf_keys[0]]  # [N, 2304]

        if normalize:
            sf_feat = l2_normalize_np_array(sf_feat)

        # Align clip counts
        min_len = min(len(clip_feat), len(sf_feat))
        clip_feat = clip_feat[:min_len]
        sf_feat = sf_feat[:min_len]

        # Concatenate: [N, 2816]
        video_feat = np.concatenate([sf_feat, clip_feat], axis=1)
    else:
        video_feat = clip_feat

    return video_feat


def load_text_features(text_feat_path, normalize=True):
    """加载预计算的 text features。"""
    data = np.load(text_feat_path, allow_pickle=True)
    keys = list(data.keys())
    feat = data[keys[0]]

    if normalize:
        feat = l2_normalize_np_array(feat)

    return feat


def build_model_inputs(video_feat, text_feat, clip_length=2, device="cuda"):
    """
    构建 Moment-DETR 模型输入。

    Args:
        video_feat: [N, feat_dim] numpy array
        text_feat: [L, 512] numpy array
        clip_length: seconds per clip
        device: cuda or cpu
    """
    num_clips = video_feat.shape[0]
    duration = num_clips * clip_length

    # Append TEF (Temporal Encoding Features): [start/duration, end/duration]
    # ctx_mode=video_tef 要求每个 clip 拼接 2 维时间位置编码
    tef_start = np.arange(0, num_clips, 1.0) / num_clips
    tef_end = tef_start + 1.0 / num_clips
    tef = np.stack([tef_start, tef_end], axis=1)  # [N, 2]
    video_feat_with_tef = np.concatenate([video_feat, tef], axis=1)  # [N, feat_dim+2]

    # Video features tensor
    vid_feat = torch.from_numpy(video_feat_with_tef).float().unsqueeze(0).to(device)  # [1, N, D+2]

    # Text features tensor
    txt_feat = torch.from_numpy(text_feat).float().unsqueeze(0).to(device)  # [1, L, 512]

    # Video mask (all valid)
    vid_mask = torch.ones(1, num_clips).bool().to(device)

    # Text mask (all valid)
    txt_mask = torch.ones(1, text_feat.shape[0]).bool().to(device)

    model_inputs = {
        "src_vid": vid_feat,
        "src_vid_mask": vid_mask,
        "src_txt": txt_feat,
        "src_txt_mask": txt_mask,
    }

    return model_inputs, num_clips


def predict(model, video_feat, text_feat, clip_length=2, device="cuda"):
    """
    运行推理，返回预测的 moments 和 saliency scores。

    Returns:
        moments: List of [start_sec, end_sec, score]
        saliency: List of per-clip saliency scores
    """
    model.eval()

    model_inputs, num_clips = build_model_inputs(
        video_feat, text_feat, clip_length, device
    )

    with torch.no_grad():
        outputs = model(**model_inputs)

    # Parse predictions
    # Moment predictions: [1, num_queries, 2] (center, width) normalized
    pred_spans = outputs["pred_spans"][0]  # [num_queries, 2]
    pred_logits = outputs["pred_logits"][0]  # [num_queries, 2] (foreground/background)

    # Convert to probabilities
    prob = torch.softmax(pred_logits, dim=-1)
    scores = prob[:, 0]  # foreground probability

    # Convert (center, width) to (start, end) in seconds
    duration = num_clips * clip_length
    moments = []
    for i in range(len(pred_spans)):
        center = pred_spans[i][0].item() * duration
        width = pred_spans[i][1].item() * duration
        start = max(0, center - width / 2)
        end = min(duration, center + width / 2)
        score = scores[i].item()
        moments.append([start, end, score])

    # Sort by score
    moments.sort(key=lambda x: x[2], reverse=True)

    # Saliency scores
    if "saliency_scores" in outputs:
        saliency = outputs["saliency_scores"][0].cpu().numpy().tolist()
    else:
        saliency = []

    return moments, saliency


def load_model(ckpt_path, device="cuda"):
    """加载训练好的模型。"""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    opt = checkpoint["opt"]
    opt.device = torch.device(device)

    model, _, _, _ = setup_model(opt)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, opt


def main():
    parser = argparse.ArgumentParser(description="Moment-DETR Inference (CLIP+SlowFast)")
    parser.add_argument("--ckpt", required=True, help="Path to model_best.ckpt")
    parser.add_argument("--query", default=None, help="Text query")
    parser.add_argument("--clip_feat", default=None, help="Path to CLIP feature .npz")
    parser.add_argument("--slowfast_feat", default=None, help="Path to SlowFast feature .npz")
    parser.add_argument("--text_feat", default=None, help="Path to pre-computed text feature .npz")
    parser.add_argument("--clip_length", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.ckpt}...")
    model, opt = load_model(args.ckpt, args.device)
    print("Model loaded!")

    if args.interactive:
        # Interactive mode
        print("\n=== Interactive Mode ===")
        print("Commands: 'quit' to exit\n")

        while True:
            clip_path = input("CLIP feature path (.npz): ").strip()
            if clip_path.lower() == "quit":
                break

            sf_path = input("SlowFast feature path (.npz, or Enter to skip): ").strip()
            if not sf_path:
                sf_path = None

            query = input("Query: ").strip()
            if query.lower() == "quit":
                break

            text_feat_path = input("Text feature path (.npz, or Enter to skip): ").strip()

            # Load features
            video_feat = load_features(clip_path, sf_path)

            if text_feat_path and os.path.exists(text_feat_path):
                text_feat = load_text_features(text_feat_path)
            else:
                print("Warning: No text feature provided. Using CLIP to encode query.")
                print("(Install CLIP: pip install git+https://github.com/openai/CLIP.git)")
                try:
                    import clip
                    clip_model, _ = clip.load("ViT-B/32", device=args.device)
                    tokens = clip.tokenize([query]).to(args.device)
                    with torch.no_grad():
                        text_feat = clip_model.encode_text(tokens)
                        text_feat = text_feat.cpu().numpy()
                        text_feat = l2_normalize_np_array(text_feat)
                except ImportError:
                    print("CLIP not installed. Please provide pre-computed text features.")
                    continue

            # Predict
            moments, saliency = predict(model, video_feat, text_feat,
                                         args.clip_length, args.device)

            # Print results
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print(f"Top-{args.top_k} Moments:")
            for i, (start, end, score) in enumerate(moments[:args.top_k]):
                print(f"  {i+1}. {start:.1f}s - {end:.1f}s  (score: {score:.4f})")
            if saliency:
                print(f"\nSaliency scores (first 20 clips):")
                print(f"  {saliency[:20]}")
            print(f"{'='*50}\n")

    else:
        # Single query mode
        if not args.clip_feat:
            print("Error: --clip_feat is required")
            return

        video_feat = load_features(args.clip_feat, args.slowfast_feat)

        if args.text_feat and os.path.exists(args.text_feat):
            text_feat = load_text_features(args.text_feat)
        elif args.query:
            try:
                import clip
                clip_model, _ = clip.load("ViT-B/32", device=args.device)
                tokens = clip.tokenize([args.query]).to(args.device)
                with torch.no_grad():
                    text_feat = clip_model.encode_text(tokens)
                    text_feat = text_feat.cpu().numpy()
                    text_feat = l2_normalize_np_array(text_feat)
            except ImportError:
                print("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")
                return
        else:
            print("Error: --query or --text_feat is required")
            return

        moments, saliency = predict(model, video_feat, text_feat,
                                     args.clip_length, args.device)

        print(f"\n{'='*50}")
        print(f"Query: {args.query}")
        print(f"Top-{args.top_k} Moments:")
        for i, (start, end, score) in enumerate(moments[:args.top_k]):
            print(f"  {i+1}. {start:.1f}s - {end:.1f}s  (score: {score:.4f})")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()