"""
Feature Extraction: Video → CLIP + SlowFast features (.npz)

Extracts features compatible with Moment-DETR training format:
  - CLIP ViT-B/32:     [num_clips, 512]   per 2-second clip
  - SlowFast R50 8x8:  [num_clips, 2304]  per 2-second clip

Output format matches the official QVHighlights .npz files.

Requirements:
    pip install pytorchvideo opencv-python Pillow
    pip install git+https://github.com/openai/CLIP.git

Usage:
    # Extract both CLIP and SlowFast features for a single video
    python3 extract_features.py --video video.mp4 --output_dir my_features/

    # Extract for a directory of videos
    python3 extract_features.py --video /path/to/videos/ --output_dir my_features/

    # Extract CLIP only (faster, no SlowFast dependency)
    python3 extract_features.py --video video.mp4 --output_dir my_features/ --clip_only

Output structure:
    my_features/
    ├── clip_features/
    │   ├── video1.npz          # [num_clips, 512]
    │   └── video2.npz
    └── slowfast_features/
        ├── video1.npz          # [num_clips, 2304]
        └── video2.npz
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# CLIP Feature Extraction
# ============================================================

class CLIPExtractor:
    """Extract CLIP ViT-B/32 features per 2-second clip."""

    def __init__(self, device="cuda"):
        import clip
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        print("[CLIP] Model loaded: ViT-B/32")

    @torch.no_grad()
    def extract_text_features(self, query):
        """
        Encode a text query with CLIP.

        Args:
            query: text string

        Returns:
            features: np.ndarray [1, 512], L2 normalized
        """
        import clip
        tokens = clip.tokenize([query], truncate=True).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = feat.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        feat = feat / np.maximum(norms, 1e-8)
        return feat

    @torch.no_grad()
    def extract(self, video_path, clip_length=2):
        """
        Extract CLIP features from a video.

        Args:
            video_path: path to video file
            clip_length: seconds per clip (default 2, matching QVHighlights)

        Returns:
            features: np.ndarray [num_clips, 512], L2 normalized
            num_clips: int
            duration: float
        """
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        num_clips = int(duration / clip_length)

        if num_clips == 0:
            raise ValueError(f"Video too short ({duration:.1f}s)")

        features = []
        for i in range(num_clips):
            center_time = (i + 0.5) * clip_length
            frame_idx = min(int(center_time * fps), total_frames - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                features.append(np.zeros(512, dtype=np.float32))
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            feat = self.model.encode_image(img_tensor)
            feat = feat.cpu().numpy().flatten().astype(np.float32)
            features.append(feat)

        cap.release()

        features = np.stack(features)  # [num_clips, 512]
        # L2 normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features = features / norms

        return features, num_clips, duration


# ============================================================
# SlowFast Feature Extraction
# ============================================================

class SlowFastExtractor:
    """
    Extract SlowFast R50 8x8 features per 2-second clip.

    Uses pytorchvideo's pretrained slowfast_r50 model.
    Extracts features from the penultimate layer (before classification head),
    producing 2304-dim features per clip.
    """

    def __init__(self, device="cuda"):
        self.device = device

        # Load pretrained SlowFast R50
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "slowfast_r50",
            pretrained=True,
        )
        self.model = self.model.to(device)
        self.model.eval()

        # Hook to extract features before the classification head
        # The feature vector is 2304-dim (slow: 2048 + fast: 256)
        self._features = None
        self.model.blocks[5].register_forward_hook(self._hook_fn)

        # SlowFast transform parameters
        self.side_size = 256
        self.crop_size = 256
        self.num_frames = 32
        self.sampling_rate = 2
        self.slowfast_alpha = 4
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]

        print("[SlowFast] Model loaded: slowfast_r50 (pretrained on Kinetics-400)")

    def _hook_fn(self, module, input, output):
        """Hook to capture penultimate layer output."""
        # output is a list of tensors from the two pathways
        # After pool: [B, C, 1, 1, 1] for each pathway
        if isinstance(output, (list, tuple)):
            pooled = []
            for o in output:
                if o.dim() == 5:
                    pooled.append(o.mean(dim=[2, 3, 4]))  # [B, C]
                else:
                    pooled.append(o)
            self._features = torch.cat(pooled, dim=1)  # [B, 2304]
        else:
            if output.dim() == 5:
                self._features = output.mean(dim=[2, 3, 4])
            else:
                self._features = output

    @torch.no_grad()
    def extract(self, video_path, clip_length=2):
        """
        Extract SlowFast features from a video.

        Args:
            video_path: path to video file
            clip_length: seconds per clip

        Returns:
            features: np.ndarray [num_clips, 2304], L2 normalized
            num_clips: int
            duration: float
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        num_clips = int(duration / clip_length)

        if num_clips == 0:
            raise ValueError(f"Video too short ({duration:.1f}s)")

        features = []
        for i in tqdm(range(num_clips), desc="  SlowFast", leave=False):
            start_time = i * clip_length
            end_time = (i + 1) * clip_length

            # Read frames for this clip
            frames = self._read_clip_frames(cap, start_time, end_time, fps, total_frames)

            if frames is None or len(frames) < 8:
                features.append(np.zeros(2304, dtype=np.float32))
                continue

            # Prepare SlowFast input
            try:
                slow_fast_input = self._prepare_input(frames)
                _ = self.model(slow_fast_input)  # trigger hook
                feat = self._features.cpu().numpy().flatten().astype(np.float32)
                features.append(feat)
            except Exception as e:
                features.append(np.zeros(2304, dtype=np.float32))

        cap.release()

        features = np.stack(features)  # [num_clips, 2304]
        # L2 normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features = features / norms

        return features, num_clips, duration

    def _read_clip_frames(self, cap, start_time, end_time, fps, total_frames):
        """Read frames from a specific time window."""
        import cv2

        start_frame = int(start_time * fps)
        end_frame = min(int(end_time * fps), total_frames)
        n_frames = end_frame - start_frame

        if n_frames <= 0:
            return None

        # Sample num_frames * sampling_rate frames uniformly
        target_n = self.num_frames * self.sampling_rate
        indices = np.linspace(start_frame, end_frame - 1, min(target_n, n_frames)).astype(int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.side_size, self.side_size))
                frames.append(frame)

        return frames

    def _prepare_input(self, frames):
        """
        Prepare SlowFast model input from raw frames.

        Returns: list of [slow_pathway, fast_pathway] tensors
        """
        # Stack and normalize: [T, H, W, C] → [C, T, H, W]
        video = np.stack(frames).astype(np.float32) / 255.0
        video = torch.from_numpy(video).permute(3, 0, 1, 2)  # [C, T, H, W]

        # Normalize
        for c in range(3):
            video[c] = (video[c] - self.mean[c]) / self.std[c]

        # Center crop
        h, w = video.shape[2], video.shape[3]
        y = (h - self.crop_size) // 2
        x = (w - self.crop_size) // 2
        video = video[:, :, y:y+self.crop_size, x:x+self.crop_size]

        # Uniform temporal subsample to num_frames
        t = video.shape[1]
        if t > self.num_frames:
            indices = torch.linspace(0, t - 1, self.num_frames).long()
            fast_pathway = torch.index_select(video, 1, indices)
        else:
            # Pad if too few frames
            fast_pathway = video
            if t < self.num_frames:
                pad = self.num_frames - t
                fast_pathway = torch.cat([fast_pathway] + [fast_pathway[:, -1:]] * pad, dim=1)

        # Slow pathway: temporal subsample by alpha
        slow_indices = torch.linspace(
            0, fast_pathway.shape[1] - 1,
            fast_pathway.shape[1] // self.slowfast_alpha
        ).long()
        slow_pathway = torch.index_select(fast_pathway, 1, slow_indices)

        # Add batch dimension: [B, C, T, H, W]
        slow_pathway = slow_pathway.unsqueeze(0).to(self.device)
        fast_pathway = fast_pathway.unsqueeze(0).to(self.device)

        return [slow_pathway, fast_pathway]


# ============================================================
# Main Extraction Pipeline
# ============================================================

def extract_video_features(
    video_path,
    output_dir,
    clip_extractor,
    slowfast_extractor=None,
    clip_length=2,
    overwrite=False,
):
    """
    Extract CLIP (and optionally SlowFast) features for a single video.

    Saves:
        output_dir/clip_features/{video_name}.npz      [num_clips, 512]
        output_dir/slowfast_features/{video_name}.npz   [num_clips, 2304]
    """
    video_name = Path(video_path).stem

    clip_dir = os.path.join(output_dir, "clip_features")
    sf_dir = os.path.join(output_dir, "slowfast_features")
    os.makedirs(clip_dir, exist_ok=True)

    clip_out = os.path.join(clip_dir, f"{video_name}.npz")
    sf_out = os.path.join(sf_dir, f"{video_name}.npz")

    # Skip if already exists
    if not overwrite and os.path.exists(clip_out):
        if slowfast_extractor is None or os.path.exists(sf_out):
            print(f"  Skipping {video_name} (already exists)")
            return

    # CLIP features
    print(f"  Extracting CLIP features...")
    clip_feat, num_clips, duration = clip_extractor.extract(video_path, clip_length)
    np.savez_compressed(clip_out, features=clip_feat)
    print(f"    Saved {clip_out}: shape {clip_feat.shape}")

    # SlowFast features
    if slowfast_extractor is not None:
        os.makedirs(sf_dir, exist_ok=True)
        print(f"  Extracting SlowFast features...")
        sf_feat, _, _ = slowfast_extractor.extract(video_path, clip_length)

        # Align clip counts (should match, but just in case)
        min_len = min(len(clip_feat), len(sf_feat))
        if len(sf_feat) != len(clip_feat):
            print(f"    Warning: clip count mismatch (CLIP={len(clip_feat)}, SF={len(sf_feat)}), truncating to {min_len}")
            sf_feat = sf_feat[:min_len]

        np.savez_compressed(sf_out, features=sf_feat)
        print(f"    Saved {sf_out}: shape {sf_feat.shape}")

    print(f"  Done: {num_clips} clips, {duration:.1f}s duration\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP + SlowFast features from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video, CLIP + SlowFast
  python3 extract_features.py --video video.mp4 --output_dir my_features/

  # Directory of videos
  python3 extract_features.py --video /path/to/videos/ --output_dir my_features/

  # CLIP only (faster)
  python3 extract_features.py --video video.mp4 --output_dir my_features/ --clip_only

Output:
  my_features/
  ├── clip_features/VIDEO_NAME.npz       [num_clips, 512]
  └── slowfast_features/VIDEO_NAME.npz   [num_clips, 2304]
""",
    )
    parser.add_argument("--video", required=True, help="Video file or directory of videos")
    parser.add_argument("--output_dir", default="extracted_features", help="Output directory")
    parser.add_argument("--clip_length", type=float, default=2.0, help="Clip length in seconds")
    parser.add_argument("--clip_only", action="store_true", help="Extract CLIP only (skip SlowFast)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing features")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Initialize extractors
    clip_extractor = CLIPExtractor(device=device)

    slowfast_extractor = None
    if not args.clip_only:
        try:
            slowfast_extractor = SlowFastExtractor(device=device)
        except Exception as e:
            print(f"Warning: Failed to load SlowFast ({e})")
            print("Falling back to CLIP-only extraction.")
            print("To fix: pip install pytorchvideo\n")

    # Collect video files
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}
    if os.path.isdir(args.video):
        video_files = sorted([
            os.path.join(args.video, f)
            for f in os.listdir(args.video)
            if os.path.splitext(f)[1].lower() in video_exts
        ])
    else:
        video_files = [args.video]

    print(f"Found {len(video_files)} video(s) to process\n")

    # Extract
    for i, vpath in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] {os.path.basename(vpath)}")
        try:
            extract_video_features(
                vpath, args.output_dir,
                clip_extractor, slowfast_extractor,
                clip_length=args.clip_length,
                overwrite=args.overwrite,
            )
        except Exception as e:
            print(f"  ERROR: {e}\n")

    print("=" * 50)
    print("Feature extraction complete!")
    print(f"Output: {args.output_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()