"""Shared helpers: frame sampling from video and a frozen CLIP encoder.

Every dataset is reduced to the same unit, a short *clip* (a time window in one video),
represented by N evenly spaced RGB frames. Using one definition everywhere keeps the
staged-vs-real comparison fair.
"""
import io
from dataclasses import dataclass

import numpy as np

CLIP_SECONDS = 2.0      # length of one clip
FRAMES_PER_CLIP = 8     # frames sampled per clip

# Zero-shot prompts. Fixed before looking at any labels; do not tune them on the test set.
FALL_PROMPTS = [
    "a photo of a person falling down",
    "a person falling to the ground",
    "a person tripping and falling over",
    "a person slipping and falling",
]
NONFALL_PROMPTS = [
    "a photo of a person standing",
    "a photo of a person walking",
    "a photo of a person sitting on a chair",
    "a photo of a person bending over",
    "a photo of a person lying in bed",
    "a photo of a person exercising",
    "a photo of a room",
]


def read_frames(src, start_s, end_s, n=FRAMES_PER_CLIP, crop=None):
    """Return up to n PIL frames evenly spaced in [start_s, end_s].

    src: path or raw bytes of a video file. crop: optional (left, top, right, bottom) box.
    Decodes sequentially (clips here are short), picking the frame nearest each target time.
    """
    import av
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    targets = np.linspace(start_s, end_s, n)
    out, k = [], 0
    with av.open(src) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        prev = None
        for fr in c.decode(s):
            if fr.pts is None:
                continue
            t = float(fr.pts * s.time_base)
            while k < n and t >= targets[k]:
                pick = fr if prev is None or abs(t - targets[k]) <= abs(prev[0] - targets[k]) else prev[1]
                out.append(pick.to_image())
                k += 1
            if k >= n:
                break
            prev = (t, fr)
        while k < n and prev is not None:          # window runs past the end of the video
            out.append(prev[1].to_image()); k += 1
    if crop is not None:
        out = [im.crop(crop) for im in out]
    return out


def write_clip(src, start_s, end_s, path, width=320):
    """Re-encode [start_s, end_s] of src as a small browser-playable H.264 mp4."""
    import av
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    with av.open(src) as c, av.open(path, "w") as o:
        s = c.streams.video[0]
        from fractions import Fraction
        rate = Fraction(s.average_rate or 30).limit_denominator(1000)
        h = None
        ov = None
        i = 0
        for fr in c.decode(s):
            if fr.pts is None:
                continue
            t = float(fr.pts * s.time_base)
            if t < start_s:
                continue
            if t > end_s:
                break
            if ov is None:
                h = int(round(fr.height * width / fr.width / 2) * 2)
                ov = o.add_stream("libx264", rate=rate)
                ov.width, ov.height, ov.pix_fmt = width, h, "yuv420p"
                ov.options = {"crf": "28", "preset": "veryfast", "movflags": "+faststart"}
            img = fr.reformat(width=width, height=h, format="yuv420p")
            img.pts = i                       # frame index in units of 1/rate
            img.time_base = 1 / rate
            i += 1
            for p in ov.encode(img):
                o.mux(p)
        if ov is not None:
            for p in ov.encode():
                o.mux(p)


@dataclass
class ClipEncoder:
    """Frozen open_clip image/text encoder. Clip embedding = mean of L2-normalised frame embeddings."""
    model_name: str = "ViT-L-14"
    pretrained: str = "datacomp_xl_s13b_b90k"
    device: str = "cuda"
    batch_size: int = 64

    def __post_init__(self):
        import open_clip
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU (slow)")
            self.device = "cpu"
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.fall_text = self.encode_text(FALL_PROMPTS)
        self.nonfall_text = self.encode_text(NONFALL_PROMPTS)

    def encode_text(self, prompts):
        torch = self.torch
        with torch.no_grad():
            t = self.model.encode_text(self.tokenizer(prompts).to(self.device)).float()
        return torch.nn.functional.normalize(t, dim=-1)

    def encode_images(self, images):
        """images: list of PIL -> (len, d) float32 numpy, L2-normalised."""
        torch = self.torch
        outs = []
        for i in range(0, len(images), self.batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i:i + self.batch_size]]).to(self.device)
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        e = self.model.encode_image(x)
                else:
                    e = self.model.encode_image(x)
            outs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        return torch.cat(outs).numpy()

    def zero_shot(self, frame_emb):
        """frame_emb: (n, d) -> (prob, margin).
        prob:   mean over frames of the softmax mass on fall prompts (saturates easily);
        margin: mean over frames of logsumexp(fall logits) - logsumexp(non-fall logits).
        Use margin for ranking; it does not saturate."""
        torch = self.torch
        e = torch.from_numpy(frame_emb).to(self.device)
        logits = 100.0 * e @ torch.cat([self.fall_text, self.nonfall_text]).T
        k = len(FALL_PROMPTS)
        p = logits.softmax(-1)[:, :k].sum(-1)
        margin = torch.logsumexp(logits[:, :k], -1) - torch.logsumexp(logits[:, k:], -1)
        return float(p.mean()), float(margin.mean())

    def clip_embedding(self, frame_emb):
        m = frame_emb.mean(0)
        return m / (np.linalg.norm(m) + 1e-8)
