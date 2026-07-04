#!/usr/bin/env python3
"""
satisfying_reels.py — generate oddly-satisfying, perfectly-looping vertical
short videos (YouTube Shorts / Instagram Reels / Facebook Reels).

Everything is procedural math -> pixels -> H.264 MP4. No filming, no stock
footage, no API keys, no cost. The visuals loop seamlessly so the platform
keeps replaying them, which is the single biggest driver of watch-time.

Usage examples
--------------
    # quickest start (random style, ready-to-upload 9:16 MP4)
    python satisfying_reels.py

    # pick a style + add a viral hook + caption
    python satisfying_reels.py --style aurora \
        --hook "Wait for it..." --caption "Follow for daily satisfying loops"

    # add your own music / ASMR track
    python satisfying_reels.py --style plasma --audio my_track.mp3

    # batch out 10 random reels for a week of posting
    python satisfying_reels.py --batch 10

    # list the available looping styles
    python satisfying_reels.py --list-styles

Requirements:  pip install -r requirements.txt   (numpy, pillow, imageio-ffmpeg)
"""

import argparse
import os
import subprocess
import sys

import numpy as np

# --------------------------------------------------------------------------- #
#  ffmpeg discovery (prefer the pip-bundled binary, fall back to system)
# --------------------------------------------------------------------------- #
def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        from shutil import which
        exe = which("ffmpeg")
        if exe:
            return exe
        sys.exit(
            "ffmpeg not found. Install it with:  pip install imageio-ffmpeg\n"
            "(or install ffmpeg from https://ffmpeg.org and put it on your PATH)"
        )


# --------------------------------------------------------------------------- #
#  Color palettes — smooth, high-contrast, "satisfying" gradients.
#  Each is a list of RGB control points the renderer interpolates cyclically.
# --------------------------------------------------------------------------- #
PALETTES = {
    "sunset":  [(255, 94, 77), (255, 154, 0), (255, 206, 84), (199, 0, 144), (88, 0, 140)],
    "ocean":   [(0, 119, 182), (0, 180, 216), (144, 224, 239), (2, 62, 138), (3, 4, 94)],
    "aurora":  [(0, 255, 170), (0, 200, 255), (120, 80, 255), (255, 0, 200), (0, 255, 170)],
    "candy":   [(255, 0, 153), (255, 110, 199), (160, 120, 255), (110, 200, 255), (255, 0, 153)],
    "lava":    [(20, 0, 0), (130, 0, 0), (255, 80, 0), (255, 200, 0), (255, 255, 200)],
    "mono":    [(10, 10, 20), (60, 60, 90), (140, 150, 200), (220, 230, 255), (10, 10, 20)],
    "emerald": [(2, 48, 32), (0, 128, 96), (0, 200, 140), (160, 255, 200), (2, 48, 32)],
}
DEFAULT_PALETTE = {
    "plasma": "candy", "spiral": "aurora", "waves": "ocean",
    "rings": "sunset", "aurora": "aurora", "tunnel": "lava",
}


def build_lut(palette_name: str, size: int = 1024) -> np.ndarray:
    """Return an (size, 3) uint8 lookup table that wraps around seamlessly."""
    cols = np.array(PALETTES[palette_name], dtype=np.float32)
    # Make it cyclic by appending the first color at the end.
    cols = np.vstack([cols, cols[0]])
    n = len(cols) - 1
    idx = np.linspace(0, n, size, endpoint=False)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n)
    frac = (idx - lo)[:, None]
    lut = cols[lo] * (1 - frac) + cols[hi] * frac
    return np.clip(lut, 0, 255).astype(np.uint8)


def colorize(field: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map a float field (any range) through a cyclic LUT -> RGB uint8 frame."""
    n = lut.shape[0]
    idx = (np.mod(field, 1.0) * n).astype(np.int32) % n
    return lut[idx]


# --------------------------------------------------------------------------- #
#  Coordinate grid (built once, reused for every frame)
# --------------------------------------------------------------------------- #
def make_grid(w: int, h: int):
    # normalized, aspect-correct coordinates centered at 0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xs - w / 2) / (h / 2)        # divide both by h -> keep circles round
    y = (ys - h / 2) / (h / 2)
    r = np.sqrt(x * x + y * y)
    a = np.arctan2(y, x)
    return x, y, r, a


# --------------------------------------------------------------------------- #
#  Styles — each returns a scalar field for phase p in [0, 2*pi).
#  Because every term is periodic in p, frame 0 == frame N -> a perfect loop.
# --------------------------------------------------------------------------- #
def style_plasma(g, p):
    x, y, r, a = g
    f = (np.sin(x * 6 + p)
         + np.sin(y * 6 - p)
         + np.sin((x + y) * 5 + p)
         + np.sin(r * 9 - p * 2))
    return f * 0.18


def style_spiral(g, p):
    x, y, r, a = g
    return (a * 3 + r * 10 - p * 2) / (2 * np.pi)


def style_waves(g, p):
    x, y, r, a = g
    f = np.sin(y * 7 + np.sin(x * 3 + p) * 2 + p) + np.sin(x * 4 - p)
    return f * 0.25


def style_rings(g, p):
    x, y, r, a = g
    return np.sin(r * 14 - p * 2) * 0.5


def style_aurora(g, p):
    x, y, r, a = g
    f = (np.sin(x * 3 + np.sin(y * 2 + p) + p)
         + np.sin(y * 4 + np.sin(x * 2 - p) - p)
         + np.sin(r * 6 + p))
    return f * 0.16


def style_tunnel(g, p):
    x, y, r, a = g
    return (np.sin(a * 6 + p) * 0.15 + 1.0 / (r + 0.25) - p / np.pi)


STYLES = {
    "plasma": style_plasma, "spiral": style_spiral, "waves": style_waves,
    "rings": style_rings, "aurora": style_aurora, "tunnel": style_tunnel,
}


# --------------------------------------------------------------------------- #
#  Text overlay (hook on top, caption near bottom) with safe margins + stroke
# --------------------------------------------------------------------------- #
def load_font(size: int):
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def draw_text_block(draw, text, font, center_x, y, max_w, fill=(255, 255, 255)):
    """Word-wrap + center a text block, with a heavy stroke for readability."""
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    bbox = font.getbbox("Ay")
    line_h = (bbox[3] - bbox[1]) + 18
    for i, line in enumerate(lines):
        lw = draw.textlength(line, font=font)
        draw.text((center_x - lw / 2, y + i * line_h), line, font=font,
                  fill=fill, stroke_width=max(3, font.size // 12),
                  stroke_fill=(0, 0, 0))
    return len(lines) * line_h


def overlay_text(frame: np.ndarray, hook: str, caption: str) -> np.ndarray:
    if not hook and not caption:
        return frame
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    if hook:
        draw_text_block(draw, hook, load_font(int(w * 0.11)),
                        w / 2, int(h * 0.10), int(w * 0.86))
    if caption:
        draw_text_block(draw, caption, load_font(int(w * 0.055)),
                        w / 2, int(h * 0.80), int(w * 0.86),
                        fill=(255, 240, 180))
    return np.asarray(img)


# --------------------------------------------------------------------------- #
#  Render one video
# --------------------------------------------------------------------------- #
def render(style, palette, duration, fps, w, h, hook, caption, audio, out, seed):
    rng = np.random.default_rng(seed)
    g = make_grid(w, h)
    lut = build_lut(palette)
    # a small per-video color drift so batches don't look identical
    hue_shift = float(rng.uniform(0, 1))
    n_frames = int(round(duration * fps))
    style_fn = STYLES[style]

    ff = find_ffmpeg()
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio:
        if not os.path.exists(audio):
            sys.exit(f"audio file not found: {audio}")
        cmd += ["-stream_loop", "-1", "-i", audio, "-shortest",
                "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for i in range(n_frames):
            p = 2 * np.pi * i / n_frames          # phase -> seamless loop
            field = style_fn(g, p) + hue_shift
            frame = colorize(field, lut)
            frame = overlay_text(frame, hook, caption)
            proc.stdin.write(frame.astype(np.uint8).tobytes())
            if i % 15 == 0 or i == n_frames - 1:
                pct = (i + 1) / n_frames * 100
                print(f"\r  rendering {style:<7} {pct:5.1f}%", end="", flush=True)
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        if proc.wait() != 0:
            print("\nffmpeg error:\n" + err[-1500:])
            sys.exit(1)
    finally:
        if proc.poll() is None:
            proc.kill()
    print(f"\r  rendered  {style:<7} 100.0%  ->  {out}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Generate oddly-satisfying looping vertical short videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=list(STYLES) + ["random"], default="random",
                    help="visual style (default: random)")
    ap.add_argument("--palette", choices=list(PALETTES), default=None,
                    help="color palette (default: a good match for the style)")
    ap.add_argument("--duration", type=float, default=7.0,
                    help="seconds per loop (5-8 is the sweet spot, default 7)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--hook", default="", help="big hook text near the top")
    ap.add_argument("--caption", default="", help="smaller caption near the bottom")
    ap.add_argument("--audio", default=None, help="optional music/ASMR file to loop in")
    ap.add_argument("--out", default=None, help="output .mp4 path")
    ap.add_argument("--batch", type=int, default=1, help="how many videos to generate")
    ap.add_argument("--fast", action="store_true",
                    help="render at half resolution for a quick preview")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--list-styles", action="store_true")
    args = ap.parse_args()

    if args.list_styles:
        print("Styles : " + ", ".join(STYLES))
        print("Palettes: " + ", ".join(PALETTES))
        return

    w, h = args.width, args.height
    if args.fast:
        w, h = w // 2, h // 2

    rng = np.random.default_rng(args.seed)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(outdir, exist_ok=True)

    for n in range(args.batch):
        style = (rng.choice(list(STYLES)) if args.style == "random" else args.style)
        palette = args.palette or DEFAULT_PALETTE.get(style, "aurora")
        seed = (args.seed + n) if args.seed is not None else int(rng.integers(1e9))
        if args.out and args.batch == 1:
            out = args.out
        else:
            out = os.path.join(outdir, f"reel_{style}_{seed}.mp4")
        print(f"[{n + 1}/{args.batch}] style={style} palette={palette} "
              f"{w}x{h} {args.duration}s @ {args.fps}fps")
        render(style, palette, args.duration, args.fps, w, h,
               args.hook, args.caption, args.audio, out, seed)

    print("\nDone. Upload straight to YouTube Shorts / Instagram Reels / FB Reels.")


if __name__ == "__main__":
    main()
