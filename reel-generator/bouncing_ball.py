#!/usr/bin/env python3
"""
bouncing_ball.py — generate the viral "ball bouncing inside a circle" reel.

A ball bounces inside a ring under gravity, GROWS a little on every bounce,
leaves a glowing rainbow trail, and plays a musical ASMR *ping* on each hit
(the pings climb a pentatonic scale, so it sounds satisfying, not random).
When the ball gets big enough to fill the ring it "escapes" with a flash.

This is the #1 viral code-simulation format on Shorts / Reels / TikTok.
100% headless: numpy + Pillow -> H.264, audio muxed via ffmpeg. No display,
no pygame, no 3D. Runs anywhere Python runs.

Examples
--------
    python bouncing_ball.py
    python bouncing_ball.py --hook "Wait for it..." --balls 1 --duration 12
    python bouncing_ball.py --palette fire --gravity 1800 --growth 0.02
    python bouncing_ball.py --batch 5

Requirements:  pip install -r requirements.txt   (numpy, pillow, imageio-ffmpeg)
"""

import argparse
import colorsys
import math
import os
import struct
import subprocess
import sys
import wave

import numpy as np


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
        sys.exit("ffmpeg not found. Run:  pip install imageio-ffmpeg")


PALETTES = {
    "rainbow": None,                       # cycles full hue wheel by bounce
    "fire":    [(255, 70, 0), (255, 140, 0), (255, 210, 60), (255, 255, 200)],
    "ice":     [(120, 220, 255), (60, 150, 255), (180, 240, 255), (230, 250, 255)],
    "neon":    [(255, 0, 200), (0, 255, 200), (180, 0, 255), (0, 200, 255)],
    "gold":    [(255, 215, 0), (255, 170, 0), (255, 240, 180), (200, 130, 0)],
}


def bounce_color(palette: str, k: int):
    """Color for the k-th bounce."""
    if palette == "rainbow" or PALETTES.get(palette) is None:
        r, g, b = colorsys.hsv_to_rgb((k * 0.08) % 1.0, 0.9, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))
    cols = PALETTES[palette]
    return cols[k % len(cols)]


# --------------------------------------------------------------------------- #
#  Physics: simulate the whole trajectory first, record per-frame state.
# --------------------------------------------------------------------------- #
def simulate(w, h, fps, duration, gravity, growth, speed_boost, n_balls,
             restitution, seed):
    rng = np.random.default_rng(seed)
    cx, cy = w / 2.0, h * 0.46
    ring_r = min(w, h) * 0.42
    dt = 1.0 / fps
    n_frames = int(round(duration * fps))

    balls = []
    for i in range(n_balls):
        ang = rng.uniform(0, 2 * math.pi)
        balls.append({
            "p": np.array([cx + math.cos(ang) * ring_r * 0.2,
                           cy + math.sin(ang) * ring_r * 0.2]),
            "v": np.array([rng.uniform(-300, 300), rng.uniform(-200, 100)]),
            "r": ring_r * 0.05,
            "bounces": 0,
            "trail": [],
        })

    frames = []          # list of (list of ball-render-states)
    bounce_events = []    # (frame_index, ball_bounces_count)
    escaped_at = None

    for f in range(n_frames):
        snapshot = []
        for b in balls:
            b["v"][1] += gravity * dt
            b["p"] = b["p"] + b["v"] * dt
            d = b["p"] - np.array([cx, cy])
            dist = float(np.hypot(d[0], d[1]))
            if dist + b["r"] >= ring_r and dist > 1e-6:
                n = -d / dist                       # inward normal
                b["p"] = np.array([cx, cy]) + d * ((ring_r - b["r"]) / dist)
                vn = float(b["v"] @ (-n))            # speed along outward normal
                b["v"] = (b["v"] - 2 * vn * (-n)) * restitution * speed_boost
                # clamp insane speeds
                sp = float(np.hypot(*b["v"]))
                if sp > 2600:
                    b["v"] *= 2600 / sp
                b["bounces"] += 1
                b["r"] = min(b["r"] + ring_r * growth, ring_r * 0.96)
                bounce_events.append((f, b["bounces"]))
                if b["r"] >= ring_r * 0.95 and escaped_at is None:
                    escaped_at = f
            b["trail"].append((float(b["p"][0]), float(b["p"][1]), b["r"]))
            if len(b["trail"]) > 22:
                b["trail"].pop(0)
            snapshot.append({
                "p": (float(b["p"][0]), float(b["p"][1])),
                "r": b["r"],
                "color": bounce_color_state(b),
                "trail": list(b["trail"]),
                "bounces": b["bounces"],
            })
        frames.append(snapshot)

    meta = {"cx": cx, "cy": cy, "ring_r": ring_r,
            "total_bounces": sum(b["bounces"] for b in balls),
            "escaped_at": escaped_at}
    return frames, bounce_events, meta


def bounce_color_state(b):
    # set once per ball per frame; uses palette closure set globally
    return _PALETTE_FN(b["bounces"])


_PALETTE_FN = lambda k: bounce_color("rainbow", k)


# --------------------------------------------------------------------------- #
#  Audio: a decaying sine "ping" per bounce, climbing a pentatonic scale.
# --------------------------------------------------------------------------- #
def build_audio(bounce_events, duration, fps, sr=44100, wav_path="bounce.wav"):
    n = int(duration * sr) + sr
    buf = np.zeros(n, dtype=np.float32)
    semis = [0, 2, 4, 7, 9]                  # major pentatonic
    base = 523.25                            # C5
    for (frame_idx, count) in bounce_events:
        t0 = frame_idx / fps
        start = int(t0 * sr)
        step = count - 1
        octave = step // len(semis)
        semi = semis[step % len(semis)] + 12 * (octave % 3)
        freq = base * (2 ** (semi / 12.0))
        dur = 0.18
        m = int(dur * sr)
        if start + m > n:
            m = n - start
        tt = np.arange(m) / sr
        env = np.exp(-tt * 22.0)
        tone = (np.sin(2 * math.pi * freq * tt)
                + 0.4 * np.sin(2 * math.pi * 2 * freq * tt)) * env * 0.5
        buf[start:start + m] += tone.astype(np.float32)
    peak = float(np.max(np.abs(buf))) or 1.0
    buf = (buf / peak * 0.85 * 32767).astype(np.int16)
    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(buf.tobytes())
    return wav_path


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def load_font(size):
    from PIL import ImageFont
    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/Library/Fonts/Arial Bold.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def render(frames, meta, w, h, fps, hook, audio_path, out, show_counter):
    from PIL import Image, ImageDraw
    ff = find_ffmpeg()
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio_path:
        cmd += ["-i", audio_path, "-shortest", "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cx, cy, ring_r = meta["cx"], meta["cy"], meta["ring_r"]
    font_hook = load_font(int(w * 0.085))
    font_count = load_font(int(w * 0.10))
    n = len(frames)
    try:
        for f, snapshot in enumerate(frames):
            img = Image.new("RGB", (w, h), (8, 8, 16))
            draw = ImageDraw.Draw(img, "RGBA")
            # escape flash
            if meta["escaped_at"] is not None and f >= meta["escaped_at"]:
                g = max(0, 60 - (f - meta["escaped_at"])) * 3
                draw.rectangle([0, 0, w, h], fill=(g, g, g, 90))
            # the ring
            draw.ellipse([cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
                         outline=(255, 255, 255), width=10)
            draw.ellipse([cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
                         outline=(120, 120, 160, 120), width=22)
            for b in snapshot:
                # trail
                t = b["trail"]
                for i, (tx, ty, tr) in enumerate(t):
                    a = int(160 * (i + 1) / len(t))
                    rr = tr * (0.4 + 0.6 * (i + 1) / len(t))
                    col = b["color"] + (a,)
                    draw.ellipse([tx - rr, ty - rr, tx + rr, ty + rr], fill=col)
                # ball
                px, py, r = b["p"][0], b["p"][1], b["r"]
                draw.ellipse([px - r, py - r, px + r, py + r], fill=b["color"])
                draw.ellipse([px - r, py - r, px + r, py + r],
                             outline=(255, 255, 255), width=4)
                # glossy highlight
                hr = r * 0.32
                draw.ellipse([px - r * 0.4 - hr, py - r * 0.4 - hr,
                              px - r * 0.4 + hr, py - r * 0.4 + hr],
                             fill=(255, 255, 255, 150))
            if hook:
                tw = draw.textlength(hook, font=font_hook)
                draw.text((w / 2 - tw / 2, h * 0.06), hook, font=font_hook,
                          fill=(255, 255, 255), stroke_width=6,
                          stroke_fill=(0, 0, 0))
            if show_counter:
                c = str(max(b["bounces"] for b in snapshot))
                tw = draw.textlength(c, font=font_count)
                draw.text((w / 2 - tw / 2, h * 0.86), c, font=font_count,
                          fill=(255, 255, 255), stroke_width=6,
                          stroke_fill=(0, 0, 0))
            proc.stdin.write(np.asarray(img).tobytes())
            if f % 20 == 0 or f == n - 1:
                print(f"\r  rendering {(f + 1) / n * 100:5.1f}%", end="", flush=True)
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        if proc.wait() != 0:
            print("\nffmpeg error:\n" + err[-1500:])
            sys.exit(1)
    finally:
        if proc.poll() is None:
            proc.kill()
    print(f"\r  rendered  100.0%  ->  {out}  "
          f"({meta['total_bounces']} bounces)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Generate the viral bouncing-ball-in-a-circle reel.")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--balls", type=int, default=1)
    ap.add_argument("--gravity", type=float, default=1600.0)
    ap.add_argument("--growth", type=float, default=0.012,
                    help="ring-fraction the ball grows per bounce")
    ap.add_argument("--speed-boost", type=float, default=1.03,
                    help="velocity multiplier per bounce (escalation)")
    ap.add_argument("--restitution", type=float, default=1.0)
    ap.add_argument("--palette", choices=list(PALETTES), default="rainbow")
    ap.add_argument("--hook", default="")
    ap.add_argument("--no-counter", action="store_true")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    global _PALETTE_FN
    _PALETTE_FN = lambda k: bounce_color(args.palette, k)

    w, h = (args.width // 2, args.height // 2) if args.fast else (args.width, args.height)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.batch):
        seed = (args.seed + i) if args.seed is not None else int(rng.integers(1e9))
        out = (args.out if args.out and args.batch == 1
               else os.path.join(outdir, f"bounce_{args.palette}_{seed}.mp4"))
        print(f"[{i + 1}/{args.batch}] balls={args.balls} {w}x{h} "
              f"{args.duration}s palette={args.palette} seed={seed}")
        frames, events, meta = simulate(
            w, h, args.fps, args.duration, args.gravity, args.growth,
            args.speed_boost, args.balls, args.restitution, seed)
        wav = None
        if not args.no_sound:
            wav = build_audio(events, args.duration, args.fps,
                              wav_path=os.path.join(outdir, f"_bounce_{seed}.wav"))
        render(frames, meta, w, h, args.fps, args.hook, wav, out,
               not args.no_counter)
        if wav and os.path.exists(wav):
            os.remove(wav)

    print("\nDone. Upload to YouTube Shorts / Instagram Reels / FB Reels.")


if __name__ == "__main__":
    main()
