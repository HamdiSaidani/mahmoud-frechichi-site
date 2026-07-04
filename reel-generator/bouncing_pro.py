#!/usr/bin/env python3
"""
bouncing_pro.py — the PRO bouncing-ball engine. Outside-the-box upgrades over
the basic circle version:

  PRO SHAPES   bounce inside a circle / triangle / square / pentagon / hexagon
               / star / heart  — and the arena can SPIN (--spin).
  PRO COLORS   gradient backgrounds + neon glow/bloom + themed palettes.
  PRO LOGIC    --mode grow  : ball grows each bounce (classic, builds to a flash)
               --mode split : ball splits in two each bounce -> screen fills
               --mode eat   : ball chews through the wall and ESCAPES
  PRO LOGO     --logo "TEXT" reveals your brand in the center as bounces build.
  REAL MELODY  --melody twinkle|ode|scale|random : bounces play an actual tune.

Still 100% headless: numpy + Pillow -> H.264 + AAC via bundled ffmpeg.

Examples
--------
    python bouncing_pro.py --shape hexagon --spin --mode split --palette neon
    python bouncing_pro.py --shape star --palette gold --logo "MF" --hook "Wait for it..."
    python bouncing_pro.py --shape heart --palette neon --melody twinkle
    python bouncing_pro.py --shape circle --mode eat --hook "Can it escape?"
"""

import argparse
import colorsys
import math
import os
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


# --------------------------------------------------------------------------- #
#  Palettes + backgrounds
# --------------------------------------------------------------------------- #
PALETTES = {
    "rainbow": None,
    "fire":    [(255, 70, 0), (255, 140, 0), (255, 210, 60), (255, 255, 200)],
    "ice":     [(120, 220, 255), (60, 150, 255), (180, 240, 255), (230, 250, 255)],
    "neon":    [(255, 0, 200), (0, 255, 200), (180, 0, 255), (0, 200, 255)],
    "gold":    [(255, 215, 0), (255, 170, 0), (255, 240, 180), (200, 130, 0)],
}
BG_TOP = {"rainbow": (12, 8, 28), "fire": (24, 6, 0), "ice": (4, 10, 24),
          "neon": (10, 4, 24), "gold": (20, 14, 0)}
BG_BOT = {"rainbow": (4, 4, 10), "fire": (6, 2, 0), "ice": (2, 4, 12),
          "neon": (2, 2, 8), "gold": (6, 4, 0)}


def palette_color(palette, k):
    if PALETTES.get(palette) is None:
        r, g, b = colorsys.hsv_to_rgb((k * 0.08) % 1.0, 0.9, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))
    cols = PALETTES[palette]
    return cols[k % len(cols)]


def gradient_bg(w, h, palette):
    top = np.array(BG_TOP.get(palette, (10, 10, 18)), np.float32)
    bot = np.array(BG_BOT.get(palette, (4, 4, 10)), np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    col = (top[None, None, :] * (1 - t) + bot[None, None, :] * t)
    return np.repeat(col, w, axis=1).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Shape vertices (unit radius), then scaled / rotated / translated.
# --------------------------------------------------------------------------- #
def unit_shape(shape):
    if shape == "circle":
        a = np.linspace(0, 2 * math.pi, 160, endpoint=False)
        return np.c_[np.cos(a), np.sin(a)]
    if shape in ("triangle", "square", "pentagon", "hexagon"):
        n = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6}[shape]
        a = np.linspace(-math.pi / 2, 1.5 * math.pi, n, endpoint=False)
        return np.c_[np.cos(a), np.sin(a)]
    if shape == "star":
        pts = []
        for i in range(10):
            rr = 1.0 if i % 2 == 0 else 0.45
            ang = -math.pi / 2 + i * math.pi / 5
            pts.append((math.cos(ang) * rr, math.sin(ang) * rr))
        return np.array(pts)
    if shape == "heart":
        t = np.linspace(0, 2 * math.pi, 80, endpoint=False)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        pts = np.c_[x, -y]
        pts /= np.max(np.abs(pts))
        return pts
    raise ValueError(shape)


def shape_vertices(base, cx, cy, R, angle):
    ca, sa = math.cos(angle), math.sin(angle)
    rot = base @ np.array([[ca, -sa], [sa, ca]]).T
    return rot * R + np.array([cx, cy])


# --------------------------------------------------------------------------- #
#  Physics simulation
# --------------------------------------------------------------------------- #
def closest_on_seg(p, a, b):
    ab = b - a
    denom = float(ab @ ab) or 1e-9
    t = max(0.0, min(1.0, float((p - a) @ ab) / denom))
    return a + t * ab


def simulate(cfg):
    rng = np.random.default_rng(cfg["seed"])
    w, h, fps = cfg["w"], cfg["h"], cfg["fps"]
    cx, cy = w / 2.0, h * 0.46
    R = min(w, h) * 0.42
    dt = 1.0 / fps
    n_frames = int(round(cfg["duration"] * fps))
    base = unit_shape(cfg["shape"])
    n_edges = len(base)
    spin = cfg["spin_rate"] if cfg["spin"] else 0.0

    def new_ball(p, v, r, bn):
        return {"p": np.array(p, float), "v": np.array(v, float),
                "r": r, "bn": bn, "trail": []}

    r0 = R * (0.05 if cfg["mode"] != "split" else 0.045)
    balls = [new_ball([cx, cy - R * 0.1],
                      [rng.uniform(-260, 260), rng.uniform(-120, 80)], r0, 0)]
    broken = set()
    frames, bounce_frames, escaped_at = [], [], None
    cap = cfg["max_balls"]

    for f in range(n_frames):
        angle = spin * (f / fps)
        V = shape_vertices(base, cx, cy, R, angle)
        centroid = np.array([cx, cy])
        snap = []
        spawned = []
        for b in balls:
            b["v"][1] += cfg["gravity"] * dt
            # sub-step so a fast ball can't tunnel through a thin wall
            speed = float(np.hypot(*b["v"]))
            steps = max(2, min(48, int(speed * dt / (b["r"] * 0.4)) + 1))
            sub_dt = dt / steps
            bounced = False
            for _ in range(steps):
                b["p"] = b["p"] + b["v"] * sub_dt
                for i in range(n_edges):
                    if i in broken:
                        continue
                    a, bb = V[i], V[(i + 1) % n_edges]
                    c = closest_on_seg(b["p"], a, bb)
                    d = b["p"] - c
                    dist = float(np.hypot(d[0], d[1]))
                    if dist < b["r"] and dist > 1e-9:
                        n = d / dist
                        vn = float(b["v"] @ n)
                        if vn < 0:                   # moving into the wall
                            b["v"] = (b["v"] - 2 * vn * n) * cfg["speed_boost"]
                            sp = float(np.hypot(*b["v"]))
                            if sp > 2200:
                                b["v"] *= 2200 / sp
                            b["p"] = c + n * b["r"]
                            b["bn"] += 1
                            bounce_frames.append(f)
                            if cfg["mode"] == "grow":
                                b["r"] = min(b["r"] + R * cfg["growth"], R * 0.95)
                                if b["r"] >= R * 0.94 and escaped_at is None:
                                    escaped_at = f
                            elif cfg["mode"] == "split" and len(balls) + len(spawned) < cap:
                                ang2 = math.atan2(b["v"][1], b["v"][0]) + rng.uniform(0.4, 1.0)
                                sp2 = float(np.hypot(*b["v"]))
                                spawned.append(new_ball(
                                    b["p"].tolist(),
                                    [math.cos(ang2) * sp2, math.sin(ang2) * sp2],
                                    b["r"], b["bn"]))
                            elif cfg["mode"] == "eat":
                                for k in range(-2, 3):
                                    broken.add((i + k) % n_edges)
                            bounced = True
                        break
                if bounced:
                    break
            # escape check for eat mode
            if cfg["mode"] == "eat":
                if float(np.hypot(*(b["p"] - centroid))) > R * 1.15 and escaped_at is None:
                    escaped_at = f
            b["trail"].append((float(b["p"][0]), float(b["p"][1]), b["r"]))
            if len(b["trail"]) > 18:
                b["trail"].pop(0)
            snap.append({"p": (float(b["p"][0]), float(b["p"][1])), "r": b["r"],
                         "color": palette_color(cfg["palette"], b["bn"]),
                         "trail": list(b["trail"]), "bn": b["bn"]})
        balls.extend(spawned)
        frames.append({"balls": snap, "V": V.tolist(), "broken": set(broken)})

    meta = {"cx": cx, "cy": cy, "R": R, "n_edges": n_edges,
            "escaped_at": escaped_at,
            "total_bounces": len(bounce_frames),
            "max_balls": len(balls)}
    return frames, sorted(bounce_frames), meta


# --------------------------------------------------------------------------- #
#  Audio — bounces play a real melody (or a rising scale)
# --------------------------------------------------------------------------- #
MELODIES = {
    "scale":   [0, 2, 4, 7, 9],
    "twinkle": [0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0],
    "ode":     [4, 4, 5, 7, 7, 5, 4, 2, 0, 0, 2, 4, 4, 2, 2],
    "random":  None,
}


def build_audio(bounce_frames, melody, duration, fps, wav_path, sr=44100):
    rng = np.random.default_rng(7)
    n = int(duration * sr) + sr
    buf = np.zeros(n, np.float32)
    seq = MELODIES.get(melody)
    base = 523.25
    for i, fr in enumerate(bounce_frames):
        if melody == "random" or seq is None:
            semi = [0, 2, 4, 7, 9][rng.integers(0, 5)] + 12 * rng.integers(0, 2)
        elif melody == "scale":
            step = i
            semi = seq[step % len(seq)] + 12 * ((step // len(seq)) % 3)
        else:
            semi = seq[i % len(seq)]
        freq = base * (2 ** (semi / 12.0))
        start = int(fr / fps * sr)
        m = min(int(0.18 * sr), n - start)
        tt = np.arange(m) / sr
        env = np.exp(-tt * 20.0)
        tone = (np.sin(2 * math.pi * freq * tt)
                + 0.35 * np.sin(2 * math.pi * 2 * freq * tt)) * env * 0.5
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
#  Rendering (with neon glow / bloom)
# --------------------------------------------------------------------------- #
def load_font(size):
    from PIL import ImageFont
    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/Library/Fonts/Arial Bold.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def render(frames, meta, cfg, wav, out):
    from PIL import Image, ImageDraw, ImageChops, ImageFilter
    w, h, fps = cfg["w"], cfg["h"], cfg["fps"]
    bg = gradient_bg(w, h, cfg["palette"])
    font_hook = load_font(int(w * 0.085))
    font_count = load_font(int(w * 0.10))
    font_logo = load_font(int(w * 0.20))
    n = len(frames)
    n_edges = meta["n_edges"]

    ff = find_ffmpeg()
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if wav:
        cmd += ["-i", wav, "-shortest", "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for f, fr in enumerate(frames):
            img = Image.fromarray(bg.copy())
            draw = ImageDraw.Draw(img, "RGBA")
            # center logo reveal (brightens as bounces build)
            if cfg["logo"]:
                prog = min(1.0, max(b["bn"] for b in fr["balls"]) / 30.0) if fr["balls"] else 0
                al = int(40 + prog * 150)
                tw = draw.textlength(cfg["logo"], font=font_logo)
                draw.text((w / 2 - tw / 2, h * 0.39), cfg["logo"], font=font_logo,
                          fill=(255, 255, 255, al))
            # escape flash
            if meta["escaped_at"] is not None and f >= meta["escaped_at"]:
                g = max(0, 50 - (f - meta["escaped_at"])) * 4
                draw.rectangle([0, 0, w, h], fill=(g, g, g, 80))
            # arena walls (skip broken edges so gaps show)
            V = fr["V"]
            broken = fr["broken"]
            for i in range(n_edges):
                if i in broken:
                    continue
                a = V[i]
                b = V[(i + 1) % n_edges]
                draw.line([a[0], a[1], b[0], b[1]], fill=(255, 255, 255), width=9)
            # balls + trails
            for b in fr["balls"]:
                t = b["trail"]
                for i, (tx, ty, tr) in enumerate(t):
                    al = int(150 * (i + 1) / len(t))
                    rr = tr * (0.4 + 0.6 * (i + 1) / len(t))
                    draw.ellipse([tx - rr, ty - rr, tx + rr, ty + rr],
                                 fill=b["color"] + (al,))
                px, py, r = b["p"][0], b["p"][1], b["r"]
                draw.ellipse([px - r, py - r, px + r, py + r], fill=b["color"])
                draw.ellipse([px - r, py - r, px + r, py + r],
                             outline=(255, 255, 255), width=4)
                hr = r * 0.3
                draw.ellipse([px - r * 0.4 - hr, py - r * 0.4 - hr,
                              px - r * 0.4 + hr, py - r * 0.4 + hr],
                             fill=(255, 255, 255, 140))
            # neon glow / bloom
            if cfg["glow"]:
                blur = img.filter(ImageFilter.GaussianBlur(int(w * 0.012)))
                img = ImageChops.screen(img, blur)
                draw = ImageDraw.Draw(img, "RGBA")
            # text overlays (drawn after glow so they stay crisp)
            if cfg["hook"]:
                tw = draw.textlength(cfg["hook"], font=font_hook)
                draw.text((w / 2 - tw / 2, h * 0.06), cfg["hook"], font=font_hook,
                          fill=(255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0))
            if cfg["counter"] and fr["balls"]:
                c = str(max(b["bn"] for b in fr["balls"]))
                if cfg["mode"] == "split":
                    c = str(len(fr["balls"]))
                tw = draw.textlength(c, font=font_count)
                draw.text((w / 2 - tw / 2, h * 0.86), c, font=font_count,
                          fill=(255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0))
            proc.stdin.write(np.asarray(img.convert("RGB")).tobytes())
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
          f"(bounces={meta['total_bounces']}, balls={meta['max_balls']})")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="PRO bouncing-ball reel generator.")
    ap.add_argument("--shape", default="circle",
                    choices=["circle", "triangle", "square", "pentagon",
                             "hexagon", "star", "heart"])
    ap.add_argument("--mode", default="grow", choices=["grow", "split", "eat"])
    ap.add_argument("--palette", default="rainbow", choices=list(PALETTES))
    ap.add_argument("--melody", default="scale", choices=list(MELODIES))
    ap.add_argument("--spin", action="store_true", help="rotate the arena")
    ap.add_argument("--spin-rate", type=float, default=0.6, help="radians/sec")
    ap.add_argument("--logo", default="", help="brand text revealed in the center")
    ap.add_argument("--hook", default="")
    ap.add_argument("--duration", type=float, default=14.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--gravity", type=float, default=1700.0)
    ap.add_argument("--growth", type=float, default=0.02)
    ap.add_argument("--speed-boost", type=float, default=1.03)
    ap.add_argument("--max-balls", type=int, default=120)
    ap.add_argument("--no-counter", action="store_true")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--no-glow", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    W, H = (args.width // 2, args.height // 2) if args.fast else (args.width, args.height)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.batch):
        seed = (args.seed + i) if args.seed is not None else int(rng.integers(1e9))
        cfg = {"w": W, "h": H, "fps": args.fps, "duration": args.duration,
               "shape": args.shape, "mode": args.mode, "palette": args.palette,
               "melody": args.melody, "spin": args.spin, "spin_rate": args.spin_rate,
               "logo": args.logo, "hook": args.hook, "gravity": args.gravity,
               "growth": args.growth, "speed_boost": args.speed_boost,
               "max_balls": args.max_balls, "counter": not args.no_counter,
               "glow": not args.no_glow, "seed": seed}
        out = (args.out if args.out and args.batch == 1
               else os.path.join(outdir, f"pro_{args.shape}_{args.mode}_{seed}.mp4"))
        print(f"[{i + 1}/{args.batch}] shape={args.shape} mode={args.mode} "
              f"palette={args.palette} spin={args.spin} {W}x{H} seed={seed}")
        frames, bf, meta = simulate(cfg)
        wav = None
        if not args.no_sound:
            wav = build_audio(bf, args.melody, args.duration, args.fps,
                              os.path.join(outdir, f"_pro_{seed}.wav"))
        render(frames, meta, cfg, wav, out)
        if wav and os.path.exists(wav):
            os.remove(wav)

    print("\nDone. Upload to YouTube Shorts / Instagram Reels / FB Reels.")


if __name__ == "__main__":
    main()
