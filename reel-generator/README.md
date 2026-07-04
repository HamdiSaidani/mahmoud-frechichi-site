# 🌀 Satisfying Reels Generator

Generate **oddly-satisfying, perfectly-looping vertical videos** for
**YouTube Shorts**, **Instagram Reels**, and **Facebook Reels** — from pure
math. No filming, no stock footage, no API keys, no monthly cost.

The visuals loop *seamlessly*, so the platform keeps replaying them. High
replay = high watch-time = the single biggest thing the algorithm rewards.

> ⚠️ Honest note: no tool can *guarantee* a video goes viral — that depends on
> the algorithm, timing, your hook, and luck. This maximizes your odds by
> nailing the format (9:16, seamless loop, bold hook, fast pacing) and letting
> you pump out a lot of clean content cheaply.

## Setup (one time)

```bash
cd reel-generator
pip install -r requirements.txt
```

`imageio-ffmpeg` bundles its own ffmpeg binary, so you don't need to install
ffmpeg separately.

## Use it

```bash
# quickest start — random style, 1080x1920 MP4 written to ./out/
python satisfying_reels.py

# pick a style, add a viral hook + caption
python satisfying_reels.py --style aurora \
    --hook "Wait for it..." --caption "Follow for daily satisfying loops"

# bring your own music / ASMR track (it loops automatically)
python satisfying_reels.py --style plasma --audio my_track.mp3

# generate 10 reels at once for a week of posting
python satisfying_reels.py --batch 10

# quick low-res preview (renders in a couple seconds)
python satisfying_reels.py --style spiral --fast

# see all styles + palettes
python satisfying_reels.py --list-styles
```

Output MP4s land in `reel-generator/out/`.

## Options

| Flag | Default | What it does |
|------|---------|--------------|
| `--style` | `random` | `plasma`, `spiral`, `waves`, `rings`, `aurora`, `tunnel`, or `random` |
| `--palette` | per-style | `sunset`, `ocean`, `aurora`, `candy`, `lava`, `mono`, `emerald` |
| `--duration` | `7` | Loop length in seconds (5–8 is the sweet spot) |
| `--hook` | — | Big text near the top (the scroll-stopper) |
| `--caption` | — | Smaller text near the bottom (CTA / follow prompt) |
| `--audio` | — | Music/ASMR file to loop under the video |
| `--batch` | `1` | How many videos to generate in one run |
| `--fast` | off | Half-resolution quick preview |
| `--seed` | random | Reproduce an exact video, or seed a batch |
| `--fps` / `--width` / `--height` | `30` / `1080` / `1920` | Output format |

## Tips that actually move the needle

1. **Hook in the first second** — `--hook "Wait for it..."` / `"This is so satisfying"` / `"Don't blink"`.
2. **Add trending audio in-app** after upload, or pass `--audio` for your own ASMR track.
3. **Post consistently** — use `--batch 10` and schedule one a day.
4. **Same caption + CTA** every time builds a recognizable brand.
5. **5–8 second loops** outperform longer ones for replay rate.

## How it works

Each style is a periodic math field evaluated over a phase `p` that runs
`0 → 2π` across the clip. Because every term is periodic in `p`, the last frame
flows back into the first with no visible jump — a true seamless loop. Frames
are streamed straight into ffmpeg and encoded to web-friendly H.264 (`yuv420p`,
`+faststart`).
