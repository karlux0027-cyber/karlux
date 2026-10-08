#!/usr/bin/env python3
"""voice2video: turn a voice-to-text transcript into a finished MP4.

Pipeline:
  1. Read the transcript (a text file, stdin, or transcribe an audio file).
  2. Ask Claude to clean up the dictation and plan a short video: a title,
     a color mood, and a list of scenes (headline, key points, narration).
  3. Pick the soundtrack: your own recording, a robot voice (espeak-ng),
     or silence.
  4. Draw every frame with Pillow and encode the video with ffmpeg.

Run `python voice2video.py --help` for options.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MODEL = "claude-opus-5-5"

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------

# Each mood: (gradient top, gradient bottom, accent, text color)
MOODS = {
    "calm": ("#0f2027", "#2c5364", "#7fdbda", "#f4f7f8"),
    "bold": ("#1a0033", "#6a0dad", "#ffcc00", "#ffffff"),
    "warm": ("#3e1f0d", "#c0582f", "#ffd59e", "#fff8ef"),
    "tech": ("#050a18", "#123a6b", "#36d1ff", "#eaf6ff"),
    "fresh": ("#0b3d2e", "#1f8a5b", "#d4ff6b", "#f2fff6"),
    "dark": ("#0b0b0f", "#2a2a35", "#ff5c7a", "#f5f5f7"),
}

SIZES = {
    "vertical": (1080, 1920),   # TikTok / Reels / Shorts
    "landscape": (1920, 1080),  # YouTube
    "square": (1080, 1080),     # Instagram feed
}

FONT_CANDIDATES = {
    "bold": [
        "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf",
        "/usr/share/fonts/opentype/inter/Inter-Bold.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
    "regular": [
        "/usr/share/fonts/opentype/inter/Inter-Medium.otf",
        "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
}


def load_font(kind: str, size: int, override: str | None = None) -> ImageFont.FreeTypeFont:
    paths = ([override] if override else []) + FONT_CANDIDATES[kind]
    for p in paths:
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    headline: str
    points: list[str]
    narration: str
    duration: float = 0.0
    audio: Path | None = None


@dataclass
class Plan:
    title: str
    mood: str
    scenes: list[Scene] = field(default_factory=list)


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "mood": {"type": "string", "enum": list(MOODS)},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "points": {"type": "array", "items": {"type": "string"}},
                    "narration": {"type": "string"},
                },
                "required": ["headline", "points", "narration"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "mood", "scenes"],
    "additionalProperties": False,
}

PLANNER_PROMPT = """You turn raw voice-to-text dictation into a plan for a short, \
text-on-screen video.

The dictation may ramble, repeat itself, contain filler words ("um", "like", \
"you know") and transcription mistakes. Fix those, but keep the speaker's \
meaning, voice and facts. Do not invent claims they did not make.

Return:
- title: a short, catchy video title (max 8 words).
- mood: the color mood that fits the content best.
- scenes: {scene_hint} scenes, in the order the ideas were spoken. For each:
  - headline: max 7 words, punchy, shown large on screen.
  - points: 0 to 3 short supporting points, max 8 words each.
  - narration: the cleaned-up words for this part of the dictation, written \
to be read aloud. Together, the narrations should cover the whole dictation \
in order, without adding new content.

Dictation:
<dictation>
{transcript}
</dictation>"""


def plan_with_claude(transcript: str, scenes: int | None) -> Plan:
    try:
        import anthropic
    except ImportError:
        sys.exit("The 'anthropic' package is missing. Run: pip install -r requirements.txt")

    scene_hint = str(scenes) if scenes else "3 to 8 (pick what fits the length)"
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": PLAN_SCHEMA},
        },
        # If the request is declined by a safety check, the API retries it
        # on a suitable fallback model instead of failing.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[
            {
                "role": "user",
                "content": PLANNER_PROMPT.format(transcript=transcript, scene_hint=scene_hint),
            }
        ],
    )
    if response.stop_reason == "refusal":
        sys.exit("Claude declined to plan this video.")
    if response.stop_reason == "max_tokens":
        sys.exit("The plan was cut off. Try a shorter transcript or fewer scenes.")

    text = next(b.text for b in response.content if b.type == "text")
    return plan_from_dict(json.loads(text))


def plan_from_dict(data: dict) -> Plan:
    mood = data.get("mood") if data.get("mood") in MOODS else "tech"
    scenes = [
        Scene(
            headline=s["headline"].strip(),
            points=[p.strip() for p in s.get("points", []) if p.strip()][:3],
            narration=s["narration"].strip(),
        )
        for s in data["scenes"]
        if s.get("headline") or s.get("narration")
    ]
    if not scenes:
        sys.exit("The plan came back with no scenes.")
    return Plan(title=data.get("title", "").strip(), mood=mood, scenes=scenes)


def plan_offline(transcript: str, scenes: int | None) -> Plan:
    """No-AI fallback: split the text into sentence groups."""
    transcript = re.sub(r"\b(um+|uh+|erm|you know|like,)\s*", "", transcript, flags=re.I)
    transcript = re.sub(r"\s+([,.!?])", r"\1", transcript)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]
    if not sentences:
        sys.exit("The transcript is empty.")
    n = scenes or max(1, min(8, math.ceil(len(sentences) / 2)))
    n = min(n, len(sentences))
    per = math.ceil(len(sentences) / n)
    out = []
    for i in range(0, len(sentences), per):
        chunk = sentences[i : i + per]
        words = chunk[0].rstrip(".!?").split()
        headline = " ".join(words[:7]) + ("…" if len(words) > 7 else "")
        out.append(Scene(headline=headline, points=[], narration=" ".join(chunk)))
    return Plan(title="", mood="tech", scenes=out)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}\n{result.stderr[-2000:]}")
    return result.stdout


def media_duration(path: Path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)])
    return float(out.strip())


def word_count(text: str) -> int:
    return max(1, len(text.split()))


def time_with_own_audio(plan: Plan, audio: Path) -> float:
    """Spread the recording's length over the scenes by how much is said in each."""
    total = media_duration(audio)
    words = [word_count(s.narration) for s in plan.scenes]
    for s, w in zip(plan.scenes, words):
        s.duration = total * w / sum(words)
    return total


def time_with_tts(plan: Plan, workdir: Path, voice: str, wpm: int) -> Path:
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak:
        sys.exit("No text-to-speech found. Install espeak-ng, or use --audio / --voice none.")
    pad = 0.5
    parts = []
    for i, s in enumerate(plan.scenes):
        wav = workdir / f"scene{i:02d}.wav"
        run([espeak, "-v", voice, "-s", str(wpm), "-w", str(wav), s.narration])
        # Normalize format and add a short pause so scenes can breathe.
        norm = workdir / f"scene{i:02d}_n.wav"
        run(["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-af", f"apad=pad_dur={pad}",
             "-ar", "44100", "-ac", "1", str(norm)])
        s.audio = norm
        s.duration = media_duration(norm)
        parts.append(norm)
    listfile = workdir / "parts.txt"
    listfile.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    full = workdir / "narration.wav"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listfile),
         "-c", "copy", str(full)])
    return full


def time_silent(plan: Plan, wpm: int) -> None:
    for s in plan.scenes:
        reading = word_count(s.narration) / (wpm / 60)
        s.duration = max(3.5, reading + 1.0)


def transcribe(audio: Path) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("To transcribe audio, install faster-whisper (pip install faster-whisper), "
                 "or pass the text with --text.")
    print("Transcribing audio…", file=sys.stderr)
    model = WhisperModel("base", device="auto", compute_type="auto")
    segments, _ = model.transcribe(str(audio))
    return " ".join(seg.text.strip() for seg in segments)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def caption_chunks(text: str, size: int = 7) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)] or [""]


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def blend(c1, c2, a: float):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * a) for i in range(3))


class Renderer:
    def __init__(self, plan: Plan, size: tuple[int, int], font_override: str | None,
                 captions: bool):
        self.plan = plan
        self.W, self.H = size
        self.captions = captions
        top, bottom, accent, text = MOODS[plan.mood]
        self.top, self.bottom = hex_rgb(top), hex_rgb(bottom)
        self.accent, self.text = hex_rgb(accent), hex_rgb(text)

        unit = min(self.W, self.H)
        self.margin = int(unit * 0.08)
        self.f_head = load_font("bold", int(unit * 0.085), font_override)
        self.f_title = load_font("bold", int(unit * 0.10), font_override)
        self.f_point = load_font("regular", int(unit * 0.048), font_override)
        self.f_cap = load_font("regular", int(unit * 0.040), font_override)
        self.f_small = load_font("regular", int(unit * 0.030), font_override)

        # A gradient background per scene, shifted slightly so scenes feel distinct.
        self.backgrounds = [self._gradient(i) for i in range(len(plan.scenes))]
        self.title_bg = self._gradient(0)

    def _gradient(self, shift: int) -> Image.Image:
        a = blend(self.top, self.bottom, 0.12 * (shift % 3))
        b = blend(self.bottom, self.accent, 0.10 * (shift % 2))
        grad = Image.linear_gradient("L").resize((self.W, self.H))
        img = Image.merge("RGB", [g.point(lambda v, lo=lo, hi=hi: lo + (hi - lo) * v // 255)
                                  for g, lo, hi in zip([grad] * 3, a, b)])
        # Soft accent glow in a corner.
        glow = Image.new("RGB", (self.W, self.H), self.accent)
        mask = Image.radial_gradient("L").resize((self.W * 2, self.H * 2))
        mask = mask.point(lambda v: max(0, 255 - v) * 0.22)
        corner = (-self.W // 2, -self.H) if shift % 2 else (-self.W // 2 + self.W // 3, 0)
        offset_mask = Image.new("L", (self.W, self.H), 0)
        offset_mask.paste(mask, corner)
        img.paste(glow, (0, 0), offset_mask)
        return img

    def title_frame(self, t: float, length: float) -> Image.Image:
        img = self.title_bg.copy()
        d = ImageDraw.Draw(img)
        a = ease(t / 0.6) * (1 - ease((t - (length - 0.3)) / 0.3))
        lines = wrap(d, self.plan.title, self.f_title, self.W - 2 * self.margin)
        lh = int(self.f_title.size * 1.15)
        y = (self.H - lh * len(lines)) // 2 - int((1 - a) * 40)
        color = blend(self.top, self.text, a)
        for line in lines:
            w = d.textlength(line, font=self.f_title)
            d.text(((self.W - w) / 2, y), line, font=self.f_title, fill=color)
            y += lh
        bar_w = int((self.W * 0.25) * a)
        d.rectangle([(self.W - bar_w) / 2, y + 30, (self.W + bar_w) / 2, y + 40], fill=self.accent)
        return img

    def scene_frame(self, idx: int, t: float, elapsed_total: float, total: float) -> Image.Image:
        s = self.plan.scenes[idx]
        img = self.backgrounds[idx].copy()
        d = ImageDraw.Draw(img)
        m = self.margin
        maxw = self.W - 2 * m

        # Scene counter.
        d.text((m, m), f"{idx + 1:02d} / {len(self.plan.scenes):02d}", font=self.f_small,
               fill=blend(self.top, self.text, 0.6))

        # Headline slides up and fades in.
        a = ease(t / 0.5)
        head_lines = wrap(d, s.headline, self.f_head, maxw)
        lh = int(self.f_head.size * 1.12)
        block_h = lh * len(head_lines) + 40
        pl_h = int(self.f_point.size * 1.35)
        point_lines = [wrap(d, p, self.f_point, maxw - 60) for p in s.points]
        block_h += sum(len(pl) * pl_h + 24 for pl in point_lines)
        y = max(m * 2, int(self.H * 0.42 - block_h / 2)) + int((1 - a) * 50)
        for line in head_lines:
            d.text((m, y), line, font=self.f_head, fill=blend(self.top, self.text, a))
            y += lh
        d.rectangle([m, y + 10, m + int(maxw * 0.18 * a), y + 20], fill=self.accent)
        y += 50

        # Points appear one by one over the first part of the scene.
        step = min(1.2, max(0.4, (s.duration * 0.5) / max(1, len(s.points))))
        for i, lines in enumerate(point_lines):
            pa = ease((t - 0.6 - i * step) / 0.4)
            if pa <= 0:
                break
            x = m + int((1 - pa) * 40)
            color = blend(self.bottom, self.text, pa)
            cy = y + pl_h // 2
            r = int(self.f_point.size * 0.18)
            d.ellipse([x, cy - r, x + 2 * r, cy + r], fill=blend(self.bottom, self.accent, pa))
            for line in lines:
                d.text((x + 60, y), line, font=self.f_point, fill=color)
                y += pl_h
            y += 24

        # Captions: the narration, a few words at a time.
        if self.captions and s.narration:
            chunks = caption_chunks(s.narration)
            ci = min(len(chunks) - 1, int(t / max(s.duration, 0.01) * len(chunks)))
            cap_lines = wrap(d, chunks[ci], self.f_cap, maxw - 60)
            ch = int(self.f_cap.size * 1.3)
            box_h = ch * len(cap_lines) + 40
            by = self.H - m - box_h - 30
            overlay = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.rounded_rectangle([m, by, self.W - m, by + box_h], radius=24, fill=(0, 0, 0, 140))
            img.paste(overlay, (0, 0), overlay)
            d = ImageDraw.Draw(img)
            cy = by + 20
            for line in cap_lines:
                w = d.textlength(line, font=self.f_cap)
                d.text(((self.W - w) / 2, cy), line, font=self.f_cap, fill=(255, 255, 255))
                cy += ch

        # Progress bar along the bottom.
        prog = min(1.0, elapsed_total / max(total, 0.01))
        d.rectangle([0, self.H - 10, int(self.W * prog), self.H], fill=self.accent)
        return img


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


def render_video(plan: Plan, out: Path, size, fps: int, title_len: float,
                 audio: Path | None, font: str | None, captions: bool) -> None:
    r = Renderer(plan, size, font, captions)
    W, H = size
    show_title = bool(plan.title) and title_len > 0
    lead = title_len if show_title else 0.0
    scenes_total = sum(s.duration for s in plan.scenes)
    total = lead + scenes_total

    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio:
        # Delay the voice so it starts when the first scene does.
        cmd += ["-i", str(audio)]
        delay_ms = int(lead * 1000)
        cmd += ["-filter_complex", f"[1:a]adelay={delay_ms}|{delay_ms},apad[a]",
                "-map", "0:v", "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin
    n_frames = int(math.ceil(total * fps))
    bounds = []
    acc = lead
    for s in plan.scenes:
        bounds.append((acc, acc + s.duration))
        acc += s.duration

    idx = 0
    try:
        for f in range(n_frames):
            t = f / fps
            if t < lead:
                frame = r.title_frame(t, lead)
            else:
                while idx < len(bounds) - 1 and t >= bounds[idx][1]:
                    idx += 1
                frame = r.scene_frame(idx, t - bounds[idx][0], t - lead, scenes_total)
            proc.stdin.write(frame.tobytes())
            if f % fps == 0:
                print(f"\rRendering {f / n_frames:5.0%}", end="", file=sys.stderr)
    finally:
        proc.stdin.close()
        code = proc.wait()
    print("\rRendering 100%", file=sys.stderr)
    if code != 0:
        sys.exit("ffmpeg failed while encoding the video.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Turn voice-to-text into a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("text", nargs="?", help="Transcript file. Use '-' or leave out to read stdin.")
    ap.add_argument("-o", "--out", default="video.mp4", help="Output MP4 path.")
    ap.add_argument("--audio", help="Your voice recording. Used as the soundtrack "
                    "(and transcribed if no text is given).")
    ap.add_argument("--voice", default="none", choices=["none", "tts"],
                    help="When no --audio: 'tts' reads the script with espeak-ng, "
                    "'none' makes a silent video with captions.")
    ap.add_argument("--tts-voice", default="en-us", help="espeak-ng voice name.")
    ap.add_argument("--format", default="vertical", choices=list(SIZES))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scenes", type=int, help="Number of scenes (default: Claude decides).")
    ap.add_argument("--mood", choices=list(MOODS), help="Override the color mood.")
    ap.add_argument("--title-seconds", type=float, default=2.0,
                    help="Length of the opening title card. 0 to skip.")
    ap.add_argument("--no-captions", action="store_true", help="Hide the spoken-words captions.")
    ap.add_argument("--font", help="Path to a .ttf/.otf font to use instead of the default.")
    ap.add_argument("--offline", action="store_true",
                    help="Skip Claude and split the text into scenes by sentences.")
    ap.add_argument("--plan-only", action="store_true",
                    help="Print the scene plan as JSON and stop.")
    ap.add_argument("--plan", help="Use a saved plan JSON file instead of asking Claude.")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg is required. Install it from https://ffmpeg.org/download.html")

    audio = Path(args.audio).expanduser() if args.audio else None
    if audio and not audio.exists():
        sys.exit(f"Audio file not found: {audio}")

    # 1. Transcript and plan.
    if args.plan:
        plan = plan_from_dict(json.loads(Path(args.plan).read_text()))
    else:
        if args.text and args.text != "-":
            transcript = Path(args.text).expanduser().read_text()
        elif audio and sys.stdin.isatty():
            transcript = transcribe(audio)
        else:
            if sys.stdin.isatty():
                print("Paste your dictation, then press Ctrl-D:", file=sys.stderr)
            transcript = sys.stdin.read()
        transcript = transcript.strip()
        if not transcript:
            sys.exit("No transcript text was given.")
        if args.offline:
            plan = plan_offline(transcript, args.scenes)
        else:
            print("Planning scenes with Claude…", file=sys.stderr)
            plan = plan_with_claude(transcript, args.scenes)
    if args.mood:
        plan.mood = args.mood

    if args.plan_only:
        print(json.dumps({"title": plan.title, "mood": plan.mood,
                          "scenes": [{"headline": s.headline, "points": s.points,
                                      "narration": s.narration} for s in plan.scenes]},
                         indent=2))
        return

    # 2. Soundtrack and timing.
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        soundtrack = None
        if audio:
            time_with_own_audio(plan, audio)
            soundtrack = audio
        elif args.voice == "tts":
            soundtrack = time_with_tts(plan, workdir, args.tts_voice, 165)
        else:
            time_silent(plan, 160)

        # 3. Render.
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        render_video(plan, out, SIZES[args.format], args.fps, args.title_seconds,
                     soundtrack, args.font, not args.no_captions)

    total = sum(s.duration for s in plan.scenes) + (args.title_seconds if plan.title else 0)
    print(f"Done: {out} ({len(plan.scenes)} scenes, {total:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
