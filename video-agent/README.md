# voice2video

Turns your voice-to-text (from Wispr Flow, your phone's dictation, etc.) into a finished MP4 video.

What it does:

1. Takes your dictated text (and, if you want, the original voice recording).
2. Sends the text to Claude, which removes filler words, fixes dictation mistakes, and splits it into scenes. Each scene gets a headline, up to 3 short points, and narration.
3. Draws every frame (colored background, animated headline and points, captions, progress bar) and builds the video with ffmpeg.
4. Adds sound: your own recording, a computer voice, or none.

## Setup

You need Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html).

```bash
cd video-agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # from https://console.anthropic.com
```

Optional:
- Computer voice: install `espeak-ng` (`brew install espeak-ng` / `sudo apt install espeak-ng`).
- Transcribe audio when you have no text: `pip install faster-whisper`.

## Use it

```bash
# Text file -> silent vertical video with captions
python voice2video.py my-dictation.txt -o video.mp4

# Paste text straight in (press Ctrl-D when done)
python voice2video.py -o video.mp4

# Use your own voice recording as the soundtrack
python voice2video.py my-dictation.txt --audio memo.m4a -o video.mp4

# Only have the recording? It gets transcribed first (needs faster-whisper)
python voice2video.py --audio memo.m4a -o video.mp4

# Computer voice reads the cleaned-up script
python voice2video.py my-dictation.txt --voice tts -o video.mp4

# YouTube shape, 5 scenes, warm colors
python voice2video.py my-dictation.txt --format landscape --scenes 5 --mood warm
```

### Check or edit the script before rendering

```bash
python voice2video.py my-dictation.txt --plan-only > plan.json
# edit plan.json by hand
python voice2video.py --plan plan.json --audio memo.m4a -o video.mp4
```

### Options

| Option | What it does |
|---|---|
| `--format vertical\|landscape\|square` | 1080x1920 (Reels/TikTok/Shorts), 1920x1080 (YouTube), 1080x1080 |
| `--audio FILE` | Your recording becomes the soundtrack; scenes are timed to it |
| `--voice tts` | Computer voice (espeak-ng) when there's no `--audio` |
| `--scenes N` | Fixed number of scenes (default: Claude picks 3–8) |
| `--mood` | `calm`, `bold`, `warm`, `tech`, `fresh`, `dark` |
| `--title-seconds N` | Length of the opening title card (0 to skip) |
| `--no-captions` | Hide the words at the bottom |
| `--font FILE` | Use your own .ttf/.otf font |
| `--offline` | No Claude: split the text by sentences (rough, but free) |

## Notes

- When you use your own recording, scene timing is estimated from how many words are in each scene, so it lines up closely but not to the exact word.
- Rendering takes roughly as long as the video itself (about 1 second per second of video).
- Claude model used: `claude-opus-5-5`. If a request is declined by a safety check, it's retried automatically on a fallback model.
