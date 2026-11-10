"""Character voices: design one reference per character, then clone it for every line (Qwen3-TTS 1.7B, local).

  python episode/voices.py <chapter> design   3 candidate references per character (VoiceDesign) -> voices/design/
  python episode/pick_voices.py <chapter>     measures the candidates and proposes one (see that file)
  python episode/voices.py <chapter> lines    every line in the chosen voices (Base, clone) -> voices/<id>.wav

Why two models: VoiceDesign redraws the voice on every call, so it makes only the reference; every line is then cloned
from that one reference, which holds the voice from line to line. The chosen reference per character is
chapter.json -> characters.<name>.voice_pick (pick_voices.py proposes one; listen and overrule it).
Each line's "delivery" sentence goes to the clone as an instruction (how the line is said).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import chapter, paths  # noqa: E402

LANGUAGE = "English"


def load(kind):
    import torch
    from qwen_tts import Qwen3TTSModel
    P = paths()
    return Qwen3TTSModel.from_pretrained(P["tts_voice_design" if kind == "design" else "tts_base"], device_map="cuda:0",
                                         dtype=torch.bfloat16)


def design(d, ch):
    import numpy as np
    import soundfile as sf
    from transformers import set_seed
    m = load("design")
    out = d / "voices" / "design"
    out.mkdir(parents=True, exist_ok=True)
    for name, c in ch["characters"].items():
        for seed in (1, 2, 3):
            f = out / f"{name}-s{seed}.wav"
            if f.exists():
                continue
            set_seed(seed)
            wavs, sr = m.generate_voice_design(text=c["voice_ref_text"], instruct=c["voice_design"], language=LANGUAGE)
            sf.write(f, np.asarray(wavs[0], dtype=np.float32), sr)
            print("design", f.name, round(len(wavs[0]) / sr, 2), "s", flush=True)
    print("DESIGN-END")


def lines(d, ch):
    import numpy as np
    import soundfile as sf
    from transformers import set_seed
    m = load("base")
    prompts = {}
    for name, c in ch["characters"].items():
        assert c.get("voice_pick"), f"{name}: no voice_pick in chapter.json (run pick_voices.py, then listen)"
        ref = d / "voices" / "design" / c["voice_pick"]
        prompts[name] = m.create_voice_clone_prompt(ref_audio=str(ref), ref_text=c["voice_ref_text"], x_vector_only_mode=False)
    out = d / "voices"
    for k, ln in enumerate(ch["lines"]):
        f = out / f"{ln['id']}.wav"
        if f.exists():
            continue
        ids = m._tokenize_texts([m._build_instruct_text(ln["delivery"])])
        set_seed(ln.get("seed", 100 + k))
        wavs, sr = m.generate_voice_clone(text=ln.get("say", ln["text"]), language=LANGUAGE, voice_clone_prompt=prompts[ln["char"]],
                                          max_new_tokens=600, instruct_ids=ids)
        a = np.asarray(wavs[0], dtype=np.float32)
        assert len(a) / sr < 600 / 12 - 1, f"{ln['id']}: hit the token limit, the line may be cut off"
        sf.write(f, a, sr)
        print(f.name, round(len(a) / sr, 2), "s", flush=True)
    print("LINES-END")


if __name__ == "__main__":
    d, ch = chapter(sys.argv[1])
    {"design": design, "lines": lines}[sys.argv[2]](d, ch)
