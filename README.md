# priism

Stem separation engine for DJ preparation. priism splits a track into stems, possibly many and from
several models, writes them losslessly with a `manifest.json`, and exports Native Instruments
`.stem.mp4` files from a 4-slot grouping. It is driven by **MyMixxx**, a fork of
[Mixxx](https://mixxx.org), through a small command line contract
([docs/contract.md](docs/contract.md)).

Guarantees, whatever models are combined:

- every stem has the track's sample rate and exact frame count, aligned on frame 0, so cues and
  beatgrids carry over;
- stems + residual sum back to the track (bit-exact for 16/24-bit sources);
- the NI export is lossless ALAC by default, decoding bit-exact and aligned (checked in Mixxx on
  Windows); AAC is optional, but Mixxx on Windows decodes it 661 frames late.

## Install

Python 3.11+, ffmpeg on PATH.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -e ".[demucs]"     # add dev for the tests
```

## Use

```bash
priism separate track.wav --out out/track --profile standard --progress jsonl
priism export-ni out/track --project out/track/project.json --out track.stem.mp4
priism version --json
```

`separate` writes `out/track/stems/*.flac` and, last, `out/track/manifest.json`. `export-ni` reads a
`project.json` saying which stems go into the 4 slots, with names, colours and gains. With
`--progress jsonl`, progress, warnings and the result arrive as JSON lines on stdout.

For Mixxx, pass the track **as Mixxx decodes it** (a WAV written by MyMixxx), not the MP3 itself:
Mixxx's MP3 decoder does not start at the same sample as other decoders.

## Profiles

A profile is a tree of subtractive steps: a model splits its input, and what its named outputs
leave is passed on to a later step or to the residual. The best model for each stem can then go
first without any stem being counted twice. See `src/priism/engine/profile.py`.

| Profile | Models | Stems |
|---|---|---|
| `standard` | htdemucs_ft (4 fine-tuned models) | drums, bass, other, vocals |
| `fast` | htdemucs (1 model) | drums, bass, other, vocals |
| `fake` | none (band filters) | low, mid, high: for testing clients |

## Development

```bash
python -m pytest tests/test_engine_contract.py tests/test_engine_separate.py
python scripts/make_fixture.py      # regenerate fixtures/contract after a contract change
```

## Research code

`priism.lab` (command `priism-lab`, extra `[lab]`) holds the research side: synthetic song generators,
a class-agnostic attractor separator built on a pretrained BS-RoFormer, and the RunPod job queue
used to train it. See [docs/modele.md](docs/modele.md) and [docs/pod.md](docs/pod.md) (in French).
