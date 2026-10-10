# priism ↔ MyMixxx contract (schema 1.3)

priism is the stem separation engine of MyMixxx. MyMixxx runs it as a command line program
(`QProcess`); they exchange files described here. The JSON Schemas in
[`src/priism/engine/schema/`](../src/priism/engine/schema) are the machine-readable part; this page
adds the rules a schema cannot express. A shared example lives in
[`fixtures/contract/`](../fixtures/contract) (regenerate with `python scripts/make_fixture.py`).

Any change goes through Lucas and bumps `schema_version`: minor for additive, backward compatible
changes; major for breaking ones. Readers accept any minor of a major they support and ignore
unknown fields.

| Version | Change |
|---|---|
| 1.0 | Initial proposal (MyMixxx brief, 2026-10-10). |
| 1.1 | `source.decoder`, `profile`, `models` table referenced by `stems[].models`, optional `stems[].parent`; input decoded by MyMixxx; `export-ni` peak report. |
| 1.2 | `export-ni --codec alac\|aac` (ALAC by default: Mixxx on Windows decodes AAC 661 frames late), `--on-clip error\|warn` replaces `--fail-on-clip`; `codec` in the `done` event; peaks in the clipping `error` event. |
| 1.3 | `version --json` lists each profile's progress `stages` in order. |

## Commands

```
priism separate <audio> --out <dir> [--profile <name|file.toml>] [--source-path <original>]
                [--source-sha256 <hex>] [--source-decoder <str>] [--device cpu|cuda] [--progress jsonl]
priism export-ni [<dir>] [--project <project.json>] --out <file.stem.mp4> [--codec alac|aac]
                 [--on-clip error|warn] [--progress jsonl]
priism version [--json]
```

- Exit code 0 on success. On failure: non-zero exit code and a message on stderr.
- `--progress jsonl`: one JSON object per line on stdout, flushed immediately:
  - `{"event":"progress","stage":"step1:demucs:htdemucs_ft","fraction":0.42}`. Stages:
    `decode`, `load:<model>`, `step<N>:<model>`, `write` (separate); `mixdown`, `encode` (export-ni).
    `fraction` is per stage, in [0, 1].
  - `{"event":"warning","message":"..."}`
  - `{"event":"done", ...}`: `separate` gives `{"manifest": "<path>"}`; `export-ni` gives
    `{"output": "<path>", "codec": "alac", "peaks_dbfs": {"mix": -0.5, "slots": [x, x, x, x]}, "clipped": ["<track name>", ...]}`.
  - `{"event":"error","message":"..."}` (also written to stderr). When `export-ni` refuses to clip,
    the event also carries `peaks_dbfs` and `clipped`, so the client can offer to lower the gains.
- Killable at any time. `separate` deletes any previous `manifest.json` before starting and writes
  the new one last, atomically (temporary file + rename): a folder without `manifest.json` is
  unfinished. `export-ni` writes its output atomically too.
- `version --json` gives `priism_version`, `schema_version`, `supported_schema_majors`, the bundled
  `profiles` (version, description, models, `stages`) and the available `models`. `stages` lists, in
  order, the stages a `separate` run with that profile emits, so a client can lay out a global
  progress bar before the first event: `decode`, `load:<model>` per model, `step<N>:<model>` per step
  of the separation tree (N from 1), `write`. Stage names are stable within a major version.

## Input of `separate` and the timeline

Every stem must start at frame 0 of the track **as Mixxx decodes it**, since MyMixxx reuses the
track's cues and beatgrid on the stems. Mixxx's decoders do not match other decoders (its MP3
decoder skips the first MP3 frame and ignores the LAME encoder delay; AAC goes through
MediaFoundation on Windows). So **MyMixxx decodes the track itself** (`SoundSourceProxy`) and passes
a PCM WAV at the track's native rate, plus:

- `--source-path`: the original file, recorded in the manifest;
- `--source-sha256`: the caller's identity hash of the track. MyMixxx uses the SHA-256 of the
  decoded audio (interleaved float32 little-endian samples, as decoded by Mixxx, no header), which
  survives tag rewrites. Without the flag, priism hashes the input file;
- `--source-decoder`: e.g. `mixxx:SoundSourceMp3`, informative.

priism reads the WAV as it is: no resampling, no trimming. Mono is copied to both channels. Other
formats are accepted for standalone use (`decoder` = `priism:ffmpeg` or `priism:libsndfile-*`), with
no guarantee that their timeline matches Mixxx's.

## `manifest.json` (output of `separate`)

```json
{
  "schema_version": "1.3",
  "priism_version": "0.2.0",
  "profile": {"name": "standard", "version": "1"},
  "source": {"path": "D:/Music/track.mp3", "sha256": "…", "sample_rate": 44100, "channels": 2,
             "frames": 10584000, "decoder": "mixxx:SoundSourceMp3"},
  "models": {"demucs:htdemucs_ft": {"backend": "demucs", "name": "htdemucs_ft", "version": "demucs-4.1.0",
                                     "sha256": "…"}},
  "stems": [
    {"id": "vocals", "label": "Vocals", "kind": "vocals", "file": "stems/vocals.flac",
     "models": ["demucs:htdemucs_ft"], "parent": "mix"}
  ],
  "residual": {"file": "stems/residual.flac"},
  "suggested_slots": [["drums"], ["bass"], ["other", "residual"], ["vocals"]]
}
```

Rules beyond the schema:

- **Alignment**: every stem file has `source.sample_rate`, exactly `source.frames` frames, 2 channels,
  and frame 0 = frame 0 of the input.
- **Sum**: stems + residual = source. The stems are stored exactly; the only error is the residual's
  24-bit rounding, at most 2⁻²⁴ (−144 dBFS) per sample, zero for a 16- or 24-bit PCM input. A file
  whose signal exceeds full scale is stored as 32-bit float WAV instead of FLAC (exact to float32
  precision).
- **Files**: lossless, FLAC 24-bit or WAV float. Paths are POSIX, relative to the manifest's folder,
  without `..`.
- **Ids**: `^[a-z0-9_]+$`, unique; `residual` and `mix` are reserved. The residual is referenced as
  `residual` (in `suggested_slots` and projects) but is not listed in `stems`.
- **kind**: open vocabulary (`vocals`, `drums`, `bass`, `guitar`, `piano`, `synth`, `strings`, `fx`,
  `other`, …). Clients must accept unknown kinds.
- **models**: every id in `stems[].models` is a key of the `models` table. A stem lists every model
  its signal went through, in order (a stem split from an instrumental lists the vocal model too).
  `sha256` fingerprints the weights.
- **parent**: the node of the separation tree a stem was split from: `mix` (the source) or an
  intermediate node that is not itself a stem.
- **suggested_slots**: at most 4 groups of stem ids (or `residual`), a default for the editor.
- **Determinism**: same input, profile version, models and priism version give the same stems
  (CPU; GPU kernels may differ in the last bits).

## `project.json` (written by MyMixxx, input of `export-ni`)

```json
{
  "schema_version": "1.3",
  "manifest": "manifest.json",
  "slots": [
    {"name": "Drums",  "color": "#FF5A3C", "stems": ["drums", "percussion"], "gain": [1.0, 1.0]},
    {"name": "Bass",   "color": "#3C8CFF", "stems": ["bass"]},
    {"name": "Music",  "color": "#FFD23C", "stems": ["other", "residual"]},
    {"name": "Vocals", "color": "#3CFF8C", "stems": ["vocals"]}
  ],
  "discarded": []
}
```

Rules beyond the schema: exactly 4 slots with unique names; every stem id exists in the manifest
(or is `residual`); a stem is in at most one slot and never both used and discarded; `gain`
(linear, default 1.0) has one value per stem; a slot may be empty (silent track). Stems in no slot
are left out of the slots; the mix track still holds the whole source.

## The NI `.stem.mp4` (output of `export-ni`)

- 5 stereo tracks, same codec, the source's sample rate: track 0 = the full mix (sum of all stems
  and the residual, i.e. the source), tracks 1–4 = the slots, slot = Σ gain × stem.
- **Codec: ALAC 24-bit by default** (what MyMixxx uses). Lossless and without encoder delay: every
  track decodes bit-exact (for stems on the 24-bit grid) to exactly `source.frames` frames. Checked
  in Mixxx on Windows (`SoundSourceSTEM`): lag 0, max error 0.
- `--codec aac` (256 kb/s, about 4× smaller): the encoder delay is recorded in an edit list, which
  FFmpeg's native decoder honours (lag 0), but Mixxx on Windows decodes AAC with libfdk_aac and comes
  out **661 frames late**. Not for MyMixxx; kept for sharing and other software.
- `moov/udta/stem` box: JSON `{"version": 1, "stems": [{"name", "color"} × 4], "mastering_dsp": …}`
  with the compressor and limiter disabled (Traktor would otherwise process the slots).
- No limiter and no silent gain change. A track exceeding full scale is an **error by default with
  ALAC** (integer PCM would clip it), with the peaks in the error event; `--on-clip warn` clips it
  instead, with a `warning` event. With AAC the default is `warn` and the peaks are kept as is.
