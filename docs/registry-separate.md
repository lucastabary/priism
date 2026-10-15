# Registry separator (prototype)

Turns the source-registry research (phases 1-2: whole-song clustering of BS-Roformer-SW tokens with the
identity head) into audio stems, with no new training, and measures it on MoisesDB. Code:
`src/priism/lab/registry/separate.py` (pipeline) and `separate_eval.py` (evaluation); tests in
`tests/test_registry_separate.py` (fake model, CPU).

## Pipeline

1. **One SW pass** over the song, in the cache's windows (8 s, 1 s margins dropped at inner edges; the
   frames kept are exactly those of `features.song_tokens`). Per window it keeps:
   - the 6 stem STFT estimates (complex mask x mix STFT, as `BSRoformer.forward` builds them before its
     iSTFT), stitched frame by frame into whole-song spectra `(6, C, F, T)`;
   - the trunk taps (`--layers`, default `6,final` as `cache-b6`), projected with the cache's `proj.npz`
     window by window, then pooled x4 and cast to float16: the very tokens the head was trained on
     (`Tp x 62` tokens of ~46 ms x one band).
2. **Embedding**: the identity head (`train-id` `best.pt`), or the raw tokens without `--head`.
3. **Per stem split**: a token is *owned* by a stem when the stem holds `>= --own` (0.5) of the 6 stems'
   power in it and it is audible (60 dB floor). The stem's owned tokens are clustered over the whole song
   (`track.whole_song`, cosine cut `--cut`); clusters with fewer than `--min-share` (2 %) of the owned
   tokens are dropped; every token of the grid then goes to the nearest kept centroid (hard, or softmax
   with `--temp`). A stem owning fewer than 64 tokens is not split.
4. **Render**: sub-source k of stem s = stem estimate x weight k, the weight upsampled from (token frame,
   band) to (STFT frame, bin) and smoothed over `--smooth` token frames (triangular, renormalised),
   then iSTFT at the exact input length.

The weights of a stem sum to 1 on every token and the iSTFT is linear, so **a stem's sub-sources sum to
its SW estimate** (to float rounding): the split only redistributes the stem. Inputs at another rate
than the model's (44.1 kHz) are resampled in, and outputs resampled back and fitted to the exact input
frame count; mono inputs give mono outputs.

Memory: the stem spectra dominate, complex64, ~0.6 GB per minute of audio (3 GB for 5 min). The full
trunk taps never exist for the whole song. GPU: one 8 s window at a time (SW in bf16).

## CLI

```bash
priism-lab registry-separate song.wav --out out/song \
  --msst-config /workspace/priism/models/BS-Roformer-SW.yaml \
  --msst-ckpt /workspace/priism/models/BS-Roformer-SW.ckpt --msst-path /workspace/priism/msst \
  --proj /root/registry/cache-b6/proj.npz --head /root/registry/p6/fx09-s0/best.pt \
  --cut 0.3 [--write-stems]
```

The input may also be a MoisesDB song folder or a NI `.stem.mp4` (the sum of its tracks / stems is
separated). Output: `<stem>_<k>.wav` (float WAV, cluster 0 the largest), `sw/<stem>.wav` with
`--write-stems`, and `sources.json` (written last) listing each file's stem, cluster, owned tokens and
level relative to the mix, the settings and the timings.

## Evaluation

```bash
priism-lab registry-separate-eval --set multitrack:/root/registry/data/moisesdb%test --out eval/test \
  <same model, --proj, --head options> --cut 0.3 [--limit 5] [--songs <folder name> ...]
```

Systems, all from the same SW pass:

| system | outputs | matching |
| --- | --- | --- |
| `sw` | the 6 SW stems (today's engine) | each reference to its best stem, many-to-one |
| `sw_1to1` | same | Hungarian on SDR |
| `registry` (`registry@<cut>` with several cuts) | the stems split by the head's clusters | Hungarian on SDR; `_m2o`: many-to-one |
| `oracle` | the same token masks, labels from the references' power per token (each reference assigned to its best SW stem, each token of a stem to the loudest of that stem's references) | Hungarian; `_m2o` |
| `decoder` (with `--decoder`; `decoder@<cut>`) | the registry's clusters rendered with the fine decoder's bin masks (below; a second SW pass) | Hungarian; `_m2o` |
| `oracle_bin` | ideal ratio masks at bin resolution among each stem's references (assigned as for `oracle`), on the SW stem: the ceiling of any real mask on SW stems | Hungarian; `_m2o` |

A reference left without an output in a one-to-one matching scores 0 dB (the SDR of a silent estimate);
extra outputs are ignored. The oracle is the upper bound of any labelling on this token grid: what is lost
there comes from the grid (two sources in one band and 46 ms frame) or from SW's stem estimate itself.

Metrics, whole track, channels concatenated, eps = 1e-8, no BSS-eval projection:
`SDR = 10 log10(|s|^2 / |s - ŝ|^2)` (MDX "global SDR") and SI-SDR with `a = <ŝ, s> / |s|^2` (Le Roux et
al. 2019, no mean removal). References more than `--min-ref-db` (40 dB) under the mix are not scored.

Output: `songs.jsonl` (one line per song, every reference and system; reruns skip songs already there) and
`summary.json`: per system and metric, mean, median, std and artist-balanced mean, by MoisesDB stem and
track type, on all references and on the *shared* ones (whose best SW stem holds another reference of the
song: the case the registry is for), plus paired comparisons on per-song mean SDR (registry and oracle vs
`sw`, oracle vs registry, decoder vs registry at the same cut, oracle_bin vs decoder and vs oracle: bootstrap 95 % CI, d_z, Wilcoxon). Progress: one flushed `song i/N ...` line
per song.

Protocol: tune `--cut` (several cuts share one SW pass: `--cut 0.2 0.3 0.4 0.5`), and if needed `--own`
and `--smooth`, on `moisesdb%dev2`; freeze them and report `moisesdb%test` (artist split of
`run.in_part`; neither the PCA nor the head saw test artists). Compare per song, not per reference
(references of one song are not independent).

## Fine decoder (`train-decoder`, `--decoder`)

Code: `src/priism/lab/registry/decoder.py`, tests `tests/test_decoder.py` (fake trunk, CPU). It keeps the
registry's queries (cluster centroids of the identity head) but predicts, per query, a mask at the STFT's
resolution (frame x bin x channel) instead of one weight per token, to lift the token-grid limit that
`oracle` measures (DANet-style attractors, CLAPSep-style FiLM on a frozen encoder):

- input: the SW taps (`6,final`, 2 x 256 per frame and band, full frame rate) -> linear to 128, plus the
  stem's band log power and power share; FiLM by (query, stem id); 1 axial RoFormer block (time, then
  bands, rotary positions); a per-band MLP (BS-Roformer MaskEstimator layout) that also reads the stem's
  log magnitude at each bin of the band -> one logit per (frame, bin, channel);
- the logits of one stem's K queries go through a softmax over K: real ratio masks on the SW stem estimate
  that sum to 1, so **sub-sources sum to the SW stem exactly** (the contract holds);
- start point: the logits get the prior cos(head embedding of the token, query) / 0.1 and the last layer
  starts at zero, so step 0 is the registry's soft token split; the first validation is the bar to beat;
- ~3.5 M trainable parameters; SW and the identity head stay frozen.

Training: random crops (6 s) of `priism-lab gen` songs, SW online in bf16 under no_grad, nothing
precomputed on disk. Each source present in the crop (>= -30 dB under the mix) goes to the SW stem holding
most of its energy; its query is the centroid of the head embeddings of the tokens it owns in the crop
(plus a little noise: inference queries are whole-song cluster centroids). Only stems holding >= 2 present
sources train. Loss per sub-source (vs its reference, signals divided by the crop mix rms): L1-SNR on the
waveform + multi-resolution STFT magnitude L1-SNR (512 / 1024 / 2048), plus a level term against outputs
more than 3 dB quieter than their reference. Validation: fixed crops of held-out songs, SI-SDR improvement
over the whole SW stem (`si_sdri`), and once the ideal-ratio-mask ceiling on the same crops
(`si_sdri_irm`). Guards: non-finite loss, `--plateau` evaluations without `--plateau-delta` dB, `--max-minutes`;
`history.jsonl` (flushed), `best.pt`, `last.pt`, `summary.json`. `max_vram_gb` and `sec_per_step` are logged.

```bash
priism-lab train-decoder --data /root/registry/data/dec-train --valid /root/registry/data/dec-valid   --head /root/registry/p6/fx09-s0/best.pt --proj /root/registry/cache-b6/proj.npz   <--msst-config / --msst-ckpt / --msst-path> --out /root/registry/dec/s0 --steps 1000 --batch 4
priism-lab registry-separate-eval ... --decoder /root/registry/dec/s0/best.pt
```

Known gap: training queries come from ground-truth ownership in a crop, inference queries from whole-song
clusters (a cluster may hold two sources, or a source two clusters); `decoder` vs `registry` on MoisesDB is
the test that counts, not the synthetic validation.

## Known limitations

- Masks act on the SW stem estimate: what SW put in the wrong stem stays there (the oracle shares this
  bound), and token masks are binary per (46 ms, band) unless `--temp` > 0.
- The cut is global; a stem with one source should not be split, and depends on `--min-share` to avoid
  scattering it. Many-to-one scores show the cost of over-splitting.
- The head was trained at the instrument level on synthetic songs: twins (two takes of one instrument)
  are expected to merge.
- MoisesDB references with bleed (`has_bleed`) are scored as they are; the label is kept per reference
  for filtering.
