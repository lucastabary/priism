"""Source registry over whole songs (plan of 2026-10-11, replaces plan-v2).

A separator that sees 4 s at a time cannot know that the 303 it hears now is the one from a minute ago.
Here the song is read in order, chunk by chunk, and a registry keeps one identity per source over the
whole song. Phases 1 and 2 work on cached features of the frozen BS-Roformer-SW trunk, so no separator is
trained yet:

- ``features``: the trunk run once per song, its tokens (frame x band) reduced by a PCA fitted on training
  songs, plus each reference source's energy per token (who owns which token);
- ``track``: ways to give every token a song-wide source id (per-chunk clustering, whole-song clustering,
  an online registry that grows when something new enters);
- ``metrics``: how consistent those ids are over the song (purity, completeness, permutations, entries);
- ``head``: the identity head trained on the cache (phase 2).
"""
