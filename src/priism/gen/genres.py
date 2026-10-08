"""Genre templates: tempo, groove and which sources a song of that genre tends to use.

A template does not fix the line-up. It gives weights; the arranger draws the
number of sources first (uniform over 2..16) and then fills the line-up from
these weights, so every genre shows up with sparse and with dense line-ups.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Source kinds the arranger knows. Drum kinds are single elements of a kit.
DRUM_KINDS = ["kick", "snare", "clap", "rim", "hat_closed", "hat_open", "ride", "crash", "tom", "cowbell",
              "clave", "conga", "shaker"]
TONAL_KINDS = ["sub", "bass", "acid", "skank", "stab", "pad", "lead", "arp", "pluck", "siren", "noise_fx"]


@dataclass(frozen=True)
class Genre:
    name: str
    bpm: tuple[float, float]
    drum_style: str  # key of drums.STYLES
    bass_style: str  # key of tonal.BASS_STYLES
    swing: tuple[float, float]  # 16th swing range, 0 is straight
    modes: tuple[str, ...]
    weights: dict[str, float] = field(default_factory=dict)  # source kind -> how likely to be picked
    required: tuple[str, ...] = ()  # kinds put first when the line-up has room (backbone of the genre)


GENRES: dict[str, Genre] = {g.name: g for g in [
    Genre("techno", (124, 140), "four_floor", "offbeat", (0.0, 0.08), ("minor", "phrygian", "dorian"),
          {"hat_closed": 3, "hat_open": 2, "clap": 2, "rim": 1, "ride": 1.5, "crash": 0.5, "tom": 1, "shaker": 1,
           "cowbell": 0.5, "bass": 2, "sub": 1, "acid": 1, "stab": 1.5, "pad": 1, "lead": 1, "arp": 1, "pluck": 1,
           "noise_fx": 1.5},
          ("kick", "hat_closed", "bass")),
    Genre("acid", (125, 145), "four_floor", "offbeat", (0.0, 0.1), ("minor", "phrygian", "minor_pentatonic"),
          {"hat_closed": 3, "hat_open": 2, "clap": 2, "snare": 1, "ride": 1, "cowbell": 0.7, "acid": 4,
           "bass": 1, "sub": 1, "stab": 1, "pad": 0.7, "arp": 0.7, "noise_fx": 1},
          ("kick", "acid", "hat_closed", "clap")),
    Genre("dub_techno", (115, 128), "four_floor", "offbeat", (0.0, 0.1), ("minor", "dorian"),
          {"hat_closed": 3, "hat_open": 1.5, "rim": 1.5, "shaker": 1.5, "ride": 1, "stab": 4, "skank": 1.5,
           "pad": 2.5, "sub": 2, "bass": 1, "pluck": 1, "noise_fx": 1.5},
          ("kick", "stab", "sub", "hat_closed")),
    Genre("minimal", (120, 130), "four_floor", "offbeat", (0.05, 0.2), ("minor", "dorian"),
          {"hat_closed": 3, "rim": 2, "clap": 1.5, "shaker": 2, "clave": 1.5, "conga": 1.5, "cowbell": 1,
           "bass": 2.5, "pluck": 2, "stab": 1, "arp": 1, "noise_fx": 1},
          ("kick", "hat_closed", "bass")),
    Genre("house", (118, 128), "house", "house", (0.08, 0.25), ("minor", "dorian", "major"),
          {"hat_closed": 2, "hat_open": 3, "clap": 3, "shaker": 2, "conga": 1.5, "ride": 1, "crash": 0.5,
           "bass": 3, "pad": 2, "stab": 2, "lead": 1, "pluck": 1.5, "arp": 1},
          ("kick", "clap", "hat_open", "bass")),
    Genre("electro", (120, 135), "electro", "electro", (0.0, 0.1), ("minor", "phrygian"),
          {"snare": 2.5, "clap": 1.5, "hat_closed": 3, "hat_open": 1, "cowbell": 1.5, "tom": 1.5, "clave": 1,
           "bass": 3, "acid": 1, "lead": 2, "arp": 2, "stab": 1, "noise_fx": 1},
          ("kick", "snare", "hat_closed", "bass")),
    Genre("breakbeat", (126, 140), "breakbeat", "rolling", (0.0, 0.12), ("minor", "dorian"),
          {"snare": 3, "hat_closed": 3, "hat_open": 1.5, "ride": 1, "crash": 1, "tom": 1, "shaker": 1,
           "bass": 3, "acid": 1, "stab": 1.5, "pad": 1, "lead": 1, "noise_fx": 1},
          ("kick", "snare", "hat_closed", "bass")),
    Genre("dnb", (166, 178), "dnb", "reese", (0.0, 0.06), ("minor", "phrygian", "dorian"),
          {"snare": 3, "hat_closed": 3, "hat_open": 1.5, "ride": 1.5, "crash": 1, "shaker": 1, "tom": 0.7,
           "bass": 3, "sub": 2.5, "pad": 2, "stab": 1, "lead": 1, "arp": 1, "pluck": 1, "noise_fx": 1.5},
          ("kick", "snare", "hat_closed", "bass", "sub")),
    Genre("dub", (68, 90), "one_drop", "dub", (0.1, 0.3), ("minor", "dorian", "major"),
          {"rim": 2.5, "hat_closed": 3, "hat_open": 1, "snare": 1.5, "shaker": 1, "tom": 1, "cowbell": 0.5,
           "skank": 4, "bass": 3, "sub": 1, "pad": 1, "lead": 1, "siren": 2, "pluck": 0.5},
          ("kick", "bass", "skank", "rim")),
    Genre("steppers", (130, 150), "steppers", "dub", (0.05, 0.2), ("minor", "dorian"),
          {"rim": 2, "snare": 2, "hat_closed": 3, "hat_open": 1, "shaker": 1, "tom": 1, "skank": 3.5,
           "bass": 3, "sub": 1, "pad": 1, "lead": 1.5, "siren": 2, "stab": 1},
          ("kick", "bass", "skank", "snare")),
    Genre("dubstep", (138, 142), "halftime", "wobble", (0.0, 0.08), ("minor", "phrygian"),
          {"snare": 3, "hat_closed": 3, "hat_open": 1, "rim": 1, "shaker": 1, "tom": 0.7, "sub": 3, "bass": 2.5,
           "skank": 1, "pad": 1.5, "siren": 1, "lead": 1, "noise_fx": 1.5},
          ("kick", "snare", "sub", "hat_closed")),
]}
