# Priism

Séparation de stems par ML pour mixer en DJ. On commence par le dub et l'acid
(batterie / basse / acid / reste), avec l'objectif de couvrir à terme tous les
types de stems et tous les genres.

L'intégration Mixxx (empaquetage `.stem.mp4` avec stemgen) viendra plus tard.

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"            # générateur acid + tests
.venv/bin/pip install -e ".[separate]"       # séparation, GPU NVIDIA
.venv/bin/pip install -e ".[separate-cpu]"   # séparation, sans GPU
.venv/bin/pip install -e ".[fast]"           # numba, accélère le filtre de la 303
```

## Générer des lignes acid synthétiques

```bash
priism acid --out data/acid --count 1000 --workers 8
```

Chaque ligne donne un WAV stéréo 44,1 kHz et un JSON avec tous ses paramètres
(seed, motif, réglages du synthé et des effets), ce qui permet de la
régénérer à l'identique.

Le synthé (`src/priism/acid/`) suit le schéma d'une TB-303 : oscillateur scie
ou carré, filtre passe-bas résonant 4 pôles saturé, enveloppe de filtre,
accents et slides, motifs de 1 à 4 mesures tirés dans une gamme. Les effets
sont tirés au hasard : saturation, delay à feedback qui s'assombrit (style
dub), reverb, largeur stéréo. La cible « acid » garde ses effets, puisque dans
un vrai morceau les retours de delay de la 303 appartiennent à la 303.

## Séparer des morceaux

```bash
priism separate ~/Musique/dub --out out/stems --profile dub-acid-baseline
```

Chaque morceau donne un dossier avec un WAV par slot et un `manifest.json`.
Les morceaux déjà traités sont sautés (`--overwrite` pour les refaire).

Un profil (`src/priism/profiles/*.toml`) choisit le modèle et dit comment ses
stems remplissent les slots. Un slot peut être `"residual"` : il reçoit le mix
moins tous les autres slots, ce qui garantit que les slots se ré-additionnent
exactement au morceau d'origine.

| Profil | Modèle | Slots |
|---|---|---|
| `standard` | htdemucs_ft | drums / bass / other / vocals |
| `dub-acid-baseline` | BS-Roformer-SW | drums / bass / acid (= other) / rest |

`dub-acid-baseline` est la référence à battre : aucun modèle existant n'a de
stem acid, donc la ligne de 303 atterrit dans `other` ou dans `bass` selon son
registre.

## Tests

```bash
.venv/bin/python -m pytest
```
