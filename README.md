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

## Générateur v2 : morceaux synthétiques, une piste par source

Depuis octobre 2026, Priism vise une séparation **par identité de source**, sans
classes imposées : deux 303 qui jouent en même temps doivent sortir sur deux
pistes. `priism gen` fabrique les données pour ça : des extraits arrangés
(techno, acid, dub techno, minimal, house, electro, breakbeat, dnb, dub,
steppers, dubstep) de 2 à 16 sources, batterie séparée élément par élément,
effets gardés dans la piste de leur instrument, et un master (compresseur,
limiteur) appliqué de façon à ce que **la somme des pistes soit exactement le
mix**. Environ 30 % des morceaux contiennent un « jumeau » : le même instrument
qui joue une autre partie (303 rythmique + 303 mélodique, deux lignes de hats…).
Environ 6 % contiennent une paire **ambiguë** (même patch, même registre, une
même ligne répartie note à note entre deux pistes) : les deux pistes partagent
un `merge_group`, la perte d'entraînement acceptera de les sortir ensemble. Un
kick, une snare ou un clap peut être une **couche** de deux sons déclenchés
ensemble : c'est une seule source. `priism.gen.augment` dégrade le mix seul
(MP3, AAC, Opus) à l'entraînement, les cibles restent propres.

```bash
priism gen --out data/songs --count 1000 --workers 8          # genres au hasard
priism gen --out data/songs --count 50 --genre dnb            # un genre précis
priism gen --genre list
```

Chaque morceau : `song_<graine>/mix.flac`, `sources/NN_<type>.flac` et
`meta.json` (genre, tempo, harmonie, structure, et pour chaque source son type,
son rôle, ses paramètres, ses mesures actives, ses effets, son jumeau
éventuel). Tout se régénère depuis la graine.

## Générer des lignes acid synthétiques

```bash
priism acid --out data/acid --count 1000 --workers 8
```

Chaque ligne donne un FLAC stéréo 44,1 kHz et un JSON avec tous ses paramètres
(seed, motif, réglages du synthé et des effets), ce qui permet de la
régénérer à l'identique.

Le synthé (`src/priism/acid/`) suit le schéma d'une TB-303 : oscillateur scie
ou carré, filtre passe-bas résonant 4 pôles saturé, enveloppe de filtre,
accents et slides, motifs de 1 à 4 mesures tirés dans une gamme. Les effets
sont tirés au hasard : saturation, delay à feedback qui s'assombrit (style
dub), reverb, largeur stéréo. La cible « acid » garde ses effets, puisque dans
un vrai morceau les retours de delay de la 303 appartiennent à la 303.

## Autres stems : sources synthétiques et entraînement modulaire

L'acid n'est qu'une source parmi d'autres. `priism synth list` montre les
générateurs (aujourd'hui `acid` et `skank`, le contretemps dub/reggae), et
`priism synth-mix` pose n'importe quelle combinaison de couches sur de vrais
morceaux, cibles ou distracteurs :

```bash
priism synth skank --out data/skank --count 1000 --workers 8
priism synth-mix data/slots_musdb_train --slots drums bass skank rest \
  --layer skank=data/skank --layer rest=data/acid:0.3:-20:-6 --out data/train_skank
```

La conception (une source + un jeu de données + un adaptateur par stem, puis
des profils qui choisissent 4 slots) est dans
[docs/modular-training.md](docs/modular-training.md).

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

## Entraînement sur un pod GPU (RunPod)

Le pod fait tourner une file de jobs (`pod/jobs/*.sh`, exécutés dans l'ordre
des noms) pour que la carte enchaîne sans attendre personne. Deux voies
tournent en parallèle : un job marqué `# lane: cpu` (préparation de données)
passe dans la voie CPU, les autres dans la voie GPU. Une ligne
`# after: 0030` fait attendre un job jusqu'à la fin du job 0030.

1. Créer le pod avec comme commande de démarrage
   `bash -c "curl -fsSL https://raw.githubusercontent.com/lucastabary/priism/main/pod/setup.sh | bash"`,
   la variable `PRIISM_WORKER_TOKEN` (donnée par `priism pod token`) et le
   port HTTP 8000 exposé.
2. Suivre et piloter depuis n'importe où :
   `priism pod status --url https://<pod>-8000.proxy.runpod.net`, puis
   `log`, `add`, `cancel`.
3. Tout vit sous `/workspace/priism` : `/workspace` est le volume réseau
   RunPod de Lucas, partagé avec d'autres projets (ne jamais toucher aux
   autres dossiers). Le volume survit à l'arrêt du pod ; les résultats se
   rapatrient aussi par `priism pod pull runs --url ... --dest <dossier>`.

Jobs actuels :
- 0200 et 0210 (CPU) rendent les chansons du générateur v2 : 200 de
  validation et 1500 d'entraînement, 30 s chacune (~35 Go en tout).
- 0300 (GPU) : 500 pas du séparateur à attracteurs sur le cœur
  BS-Roformer-SW pré-entraîné, pour vérifier la mémoire et le début
  d'apprentissage ; 0310 enchaîne le long entraînement (reprend depuis
  `last.pt` si le pod a été arrêté).
- 0230 (CPU, après les chansons synthétiques) récupère les genres
  électroniques et dub de FMA (plafonné à 60 Go, arrêt s'il reste moins de
  15 Go sur le volume).

Les jobs du premier plan (4 stems fixes, fine-tune acid, spécialistes LoRA)
sont archivés dans `pod/jobs-v1/` et ne sont plus mis en file.
