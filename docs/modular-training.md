# Entraînement modulaire par stem

But : pouvoir ajouter un type de stem (skank dub, sirène, mélodica, stabs rave…)
et l'entraîner à part, sans refaire ni abîmer ce qui marche déjà (la 303).

## Les quatre briques

Une spécialisation, c'est une **source**, un **jeu de données**, un **modèle** et
une place dans des **profils**. Chaque brique est indépendante.

### 1. Source : générer le stem isolé

`priism.sources` liste les générateurs synthétiques. Ajouter un type de stem,
c'est écrire un module avec trois fonctions et l'inscrire dans `SOURCES` :

| Fonction | Rôle |
|---|---|
| `sample_params(seed, duration_s, sample_rate)` | tire tous les réglages d'un exemple (objet avec `to_dict()`) |
| `render(params)` | rend l'audio stéréo `(n, 2)` float32 |
| `params_from_dict(d)` | relit le JSON, pour régénérer à l'identique |

```bash
priism synth list
priism synth skank --out data/skank_train --count 6000 --duration 16 --workers 16
```

Sources disponibles : `acid` (TB-303) et `skank` (accords à contretemps :
guitare étouffée, bubble d'orgue, piano, stab dub techno ; spring reverb et
« throws » d'écho). Les effets font partie de la cible, comme pour la 303 :
couper le stem coupe aussi ses échos.

Pour un stem qu'on ne sait pas synthétiser de façon crédible (voix, cuivres
réels), la source peut être un dossier d'enregistrements isolés au lieu d'un
générateur : `synth-mix` ne lit que des FLAC/WAV.

### 2. Données : poser les couches sur de vrais morceaux

`synth-mix` généralise `acid-mix` : n'importe quel nombre de couches, chacune
dirigée vers un slot.

```bash
priism synth-mix data/slots_musdb_train --slots drums bass skank rest \
  --layer skank=data/skank_train:0.8 \
  --layer rest=data/acid_train:0.3:-20:-6 \
  --out data/train_skank --count 6000
```

Une couche vers un **nouveau** slot (`skank`) est la cible. Une couche vers un
slot **existant** (`rest`) est un distracteur : ici des lignes de 303 que le
modèle skank doit apprendre à laisser dans `rest`. Croiser les sources de
cette façon (la 303 en distracteur du skank, le skank en distracteur de la
303) apprend à chaque spécialiste à discriminer ce qui ressemble à sa cible.

Limite connue : si un morceau de fond contient déjà un vrai skank, il est
étiqueté `rest`, ce qui contredit la cible. MUSDB en contient peu ; pour des
fonds dub il faudra filtrer.

### 3. Modèle : spécialiste complet ou adaptateur

Deux façons d'entraîner une spécialisation, toutes deux possibles avec
`train-init` et MSST :

**A. Spécialiste complet** (ce que fait `acid-v1`) : BS-Roformer-SW entier
réentraîné avec 4 têtes = les 4 slots. Meilleure qualité possible, mais un
checkpoint complet par spécialisation, et de l'oubli sur les stems qu'il
n'apprend pas (au sweep, drums est passé de 10,1 à 7,6 dB de SDR).

**B. Tronc gelé + adaptateur** (proposé par défaut pour les nouveaux stems) :
BS-Roformer-SW reste tel quel ; la spécialisation n'apprend qu'un LoRA sur les
couches d'attention et sa propre tête (mask estimator copié de `other`).
Quelques Mo par spécialisation, aucun oubli du modèle de base (il n'est pas
modifié), moins de mémoire GPU, et chaque spécialiste s'entraîne seul.
MSST le gère (`--train_lora_peft`, section `lora` de la config) :

```bash
priism train-init --config BS-Roformer-SW.yaml --ckpt BS-Roformer-SW.ckpt \
  --map skank=other rest=other --out runs/skank-v1 \
  --overrides '{"lora": {"r": 16, "lora_alpha": 32, "target_modules": ["to_qkv", "to_gates", "to_out.0"],
                          "modules_to_save": ["mask_estimators"]}}'
# puis train.py ... --train_lora_peft
```

Un spécialiste B n'a que deux têtes, `<cible>` et `rest` : il apprend
seulement à extraire sa source du mix. Ses données se font avec
`--slots skank rest --fold rest`, qui verse drums et bass du fond dans `rest`. Batterie et basse viennent du modèle
de base. Les noms de modules LoRA sont à valider au premier essai sur le pod.

Les adaptateurs ne sont **jamais empilés ni fusionnés** : chacun a été
entraîné seul sur le tronc d'origine, donc on fait une passe par spécialiste
(tronc + son LoRA + sa tête). Additionner des LoRA entraînés séparément les
ferait interférer ; une passe de plus par stem ne coûte que du temps de
calcul, ce qui ne gêne pas un traitement par lots hors ligne.

Une tête seule (tronc entièrement gelé) est le niveau le moins cher, mais
probablement insuffisant : la tête n'est qu'un petit réseau par bande de
fréquences qui lit les représentations du tronc, et celui-ci n'a jamais eu à
distinguer une 303 ou un skank des autres synthés rangés dans `other`. On
mesurera les trois niveaux sur la 303 avec les mêmes données : tête seule,
LoRA + tête, fine-tune complet (`acid-v1`).

Pour trancher entre A et B, on entraînera la 303 aussi en mode B, sur les
mêmes données qu'`acid-v1`, et on compare les SDR.

### 4. Composition : des stems aux 4 slots de Mixxx

La séparation peut produire plus de 4 stems ; c'est le profil qui choisit les
4 slots, ce qui garde la limite de Mixxx en bout de chaîne :

1. le modèle de base donne drums / bass / other / vocals… ;
2. chaque spécialiste demandé (acid, skank…) extrait sa source du mix ;
3. le profil range ces stems dans 4 slots, le slot `residual` garantissant
   que la somme redonne exactement le morceau.

| Profil | Slots |
|---|---|
| dub | drums / bass / skank / rest |
| acid | drums / bass / acid / rest |
| dub acid | drums / bass / acid / rest (skank dans rest), ou acid + skank réunis en « lead » |

Comme tous les stems sont gardés en WAV, on peut changer de profil (donc de
répartition en 4 slots) sans refaire la séparation. Quand deux spécialistes
réclament la même énergie, on normalise leurs masques pour que leur somme ne
dépasse pas le mix. Cette étape d'inférence s'écrira quand un spécialiste B
aura été entraîné ; l'empaquetage Mixxx reste hors du champ actuel.

## Ajouter une spécialisation, en pratique

1. Écrire la source (`src/priism/<stem>/`) et l'inscrire dans `SOURCES`, avec
   ses tests.
2. Générer train et validation (`priism synth`, seeds disjointes).
3. `priism synth-mix` avec la cible et un ou deux distracteurs.
4. `priism train-init --map <stem>=other rest=other` (mode B) puis MSST.
5. Évaluer : SDR sur la validation synthétique, et à l'oreille sur de vrais
   morceaux du genre.

Sur le pod, cela fait quatre jobs, sur le modèle de 0010 à 0100 : sources et
mix en voie CPU, entraînement en voie GPU avec `# after:`.
