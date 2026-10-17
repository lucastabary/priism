# Diagnostic à tête froide : Cleave et le séparateur du lab

*16 octobre 2026, branche `lab/cleave`. Bilan de tout ce que le lab a tenté pour obtenir un séparateur
fin, avant de décider si on lance un long entraînement. Il s'appuie sur la relecture des docs, du code et
des logs, et sur deux tests CPU faits pour ce diagnostic (section 3). Aucun GPU n'a été utilisé pour
l'écrire.*

---

## En bref

- **Verdict : arrêter Cleave comme projet d'entraînement**, au moins sous sa forme actuelle (un JEPA appris
  de zéro, puis un prior de diffusion, puis une boucle d'épluchage). Aucun gros entraînement ne se justifie
  aujourd'hui.
- **La phase 1 n'a jamais produit un Z utile.** Sur le protocole exact de `cleave-train`, la moyenne du
  log-mel sur le crop donne déjà `id_auc_kind` = **0,914** et des MFCC **0,939** [0,913 ; 0,960]. Le
  meilleur Z de six probes atteint 0,844, au pas 400 d'un run qui s'effondre ensuite. Le critère C2
  (« battre l'init aléatoire », 0,713) était trop bas. Le plancher MFCC « 0,816 » cité dans la doc venait
  d'un autre protocole (le probe B : fenêtres de 4 s, set de test), donc il n'était pas comparable.
- **Corriger l'entraînement ne suffirait pas.** Les six probes ont donné soit une dérive des cibles, soit
  un collapse, soit des cibles prévisibles sans Z (section 2). On pourrait sans doute faire vivre Z, avec un
  β mis à l'échelle de la cible : Z pourrait récupérer environ un cinquième de ce que P/L laissent (test CPU,
  section 3.2). Mais le Z qui fait ça est une **enveloppe spectrale moyenne de la piste**, exactement la
  baseline gratuite à 0,926. Tel qu'il est posé, l'objectif ne pousse pas Z au-delà.
- **Même si la phase 1 marchait, la suite de Cleave est hors budget.** Il faudrait ensuite un prior de
  diffusion par source sur de la vraie musique, puis une chaîne gloutonne sur 10 à 20 sources. Chacune de
  ces deux étapes est un problème de recherche ouvert, avec des coûts GPU bien au-delà des 25 EUR.
- **Ce qui marche dans le lab, c'est Facet** (SW + clustering d'identité) : **+0,88 dB** [+0,49 ; +1,31] sur
  SW, sur 20 morceaux MoisesDB de test. La marge mesurée est dans l'**assignation** : un oracle
  d'étiquetage sur la même grille de tokens fait **+1,8 dB** de plus que Facet. Les deux meilleures pistes
  attaquent cette marge directement, sans rien entraîner de zéro (section 6).

---

## 1. Ce que le lab a tenté, et ce que ça a donné

| Approche | La question | Le résultat | Statut |
|---|---|---|---|
| **Séparateur à attracteurs** (`docs/modele.md`) : cœur SW + 16 slots, PIT, entraîné sur données synthétiques | Séparer sans classes fixes en partant de SW | +2,5 dB par stem sur NI, contre +8,1 pour SW seul | Abandonné au profit des modèles pré-entraînés tels quels |
| **Registry, phases 0 à 2b** : tokens SW figés en cache, tête d'identité entraînée sur CPU | Regrouper les tokens par source sur tout le morceau | V-measure sur NI : 0,220 (tokens bruts) → 0,244 (tête synthétique). Sur MoisesDB : 0,237 → 0,288 (bloc 6 + FX 0,9). Significatif, 3 seeds | **Acquis** |
| **Leviers de données** (phases 4 à 6) | Qu'est-ce qui aide sur la vraie musique ? | Les chaînes FX aident, avec une dose-réponse. Le bloc 6 de SW aide (+0,015). Freesound nuit, les boucles réelles ne servent à rien, et 400 morceaux au lieu de 200 ne changent rien | Acquis : le synthétique avec FX suffit |
| **Facet** (step 3a) : SW 6 stems redécoupés par la tête d'identité | Du vrai SDR | SW 1,97 → Facet 2,85 dB. Avec le vrai nombre de sources : 3,00 (n.s.). Oracle d'étiquetage : 4,65, et oracle au bin près : +0,6 de plus | **Produit** (profil `facet`, pas encore poussé) |
| **Décodeur fin** (masques au bin près) | Lever la limite de la grille de tokens | +0,2 à +0,4 dB, sous le +0,5 visé | Mis de côté : ce n'est pas là que ça bloque |
| **Tête 3b** (+ 58 morceaux MoisesDB réels) | Améliorer l'assignation avec de la vraie musique | +0,30 [+0,16 ; +0,46] sur test-20, n.s. sur dev2 | Sous le +0,5 visé, recettes épuisées |
| **Cleave, phase 1a** | Un Z appris par JEPA sur le morceau entier, avec σ | 6 probes, aucun ne tient C1. Le meilleur Z est **sous** une moyenne du log-mel | Ce document |

Le motif est net. Ce qui a marché **réutilise un modèle pré-entraîné figé** (SW) et n'entraîne qu'une petite
tête, sur CPU ou presque. Ce qui a échoué entraîne une représentation, en partant de zéro ou en
fine-tunant, sur nos données. Avec environ 200 morceaux synthétiques et 58 réels, le lab n'a pas de quoi
apprendre une représentation audio meilleure que celles qui existent déjà.

---

## 2. Cleave phase 1 : ce que disent vraiment les probes

Six runs sur Chav, tous en phase 1a (k = 1, `data/reg-train-fx`, validation sur MoisesDB dev2, 133 pistes,
un seul seed sauf mention contraire). `pred` est la MSE sur des cibles normalisées par token : prédire la
moyenne donne ≈ 1.

| Run | Réglage | Ce qui se passe | Meilleur `id_auc_kind` (pas) | Fin |
|---|---|---|---|---|
| s0b | EMA + VICReg (défaut) | `target_rank` 70 → 240 ; `pred` remonte à 0,99 ; KL → 0 ; `z_std` 0,003 | 0,844 (400) | 0,69 |
| frozen | `--ema 1` | `pred` 0,02 ; `z_use` ≈ 0,0005 ; `target_rank` fixe à 70 | 0,794 (400) | 0,72, avec des oscillations de 0,62 à 0,79 |
| novic | sans VICReg | `target_rank` 70 → 7 ; `z_rank` 3 ; `z_use` monte à 0,37 | 0,795 (2400) | annulé |
| ctx-s0 | + branche I-JEPA | même dérive que s0b ; la perte `ctx` remonte de 0,23 à 0,93 | 0,838 (600) | annulé |
| ctxnv-s0 | ctx, sans VICReg | `target_rank` 70 → 27 et `z_rank` 4 au pas 800 : le collapse de novic | 0,731 (600) | annulé ; seed 1 non lancé |

### 2.1 Trois régimes, un seul problème de fond

- **Avec VICReg, les cibles dérivent vers du bruit.** Le VICReg exige un écart-type ≥ 1 par dimension et
  décorrèle les dimensions, sur tous les tokens de patch. La façon la moins chère d'obtenir beaucoup de
  variance décorrélée est d'encoder le détail propre à chaque patch, c'est-à-dire du bruit, que rien ne
  prédit. Même un predictor qui voit 75 % du chunk n'y arrive plus (ctx-s0 : 0,93). La branche de contexte
  n'y change rien : dans I-JEPA, rien ne pousse les *cibles* à être prévisibles, c'est l'asymétrie
  EMA / predictor qui le fait implicitement. Un VICReg de poids 1 contre une perte de prédiction de 0,3
  l'emporte.
- **Sans VICReg et sans contexte, les cibles s'effondrent.** C'est le collapse attendu quand le seul
  predictor ne lit que Z (novic). Z finit par « servir », mais pour prédire 7 directions.
- **Avec des cibles figées, Z ne vaut pas son prix.** Les tokens de l'init, une fois normalisés, sont
  dominés par un token moyen commun à tous les patchs : le prédire seul donne déjà une MSE de **0,040**
  (contre 1 pour prédire 0 ; mesuré sur dev2, `tokvar.py`). Le `pred` de 0,02-0,03 du run `frozen` est donc
  surtout ce token moyen, plus ce que la position et P/L expliquent. Ce qu'un Z *parfait* sur l'identité
  pourrait encore retirer est de l'ordre de 0,003 de MSE (section 3.2). Or avec β = 10⁻³, chaque nat de KL
  coûte 0,001 : Z ne peut se payer que ~3 nats, à peine plus de 4 bits pour décrire un instrument. Le KL le
  ramène vers le prior (σ → 0,95, `z_use` ≈ 0,0007). C'est un **posterior collapse** classique : le decoder
  (le predictor conditionné par P/L) est assez fort pour se passer du latent (Bowman et al. 2016 ; Lucas
  et al. 2019), aggravé par un **β calibré pour des cibles de variance 1 alors que la variance utile est
  ~25 fois plus petite**.

Le point commun est que **l'objectif ne paie presque rien pour l'identité.** Les cibles sont soit
imprévisibles (dérive), soit pauvres (collapse), soit expliquées sans Z (figées). Le dispositif ne laisse
pas de régime où le timbre d'un instrument pèse lourd dans la perte. La section 3.2 mesure combien il
*pourrait* peser.

### 2.2 La fenêtre du pas 400 n'est pas un signal

- s0b, frozen et ctx-s0 ont tous leur meilleur point vers le pas 400-600, au moment où les cibles sont
  encore celles de l'init (lisses, `target_rank` ≈ 70-90) et où le KL n'a pas encore écrasé Z. C'est un
  transitoire, pas un état que l'entraînement atteint.
- Le « meilleur » est le maximum de 10 à 17 évaluations d'une métrique qui bouge de 0,02 à 0,05 d'une
  évaluation à l'autre (médiane des |écarts|). C'est un biais de sélection, et un seul seed.
- Le critère d'arrêt (plateau de `id_auc_kind`) garde ce maximum transitoire comme `best.pt`.

### 2.3 Les critères comparaient à la mauvaise barre

C2 demandait de battre l'init aléatoire (0,713). La doc citait aussi le « plancher MFCC 0,816 » comme
objectif. Mais ce 0,816 vient du probe B : un autre protocole (fenêtres de 4 s, MoisesDB test, pondéré par
artiste). Sur les **mêmes paires** que `id_auc_kind` (crops de 17,6 s, pistes de dev2, seed 0), voici
les vraies barres, mesurées pour ce diagnostic (`idbase.py`, section 3.1) :

| Embedding | `id_auc_kind` | IC 95 % (bootstrap sur pistes) | `id_auc` |
|---|---|---|---|
| init aléatoire de Cleave | 0,713 | | 0,767 |
| meilleur Z de Cleave (s0b, pas 400) | 0,844 | | 0,876 |
| moyenne du log-mel | **0,914** | [0,880 ; 0,946] | 0,954 |
| moyenne + écart-type du log-mel | **0,926** | [0,894 ; 0,954] | 0,961 |
| MFCC 20, moyenne + écart-type | **0,939** | [0,913 ; 0,960] | 0,975 |

Un Z appris qui ne bat pas une moyenne de spectre ne porte rien de plus que l'enveloppe spectrale
moyenne. **Aucun probe n'a jamais approché la vraie barre.** Il faut aussi relativiser la métrique
elle-même : deux crops de la même piste MoisesDB partagent la chaîne d'enregistrement. Une moyenne de
spectre capte donc l'empreinte de la prise autant que l'instrument : c'est le recording fingerprint, déjà
soupçonné avec CLAP au probe B. Une AUC haute sur ce test ne prouve pas qu'on a l'identité utile pour
séparer.

### 2.4 Ce qui a été dit trop vite

- **Le smoke CPU du 15 octobre avait déjà donné `z_use` ≈ 0,001**, et la note disait « à surveiller ».
  C'était le symptôme principal, et il a été porté sur GPU sans être compris d'abord.
- **Cinq probes GPU en une journée, chacun corrigeant un symptôme du précédent**, sans reproduction CPU de
  la dynamique. Les probes 2 et 3 ont aussi été lancés sans ton accord. Chav est gratuit, mais on a appris
  peu par probe.
- **La hausse de `pred_err` avec k** (ρ 0,5-0,7) a été notée dans tous les runs. Tant que Z ne sert pas, elle
  mesure seulement que P/L sont plus durs à lire dans un mélange (section 2.8 de cleave.md) : ce n'est pas
  un résultat sur σ.

---

## 3. Les deux tests CPU de ce diagnostic

Scripts et résultats dans `scripts/cleave_diagnostic/` (`idbase.py`, `headroom.py`, `tokvar.py`, et leurs
JSON), sur CPU : `PYTHONPATH=src python scripts/cleave_diagnostic/idbase.py <dossier mdb-dev2>`. Données :
MoisesDB dev2 (23 morceaux). `headroom.py` prend environ une heure sur le laptop.

### 3.1 Les barres d'identité sous le protocole de `cleave-train`

Mêmes paires que l'évaluation de `cleave-train` (`id_pairs`, seed 0 : 133 pistes, deux crops de 8 chunks à
≥ 20 s d'écart). Cosinus entre embeddings standardisés, AUC « même piste contre autre piste du même type ».
L'IC vient de 500 rééchantillonnages des pistes. Le tableau est en 2.3.

### 3.2 La part de la cible que Z peut expliquer en plus de P/L

**La question :** une fois donnés la position du patch et P/L de ses frames, quelle part de la cible reste
à expliquer, et combien un Z *parfait* sur l'identité de la piste en récupère ?

**Le protocole :** 72 pistes de dev2 actives à ≥ 80 % (34 656 patchs). Z oracle = moyenne et écart-type
du log-mel d'un **autre** crop de la même piste : de l'identité, pas le contenu du crop. Un MLP
(2 × 512) est entraîné sur 70 % des morceaux et évalué en R² sur les autres, avec 3 splits et 3 seeds. Deux
cibles : le patch de log-mel brut, et les tokens de l'encodeur cible à l'init (les cibles du run `frozen`).

R² sur les morceaux tenus à part (moyenne ± écart-type sur 3 splits) :

| Entrées | log-mel brut | tokens de l'init |
|---|---|---|
| position seule | 0,362 ± 0,023 | 0,290 ± 0,025 |
| position + P/L | 0,679 ± 0,011 | 0,592 ± 0,013 |
| position + Z oracle | 0,611 ± 0,030 | 0,521 ± 0,027 |
| position + P/L + Z oracle | 0,764 ± 0,013 | 0,669 ± 0,009 |
| **gain de Z sur P/L** | **+0,085 ± 0,023** | **+0,077 ± 0,021** |
| part du résidu de P/L que Z récupère | 26 % ± 6 | 19 % ± 4 |

Le gain est positif sur les 3 splits pour les deux cibles (de +0,057 à +0,117). Trois splits, c'est peu
pour un IC ; l'écart-type sert seulement d'ordre de grandeur.

**Ce que ça montre :**
- **Il y a de la place pour l'identité dans la cible**, environ un cinquième de ce que P/L laissent. Le
  problème n'est donc pas que P/L disent tout, mais d'**échelle**. Sur les tokens de l'init, la variance
  autour du token moyen n'est que de 0,040 : ce cinquième vaut ≈ 0,003 de MSE, soit le prix de 3 nats de KL
  à β = 10⁻³ (section 2.1). Un β mis à l'échelle (ou des cibles standardisées par dimension) donnerait à Z
  une vraie raison d'exister.
- **Le meilleur Z pour cet objectif est une enveloppe spectrale moyenne.** L'oracle qui récupère ce
  cinquième n'est rien d'autre que la moyenne et l'écart-type du log-mel de la piste, c'est-à-dire la
  baseline de 3.1 (0,926). Si l'entraînement marchait, le Z de Cleave apprendrait au mieux à peu près ce
  descripteur, qu'on a déjà gratuitement. C'est l'argument le plus fort contre l'objectif de la phase 1 tel
  qu'il est : il ne pousse pas Z au-delà de ce qu'une moyenne de spectre sait déjà.

**Limites :** le MLP ne voit que le P/L de sa colonne, pas celui du chunk entier comme le predictor, et un
Z oracle plus riche récupérerait peut-être plus. La part de 19-26 % est une borne basse, mais l'ordre de
grandeur tient. Les 72 pistes viennent de MoisesDB dev2, pas des données d'entraînement synthétiques.

---

## 4. Les hypothèses de Cleave au-delà de la phase 1

Même avec une phase 1 qui marche, la chaîne reposerait sur quatre paris, chacun non testé et cher :

1. **σ croît avec le nombre de sources.** La doc l'écrit elle-même (2.8) : dans un VAE, σ mesure
   l'information que le decoder demande, pas l'ambiguïté. Un mélange en demande plus, donc σ pourrait
   *baisser*. Le contrôle hors distribution n'est pas fait. Ce pari est le cœur du critère de split.
2. **Un prior de diffusion « une source » sur de la vraie musique.** Multi-Source Diffusion (Mariani et
   al. 2023) tient sur Slakh (4 classes, 145 h de MIDI rendu) et reste loin des modèles discriminants. Le
   lab a 58 morceaux réels et un générateur sans voix ni guitares. Entraîner un tel prior est un projet de
   l'ordre de centaines d'heures de GPU.
3. **Une chaîne gloutonne sur 10 à 20 sources.** La séparation récursive (OR-PIT) tient sur 2 à 4
   locuteurs. Les erreurs se propagent, et la passe jointe finale est elle-même une diffusion multi-sources.
4. **Un coût à l'inférence compatible avec MyMixxx.** Une diffusion × sources × pas sur un morceau entier,
   sur le CPU du laptop, déjà à ~11× le temps réel avec Facet : ça ne passe pas.

Chaque étape vise un résultat (Z, puis σ, puis le prior) dont le lien avec le SDR n'arrive qu'à la toute
fin. C'est l'inverse de la règle que tu as posée le 14 octobre : des stems audibles et du SDR, pas des
métriques abstraites.

---

## 5. Le lab dans son ensemble

**Ce qui est montré** (plusieurs seeds, tests appariés, IC qui excluent 0) :
- Facet bat SW : +0,88 dB [+0,49 ; +1,31] sur 20 morceaux MoisesDB de test, et +0,91 sur 60 ; +1,6 dB sur
  les références qui partagent un stem SW.
- Le goulot de Facet est l'assignation, pas le comptage : oracle d'étiquetage − Facet avec le vrai nombre
  de sources = +1,65 dB [+1,33 ; +1,95]. La résolution de la grille coûte ~0,6 dB de plus (oracle_bin).
- Les leviers qui comptent pour la tête d'identité : le bloc 6 de SW, et des FX dans les données
  synthétiques.

**Ce qui est suggéré** (un seul seed, petit n, ou pas de test) :
- Que des données réelles en plus n'aident plus la tête (3b : +0,3 sur un set, n.s. sur l'autre).
- Que l'assignation pourrait gagner en étant entraînée sur l'objectif de séparation lui-même plutôt que
  sur l'identité (jamais testé).

**Ce qui reste de la spéculation :**
- Un Z appris par JEPA meilleur que les encodeurs existants : rien ne le soutient.
- σ comme détecteur du nombre de sources.
- Un séparateur génératif par épluchage sur de la vraie musique.

**Les leçons de méthode :**
- Comparer à la barre triviale *sous le même protocole*, avant de lancer, pas à un chiffre d'un autre
  probe.
- Reproduire sur CPU un symptôme vu au smoke avant de payer du GPU pour le voir en grand.
- Ne pas garder un `best.pt` sélectionné sur le maximum d'une métrique bruitée et d'un seul seed.

---

## 6. Ce que je propose, chiffré

Les options sont classées par SDR attendu par euro. Chacune a un critère go/no-go écrit à l'avance, mesuré
sur MoisesDB dev2 pour le réglage et sur test-20 pour la décision, en apparié par morceau, avec un IC
bootstrap et le d_z.

### Option A : entraîner l'assignation de Facet sur l'objectif de séparation (recommandée)

- **L'idée.** La tête actuelle apprend « même instrument ? » par contraste, puis on regroupe. On l'entraîne
  plutôt à reproduire **l'étiquetage oracle**, celui qui vaut +1,8 dB : quel token appartient à quelle
  référence, avec une perte invariante par permutation sur les tokens de chaque stem SW (deep clustering,
  Hershey et al. 2016, ou une perte de type PIT sur des affectations souples). Les labels oracle se
  calculent déjà dans `registry-separate-eval`.
- **Les données.** Le synthétique avec FX (illimité, labels exacts) + les 58 morceaux MoisesDB train, sur
  les tokens du cache `cache-b6`, qui existe déjà sur Chav et sur le volume du pod.
- **Le coût.** SW reste figé et le cache existe : entraînement de la tête sur CPU ou en quelques heures
  sur Chav, **0 EUR**.
- **Go/no-go :** ≥ +0,5 dB sur Facet (registry_k) sur test-20, IC qui exclut 0, et pas de perte sur les
  références non partagées. Sous +0,2, on arrête l'assignation et on passe à B.

### Option B : un meilleur point de départ que SW 6 stems (en parallèle, sans entraînement)

- **L'idée.** Facet redécoupe les stems de SW, donc tout ce que SW met dans le mauvais stem est perdu
  (l'oracle a la même borne). Des modèles pré-entraînés plus fins existent : séparation de batterie en
  éléments, lead/back vocals, guitare, piano, ou Mega 53 sur GPU. L'engine sait déjà les enchaîner
  (profils en arbre de pas soustractifs), et `eval-stems --cascade` existe sur la branche `eval-cascade`.
- **Le coût.** De l'inférence seule, sur Chav en job de fond : 0 EUR. Il faut seulement les checkpoints
  (licences à vérifier) et une heure de code d'évaluation.
- **Go/no-go :** une cascade bat SW + Facet sur test-20, en SDR et en temps de calcul acceptable sur le
  laptop (ou sur GPU seulement, si c'est ton choix produit).

### Option C : ce qui reste de Cleave, seulement si A et B plafonnent

- Garder l'idée utile de Cleave : un critère d'arrêt et de split pour décider combien de sources il y a
  dans un stem. Mais le tester sur les **embeddings figés** (bloc 6 de SW, ou la tête de Facet), par exemple
  avec un split bayésien à la X-means sur les tokens d'un stem. C'est sur CPU et ça s'évalue directement en
  SDR (registry vs registry_k).
- Ne reprendre un JEPA que si on lui donne d'abord une barre honnête (battre MFCC 0,939 sur ses propres
  paires) et une raison de penser qu'il ferait mieux que SW b6 ou MERT, entraînés sur bien plus de données.

### Le job de fond pour Chav

Le cache des tokens oracle pour l'option A, sur les 200 morceaux FX (`reg-train-fx`) et MoisesDB train, en
priorité « background » : utile quoi qu'on décide, et préemptible. Les tests de l'option B passent devant.

---

## Références

- Bowman et al. 2016, *Generating Sentences from a Continuous Space* (KL vanishing / posterior collapse) ;
  Lucas et al. 2019, *Don't Blame the ELBO!* (posterior collapse).
- Assran et al. 2023, *I-JEPA* ; Bardes et al. 2022, *VICReg*.
- Hershey et al. 2016, *Deep clustering* ; Yu et al. 2017, *Permutation invariant training*.
- Mariani et al. 2023, *Multi-Source Diffusion Models* ; Takahashi et al. 2019, *Recursive speech
  separation* (OR-PIT).

---

## 7. Suivi : l'option A lancée (16 octobre, soir)

**Le code.** `train-id --loss dc` (`registry/head.py`). C'est la même tête et le même cache que Facet, mais
la perte est un deep clustering à labels souples, sur tous les tokens audibles de deux chunks éloignés
d'un morceau. Un token y est pondéré par sa puissance à la puissance 0,5. `best.pt` garde la plus petite
perte DC de validation, calculée sur des chunks fixes. Le smoke CPU sur le vrai cache (`cache-b6`, 200 pas,
niveau track) donne une perte qui passe de 0,143 à 0,099, `val_dc` de 0,105 à 0,090, et `far_auc` qui monte
de 0,71 à 0,76, à 0,25 s par pas.

**Le protocole, écrit avant le lancement :**
- **Le bras** : `--loss dc --level track`, avec les données de la tête 3b (`p6-fx09` + `moises%train`,
  valid `p6-valid%dev` + `moises%dev2`), 5000 pas, plateau 6, seeds 0 à 2, sur Chav.
- **Les comparaisons**, appariées par morceau :
  - contre Facet (`fx09`, supcon, synthétique seul) : c'est le produit actuel ;
  - contre `fx09mdb` (supcon, mêmes données) : ça isole l'effet de la perte. Il reste une différence de
    niveau (track contre instrument).

  Les deux baselines existent déjà sur dev2 et sur test-20, à cut 0,4 et temp 0,05.
- **Le réglage** : la cut de DC est choisie sur dev2 parmi 0,2 à 0,6, en moyenne sur les seeds. Test-20 n'est
  évalué qu'à cette cut.
- **Go** : DC − Facet ≥ +0,5 dB en SDR moyen par morceau sur test-20 (moyenne des 3 seeds), avec un IC
  bootstrap qui exclut 0. **Stop** si le gain est sous +0,2 dB, ou si l'IC contient 0 : on n'ajuste pas la
  recette.

---

## 8. Les bras du diagnostic de Cleave sur Chav (16 octobre, soir)

Le modèle est petit (dim 128, 2 couches, z_dim 32), entraîné 1200 pas sur `reg-train-fx` et validé sur
MoisesDB dev2 (les 133 paires de 3.1). **Un seul seed : les seeds 1 et 2 sont en file de fond.** Toutes les
valeurs sont au pas 1200, avec le maximum entre parenthèses quand il est ailleurs.

| Bras | id_auc_kind | z_use_rel | KL (nats) | Lecture |
|---|---|---|---|---|
| frozen, β 10⁻³ | 0,755 (0,816) | 0,019 | 4 | le collapse de référence |
| frozen, β 0 | 0,779 (0,827) | 0,022 | 44 | Z stocke de l'info, sans gain d'identité ici (0,866 sur les stems NI du laptop) |
| mel, β 10⁻³ | 0,834 (0,846) | 0,000 | 2,6 | cibles stables, Z ignoré |
| mel, β 0 | 0,837 (0,854) | 0,004 | 48 | Z porte de l'info, mais le predictor s'en sert à peine |
| mel, β 0, hinge z_use | 0,864 | 0,056 (forcé) | 44 | monte encore |
| d2v + ctx, β 10⁻³ | 0,837 (0,860) | 0,018 | 6 | |
| **d2v + ctx, β 0** | **0,885**, monte encore | **0,053**, monte | 55 | le meilleur |
| résidu mel, β 10⁻³ | 0,873 | 0,000 | 2,3 | AUC sans usage de Z : le predictor du résidu échoue (≈ 1) |
| résidu d2v, β 10⁻³ | 0,830 | 0,000 | 1,2 | idem |

**Ce que ça suggère** (un seul seed, à confirmer) :
- **β = 0 est nécessaire** pour que Z porte de l'information (KL de 44 à 55 nats, contre 1 à 6). Mais ce
  n'est pas suffisant : le predictor doit aussi avoir une raison de le lire.
- **La meilleure combinaison est data2vec + branche de contexte + β = 0** : z_use_rel monte de 0 à 5 %, et
  `id_auc_kind` de 0,73 à 0,885, encore en hausse au pas 1200. Elle reste **sous la barre des MFCC (0,939)**.
- **Le résidu échoue tel quel.** Le predictor du résidu reste à ≈ 1 : Z n'explique rien du résidu au
  pas 1200. g(P, L) n'a pas fini d'apprendre (perte de 0,86 à 0,37), et le KL à β = 10⁻³ écrase Z. Il
  faudrait le retester à β = 0.
- **`id_auc_kind` ne suffit pas comme diagnostic.** Le résidu mel monte à 0,873 avec un Z que le predictor
  n'utilise pas : l'AUC vient ici de ce que le VICReg et le KL façonnent, pas de l'objectif de prédiction.
- **La sonde de fuite Z → statistiques de P/L** est déjà à R² 0,72 à l'init, et reste entre 0,6 et 0,8 :
  Z porte le registre et la loudness moyens partout. Elle ne discrimine pas les bras.

**La suite, si les seeds 1 et 2 confirment :** un run plus long du bras d2v + ctx + β = 0, au même petit
format, pour voir s'il dépasse 0,939. On ne passe au grand modèle que s'il la dépasse.

---

## 9. Deux derniers bras pour Cleave (17 octobre), critères écrits avant

Chav est en pause depuis le 16 à 21 h 53. Deux jobs sont en file, en priorité normale (devant le fond) :

- **`arm-d2v-ctx-b0-long`** (job `1be4`) : le meilleur bras de la section 8 (d2v + ctx, β 0), même config,
  seed 0, mais **8000 pas** au lieu de 1200. À 1200 pas, `id_auc_kind` montait encore, alors que le LR cosinus
  était déjà au plancher : la question est de savoir s'il plafonne ou s'il passe la barre.
- **`arm-resid-d2v-b0`** (job `17a6`) : le mode résidu de la section 8, retesté à β 0 (à β 10⁻³ le KL avait
  écrasé Z, donc l'échec n'était pas concluant).

**Critères (valeur au meilleur pas sur dev2, 133 pistes) :**
- **Go Cleave** : le bras long atteint `id_auc_kind` ≥ 0,939 (MFCC sur les mêmes paires) **et** `z_rank` ≥ 8
  (C1). Il faudra alors 2 seeds de plus avant de conclure.
- **Stop définitif** : `id_auc_kind` < 0,92 au pas 8000, ou `z_rank` < 8. On ne règle plus la recette ; Cleave
  ne garde que son critère d'arrêt (option C).
- Entre les deux : un seul bras de plus (le modèle à dim 256), pas davantage.
- **Résidu** : utile si `z_use_rel` > 0,05 avec `id_auc_kind` ≥ celui du bras d2v + ctx β 0 à 1200 pas (0,885).

En parallèle, le laptop régénère `reg-train-fx` (seed 7100000, FX_CHAIN 0,6, BUILDUP 0,5, FX_AUTO 0,5, sans IR)
dans `data/reg-train-fx`, pour pouvoir lancer ces bras sur CPU si Chav reste hors ligne.

### 9.1 Résultats (17 octobre, laptop CPU, seed 0)

Chav est resté hors ligne : les deux bras ont tourné sur le CPU du laptop (i5-1235U, ~0,7 s par pas), sur le
`reg-train-fx` régénéré en local (mêmes seeds et réglages ; le dev2 est identique : même `id_auc_kind` à
l'init, 0,7288). Les jobs Chav `1be4` et `17a6` sont annulés.

| Pas | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 |
|---|---|---|---|---|---|---|---|---|
| `id_auc_kind` | 0,884 | 0,884 | 0,922 | 0,926 | 0,931 | 0,936 | 0,939 | 0,940 (max 0,941) |
| `z_rank` | 4,8 | 4,5 | 5,6 | 6,2 | 6,5 | 7,0 | 7,4 | 7,6 |
| `z_use_rel` | 0,04 | 0,09 | 0,11 | 0,13 | 0,16 | 0,19 | 0,21 | 0,19 |

- **Bras long** : `id_auc_kind` atteint **0,941**, c'est-à-dire le niveau des MFCC (0,939), sans le dépasser.
  `z_rank` monte jusqu'à 7,6, sous le seuil de 8. Le predictor utilise enfin Z (`z_use_rel` ≈ 0,2).
- **Résidu à β 0** : 0,824 au pas 1200 (contre 0,885 pour le bras d2v + ctx), `z_use_rel` ≈ 0 : échec.

**Verdict selon les critères écrits avant : stop.** `z_rank` < 8, et même en ignorant ce seuil, Z ne fait
qu'égaler un descripteur gratuit. Un seul seed, mais la décision n'en dépend pas : un deuxième seed
favorable donnerait au mieux une égalité avec les MFCC. Cleave n'est plus entraîné. Le GPU va à l'option B
(cascade), dont les évaluations `e430` (dev2) et `33cd` (test-20) sont en file sur Chav.

### 9.2 Le facteur P ne nommait pas les notes (17 octobre)

**L'hypothèse de Lucas.** P donnait les 4 pics de saillance harmonique les plus forts. Sur une note seule,
ce sont surtout ses harmoniques (octave, quinte, sous-octave), et leurs hauteurs relatives disent l'équilibre
des partiels (les pics à 2 f0 et f0/2 sont faits des harmoniques paires). P livrait donc au predictor une
partie du profil harmonique, que Z devait porter.

**Le banc contrôlé** (`scratchpad p_synth_bp.py`, à recopier dans le repo si on le réutilise) : notes tenues et
accords (note seule, quinte, majeur, mineur, maj7) rendus avec les patchs du générateur (7 types, 12 tirages
par case, sans FX, `sub` = 0 et pas de FM de ratio < 1 pour que f0/2 ne soit pas physiquement présent).
Trames de 0,1 à 0,6 s après l'attaque.

| Détecteur | précision | rappel | trames exactes | note seule : précision / rappel / exactes |
|---|---|---|---|---|
| saillance top-4 (P actuel) | 0,29 | 0,44 | 0,00 | 0,21 / 0,83 / 0,00 |
| annulation itérative (`cancel`) | 0,47 | 0,36 | 0,21 | 0,60 / 0,74 / 0,60 |
| **Basic Pitch, seuil 0,3** | **0,87** | **0,85** | **0,61** | 0,98 / 0,96 / 0,93 |
| Basic Pitch, seuil 0,5 | 0,94 | 0,61 | 0,56 | 1,00 / 0,94 / 0,94 |

Le P actuel ne sort jamais le bon ensemble de notes ; les graves (sub, basse) et le sinus sont les pires, car
les bins de 21,5 Hz ne séparent pas les demi-tons graves. Basic Pitch (Bittner et al. 2022, ONNX via
onnxruntime, sans TensorFlow) est juste à 93 % sur une note seule et reste correct sur les accords.

**Le code.** `--p-mode salience|binary|cancel|bp` (`cleave/features.py`, `cleave/pitch_bp.py`) ; `bp` donne des
notes 0/1 au seuil 0,3, à la même cadence de trames que le front-end (86,13 par seconde). Mon wrapper colle
à `basic_pitch.run_inference` (accord binaire 100 %, écart max 0,003, dû au rééchantillonneur). Coût sur
le CPU du laptop : 1,4 s par batch de 4 × 35,7 s, donc environ 2,2 s par pas d'entraînement.

**Le bras** : `cpu-arm-d2v-ctx-b0-long-bp`, identique au bras long de 9.1 (même seed, mêmes données, 8000 pas),
seul P change. **Critère écrit avant** : `id_auc_kind` ≥ 0,951 (bras de 9.1 + 0,01) et `z_rank` ≥ 8 → 2 seeds
de plus ; sinon P n'était pas le goulot, et le stop de 9.1 tient.
