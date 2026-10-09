# Plan v2 : découvrir puis extraire

*Note de conception, 9 octobre 2026. Elle remplace la file « jumeaux » et le séparateur à 16 emplacements
(docs/modele.md) comme direction principale. Rien n'est encore codé ; chaque phase a un critère de passage
mesuré sur de la vraie musique.*

---

## 0. En bref

1. **Le problème n'est pas le mécanisme, c'est le domaine.** Sur les stems NI, BS-Roformer-SW tel quel gagne
   +8,1 dB par stem ; après notre fine-tune sur morceaux synthétiques, +2,5 dB. Tant que le modèle perd
   5,6 dB sur du vrai son, optimiser les jumeaux sur une validation synthétique ne sert à rien.
2. **Séparer « quelles sources existent » de « extraire cette source ».** La découverte se fait sur le
   **morceau entier**, en regroupant des vecteurs d'identité produits par le tronc pré-entraîné. Le
   regroupement donne un **arbre** (batterie → kick, hats… ; synthés → lead, pad…). L'extraction est un
   petit réseau conditionné par une requête, qui sort une source ou un groupe entier selon la requête.
   Le nombre de pistes devient un réglage (coupe de l'arbre), pas une sortie fragile du réseau.
3. **Réutiliser au maximum, entraîner peu.** Le tronc BS-RoFormer pré-entraîné reste gelé et partagé, et
   seules ses 2 à 4 dernières couches sont recopiées et conditionnées. La tête de masque est recopiée elle
   aussi. On entraîne donc quelques dizaines de millions de paramètres au plus.
   - Ce tronc peut être celui de SW, ou celui de **MVSep Mega 53 stems** (MSST v1.0.21, avril 2026). Mega 53
     est de la même famille que SW (BS-RoFormer) mais a appris à distinguer 53 stems : kick, snare, hh, toms,
     synth, keys, organ, lead et back vocal…
   - Une **perte d'ancrage** impose qu'une requête « groupe batterie » redonne la batterie de l'enseignant :
     on ne peut pas tomber sous lui au niveau grossier.
4. **Une seule perte pour toutes les granularités de vérité.** Synthétique (fin), MoisesDB / MedleyDB /
   Slakh (instrument), MUSDB / NI (4 stems), pseudo-stems d'un enseignant (mixte) et MixIT (2 mélanges) sont
   tous des **arbres partiels**. La perte ne contraint que ce que la vérité sait.
5. **Données : arrangement synthétique, timbres réels.** Le générateur garde son moteur d'arrangement mais
   joue de vrais échantillons, des synthés VST et de vrais stems et boucles. MixIT pur est écarté comme
   signal principal ; il reste un cas particulier gratuit de la perte (section 6).
6. **Coût visé : 150 à 250 h de GPU au total** (RTX 4090 ou L40S), soit 60 à 200 $, réparties en
   5 phases. Chaque phase a un critère de passage sur de la vraie musique.
7. **Ce qui est nouveau.** La littérature sur « N instruments quelconques dans un morceau » est bien mince.
   Le travail le plus proche, MuS3D (Aalto, septembre 2026), découvre aussi des requêtes puis sépare
   requête par requête, mais sur des segments de 10 s, avec une découverte générative séquentielle et sans
   hiérarchie. Notre apport, détaillé en section 2.3 :
   - découverte sur le **morceau entier** ;
   - **arbre** de sources ;
   - perte à **labels partiels** ;
   - **ancrage** sur un tronc gelé ;
   - données **électroniques** à timbres réels.

---

## 1. Diagnostic de l'existant

Ce que le repo a appris (chiffres tirés des commits et des commentaires des jobs) :

| Constat | Chiffre | Conséquence |
|---|---|---|
| Le fine-tune synthétique fait oublier le vrai son | NI groupé : SW **+8,1 dB**, stage B **+2,5 dB** | l'écart de domaine domine tout le reste |
| La séparation plafonne sur 2 à 16 sources | ~2,6 dB (0312), 8,9 dB en validation synthétique (stage D) | le modèle apprend le générateur |
| Les requêtes fixes se spécialisent par type | kick dans l'emplacement 9 pour 93 % des morceaux, ~7 emplacements sur 16 jamais utilisés | DETR rejoue des classes fixes déguisées |
| Les jumeaux restent fusionnés malgré 6 mécanismes | 1,9 à 3 dB ; v2 −0,09 dB ; facteurs +0,23 dB ; tête de notes top-1 à 0,08 | problème mal posé, attaqué trop tôt |
| L'attention par slots casse l'acquis | v3 à froid : 8,9 → −0,8 dB au pas 0, 0,5 dB au pas 1000 | chaque mécanisme neuf repart de loin |
| Mémoire | 2 extraits × 16 emplacements × carte pleine résolution → lot de 2 | entraînement lent et bruité |
| Pas de cohérence sur un morceau entier | l'emplacement 3 n'est pas la même source d'un extrait à l'autre ; `split.py` recolle a posteriori | défaut structurel |

Ce qui rend le projet brouillon tient moins au code qu'à la **méthode** :
- les décisions se prennent sur des sondes de 1000 pas, avec des seuils de ±0,3 dB, sur une validation
  **synthétique** ;
- chaque idée devient un drapeau de plus (v1/v2/v3, facteurs, JEPA, mix-pitch, warm/cold…), sans qu'aucune ne
  s'attaque à l'écart de 5,6 dB ;
- il n'y a pas de banc d'essai réel **fin** (instrument par instrument) qui serve de boussole. NI ne compte
  que 4 groupes, et le regroupement par oracle y est indulgent.

Ce qu'on garde : le générateur et son moteur d'arrangement, le mastering additif, le flux de morceaux
(`stream.py`), l'enseignant SW (`distill.py`), `eval_stems.py`, `listen`, `tempo_key.py`, l'outillage du pod
(file de jobs, garde, sondes).

---

## 2. Reformuler le problème

### 2.1 Une « source » est un nœud d'arbre, pas une étiquette

« Une piste par instrument » n'a pas de définition unique. Un kick fait de deux couches, c'est une source ou
deux ? Et deux lignes de hats ? Une 303 et son delay ? La vérité dépend du producteur : MUSDB s'arrête à
4 stems, MoisesDB à environ 11 catégories puis l'instrument, notre générateur descend à l'élément de batterie.
La structure naturelle est un **arbre** :

```mermaid
flowchart TB
    mix["mix"] --> dr["batterie"]
    mix --> ba["basses"]
    mix --> sy["synthés"]
    mix --> vo["voix"]
    dr --> k["kick"]
    dr --> h["hats"]
    dr --> s["snare / clap"]
    h --> h1["hat fermé"]
    h --> h2["hat ouvert"]
    sy --> a1["303"]
    sy --> p["pad"]
    sy --> l["lead"]
```

Conséquences :
- **la sortie du système est un arbre**. Le DJ choisit sa profondeur : 4 pistes pour Mixxx, 12 pour un
  remix ;
- **chaque jeu de données donne un arbre partiel**. MUSDB connaît 4 feuilles, sans rien en dessous ; le
  synthétique connaît tout. La perte ne doit punir que ce que la vérité affirme ;
- **les jumeaux identiques** (même timbre, parties entrelacées) sont une feuille qu'on ne descend qu'à la
  demande (phase 5), pas un objectif de base.

### 2.2 Ce qu'on mesure

Deux défauts opposés existent : **sur-séparer** (une source coupée en trois) et **sous-séparer** (trois
sources dans une piste). Un seul SNR ne les distingue pas. Pour chaque niveau d'arbre où une vérité existe,
on mesure donc :

| Mesure | Définition | Défaut qu'elle voit |
|---|---|---|
| **SNR strict** | appariement 1-1 (hongrois) sorties ↔ références | sous-séparation |
| **SNR groupé** | chaque sortie rejoint la référence qu'elle explique le mieux (déjà dans `eval_stems.py`) | qualité du son, indulgent |
| **Pureté** | part de l'énergie de chaque sortie qui vient de sa référence majoritaire (pondéré énergie) | sous-séparation |
| **Complétude** | part de l'énergie de chaque référence qui tombe dans sa sortie majoritaire | sur-séparation |
| **Erreur de comptage** | \|N̂ − N\| au niveau considéré | les deux |

La pureté et la complétude se calculent dans le domaine STFT : la contribution de la référence j à la
sortie k vaut Σ_tf |Ŝ_k| · m_j, où m_j = |S_j|² / Σ_i |S_i|². Le calcul est bon marché et ne demande aucun
appariement.

### 2.3 Ce qui existe, et ce qu'on y ajoute

La revue de littérature (octobre 2026, références en fin de note) confirme que le problème exact est peu
traité. Voici les travaux les plus proches.

| Travail | Mécanisme | Ce qu'on en prend | Limite pour nous |
|---|---|---|---|
| **MuS3D** (Kallinen et al., arXiv 2609.30912, sept. 2026) | découverte itérative de requêtes (DiT à *flow matching* dans l'espace PaSST), puis séparateur conditionné par requête | la décomposition « découvrir puis extraire » ; ordre de grandeur sur MoisesDB au niveau instrument : **6,8 dB à l'aveugle contre 7,5 dB avec requêtes oracle**, F1 de découverte 0,74 | segments de 10 s, pas de cohérence sur le morceau ; découverte générative une requête après l'autre (les auteurs signalent des sources manquées et un biais d'exposition) ; pas d'arbre |
| **SepTDA** (Lee et al., ICASSP 2024) | requêtes apprises → décodeur Transformer → attracteurs + probabilité d'existence + FiLM | c'est **notre modèle actuel**, en parole | le comptage se dégrade quand N grandit (90 % à 4 locuteurs, 83 % à 5) : à 16 sources corrélées, on peut s'attendre à pire |
| **Banquet** (Watcharasupat et Lerch, ISMIR 2024) | un seul décodeur conditionné par FiLM sur une requête (exemple audio → PaSST) | la tête unique conditionnée | instable (effondrement vers le silence), requête fournie par l'utilisateur |
| **Requêtes hyperellipsoïdales** (Watcharasupat et Lerch, ICASSP 2026) | requête = région de l'espace d'embedding, dont la taille règle la granularité | le **régulariseur d'appariement de niveau**, contre l'effondrement silencieux | pas de découverte automatique |
| **TUSS** (Saijo et al., ICASSP 2025) | jetons de requête préfixés et traités par les mêmes couches que le mix | variante de conditionnement : les requêtes se voient entre elles | AGPL |
| **Séparation hiérarchique** (Manilow et al., ISMIR 2020) | masques à 3 niveaux d'un arbre d'instruments, contrainte M_parent ≥ M_enfant | la contrainte de monotonie dans l'arbre | **laisse explicitement de côté deux instances d'un même type**, notre cas des deux synthés |
| **EEND-VC, Wavesplit** (diarisation) | décisions locales par segment, puis regroupement global des embeddings sur tout l'enregistrement | le principe de la **découverte au niveau du morceau** | parole |
| **Deep clustering, DANet** (2016-2017) | un embedding par bin T-F, k-means | la perte d'affinité | masque dur par bin, K requis, re-clustering par segment : défauts qu'on évite en ne s'en servant **que pour découvrir** |
| **SAM Audio** (Meta, déc. 2025) | génératif (*flow matching* sur latents DAC-VAE), requêtes texte | référence d'évaluation, enseignant ponctuel (FX) | SDR faible sur MUSDB (other −1,7 dB selon une évaluation indépendante), ~44 s par minute et par stem sur A100, ne se ressomme pas au mix |
| **MixIT pour la musique** (arXiv 2505.07631) | MixIT sur mélanges de morceaux | — | **sur-sépare**, utile au mieux en pré-entraînement |

Ce que ce plan ajoute, sans équivalent publié à notre connaissance :
1. **Découverte au niveau du morceau, par regroupement d'embeddings de jetons d'un tronc pré-entraîné.** On
   obtient la cohérence de bout en bout et un dendrogramme qui *est* l'arbre des sources.
2. **Un extracteur pour toutes les granularités.** Une requête de feuille sort un instrument, une requête de
   nœud sort un groupe. L'export 4 pistes et l'export 12 pistes sont la même opération.
3. **Une perte à labels partiels hiérarchiques.** Elle entraîne avec MUSDB, MoisesDB, Slakh, le synthétique,
   les pseudo-stems d'un enseignant et MixIT sans changer de code.
4. **L'ancrage sur un tronc gelé.** On hérite de la qualité des modèles à stems fixes au niveau grossier, ce
   qui règle l'oubli constaté, sans adopter leurs classes.
5. **Musique électronique** : générateur à timbres réels (échantillons, synthés VST, boucles) et évaluation sur
   le domaine cible.

---

## 3. Architecture : découvrir puis extraire

```mermaid
flowchart TB
    mix["Morceau entier (stéréo)"] --> trunk["Tronc gelé (SW ou Mega 53)<br/>STFT, band split, couches 1…12−L<br/>(une passe par morceau)"]
    trunk --> H["Représentation partagée h(t, b)"]
    H --> E["Tête d'identité<br/>e(t, b) ∈ R⁶⁴, normé"]
    E --> D["Découverte (morceau entier)<br/>micro-clusters → regroupement agglomératif<br/>→ dendrogramme"]
    D -->|"coupe = granularité"| Q["K requêtes<br/>centroïde q_k + carte a posteriori π_k(t, b)"]
    H --> X
    Q --> X["Extracteur conditionné<br/>copie des L dernières couches du tronc + FiLM(q_k) + π_k<br/>+ copie de la tête de masque"]
    X --> W["Filtrage de Wiener joint<br/>(les K masques se partagent le mix)"]
    W --> out["K pistes + reste = mix − Σ<br/>(+ l'arbre, pour regrouper à volonté)"]
```

### 3.1 Tronc partagé (pré-entraîné, gelé)

On garde les étapes d'un BS-RoFormer pré-entraîné jusqu'à la couche 12−L (L = 2 à 4) : STFT, band split,
puis les blocs d'attention alternée temps / bandes. On réutilise `MsstAttractorSeparator._features` tel
quel. Le tronc tourne **une fois par extrait**, sans gradient, en bf16, et toutes les requêtes le partagent.
C'est ce qui rend l'entraînement bon marché : la partie lourde ne garde aucune activation.

**Quel tronc ?** La phase 1 compare trois candidats de la même famille : BS-RoFormer, dim 256,
12 couches, 62 bandes. C'est ce que disent les fiches des modèles ; on le vérifiera au chargement. SW pèse
environ 175 M de paramètres, d'après la taille du fichier. Les trois devraient se charger par
`load_msst_roformer`.

| Tronc | A appris à distinguer | Pour nous |
|---|---|---|
| **BS-Roformer-SW** (actuel) | 6 stems : bass, drums, other, vocals, guitar, piano | les synthés sont tous dans « other » |
| **MVSep Mega 53** (MSST v1.0.21) | 53 stems hiérarchiques : drums ⊃ kick, snare, toms, hh ; vocal ⊃ lead, back ; synth, keys, organ, percussion, strings… | le plus riche en identité ; même code |
| **X-LANCE MSR 2025** (licence MIT) | têtes spécialisées *synth*, *perc*, *orch*, fine-tunées depuis SW | le plus orienté synthés |

Le pari : le tronc de Mega 53 contient déjà, gratuitement, une bonne partie de la notion d'instrument.

Si la phase 1 montre que les représentations gelées ne distinguent pas deux synthés, on ajoute un LoRA
(r = 16) sur les couches du tronc. On ne fait jamais de fine-tune complet : c'est lui qui a fait perdre
les 5,6 dB.

### 3.2 Tête d'identité

C'est un MLP sur h, qui lit éventuellement aussi une couche intermédiaire, moins spécialisée, pour sortir
e(t, b) ∈ R⁶⁴ normé. Elle compte environ 0,5 M de paramètres. Chaque jeton bande × trame reçoit un vecteur.
Deux jetons dominés par la même source doivent avoir des vecteurs proches, **même s'ils viennent de deux
moments éloignés du morceau** (même lead, notes différentes, filtre ouvert puis fermé).

C'est l'idée du *deep clustering* (Hershey 2016) et du *deep attractor network* (Chen 2017), avec trois
différences :
- elle porte sur les jetons d'un tronc pré-entraîné, pas sur des bins STFT bruts ;
- elle est entraînée **entre deux extraits du même morceau**, pour imposer la cohérence à l'échelle du
  morceau ;
- elle est entraînée avec des **labels partiels hiérarchiques** (section 4.1).

### 3.3 Découverte au niveau du morceau

Ce n'est pas un réseau mais un algorithme, sans paramètre appris.
1. On tire environ 50 000 jetons du morceau, pondérés par leur énergie (un jeton silencieux ne vote pas).
2. Un k-means à 256 micro-clusters, puis un regroupement agglomératif (lien moyen, cosinus) sur les
   centroïdes, donnent un **dendrogramme**.
3. On coupe le dendrogramme à un seuil calibré sur la validation (MoisesDB + synthétique), ou à un nombre de
   pistes demandé (`--tracks 4` pour Mixxx), ou selon un curseur de granularité.
4. Chaque cluster k donne un centroïde q_k et une carte a posteriori π_k(t, b) = softmax_j(⟨e, q_j⟩/τ). La
   carte dit **où** k domine, relativement aux autres sources.

*Variante B, qui réutilise le modèle actuel* (schéma EEND-VC de la diarisation) :
- on garde le décodeur à attracteurs pour proposer, extrait par extrait, des sources et leurs vecteurs ;
- on regroupe ces vecteurs sur tout le morceau ;
- les centroïdes deviennent les requêtes.

On l'essaie si le regroupement de jetons laisse filer les sources discrètes, qui ne dominent aucun jeton.

Ce qu'on gagne par rapport aux 16 emplacements :
- **compter** devient une coupe d'arbre, réglable, au lieu d'un seuil sur 16 logits appris sur extraits de 4 s ;
- **la cohérence sur le morceau** est acquise par construction : une requête est valable de la première à la
  dernière seconde, et le recollage de `split.py` disparaît ;
- **la hiérarchie** est gratuite : le dendrogramme est l'arbre de la section 2.1 ;
- aucun emplacement n'est lié à un type, puisqu'il n'y a plus d'emplacement.

### 3.4 Extracteur conditionné par requête

Pour chaque requête k, l'extracteur procède ainsi :
1. FiLM(h ; q_k), puis on ajoute une projection de [π_k, ⟨e, q_k⟩] : une carte « où » et une carte « quoi » ;
2. il passe dans les **L dernières couches du tronc, recopiées** et précédées chacune d'un FiLM initialisé à
   l'identité ;
3. il passe dans une **tête de masque pré-entraînée, recopiée** (« other » de SW, ou la plus générale de
   Mega 53), qui sort un masque complexe stéréo.

À l'initialisation, l'extracteur reproduit exactement le modèle pré-entraîné ; il apprend seulement à se
laisser orienter par la requête. Le principe est celui de Banquet (Watcharasupat et Lerch, ISMIR 2024 : un seul décodeur pour tous
les stems, conditionné par une requête), à une différence près : **la requête vient du mix lui-même** (un
cluster), pas d'un enregistrement de référence fourni par l'utilisateur.

La même requête peut désigner **une source ou un groupe**. Un nœud haut du dendrogramme, par exemple « toute
la batterie », a son propre centroïde et sa propre carte (la somme des π de ses membres). Un seul extracteur
sert donc toutes les granularités, et l'export 4 pistes pour Mixxx n'est qu'une coupe de l'arbre.

### 3.5 Assemblage

Un filtrage de Wiener joint répartit le mix entre les K masques et un masque « reste » : un même son ne
peut pas apparaître deux fois. Le reste vaut mix − Σ pistes, si bien que la somme redonne exactement le mix,
comme aujourd'hui.

### 3.6 Coût de calcul

| | Aujourd'hui (16 emplacements) | Plan v2 |
|---|---|---|
| Passes de tronc par extrait | 1, avec gradient | 1, sans gradient |
| Réseau par source | tête complète × 16, toujours | L couches + tête × Q requêtes tirées (Q = 4) |
| Paramètres entraînés | tronc + décodeur + têtes | tête d'identité + L couches (ou LoRA) + tête de masque |
| Lot réaliste sur 24 Go | 2 extraits de 4 s | 8 extraits de 6 s × 4 requêtes (à vérifier en phase 2) |
| Inférence, 12 pistes | 1 passe | ≈ 3 à 4 passes SW équivalentes (tronc une fois, puis 12 × L/12 du tronc + tête) |

Un BS-RoFormer de cette taille tourne à environ 0,02 fois le temps réel sur un GPU récent (mesure publiée,
arXiv 2412.06965). Un morceau de 5 minutes en 12 pistes devrait donc se séparer en moins d'une minute sur
une 4090, ce qui convient à une préparation de sets hors ligne. Ce chiffre est à mesurer en phase 0.

---

## 4. Pertes

### 4.1 Identité : affinité hiérarchique à labels partiels

On tire M ≈ 2048 jetons sur **deux extraits du même morceau**, pondérés par énergie. Pour chaque paire
(i, j), une cible d'affinité découle de la vérité disponible :

| Les jetons i et j sont dominés par… | Cible | Exemple |
|---|---|---|
| la même feuille connue | 1 | deux jetons du même hat (synthétique) |
| deux feuilles différentes | 0 | kick contre pad |
| la même feuille **grossière** dont on ignore l'intérieur | **ignorée** | deux jetons de « other » dans MUSDB |
| (option) deux feuilles sœurs dans l'arbre | α ≈ 0,3 | kick contre hat, tous deux batterie |

La perte est une BCE entre la cible et le cosinus mis à l'échelle, pondérée par le produit des énergies et
par la dominance (un jeton à 50/50 compte peu). Avec M = 2048, cela fait 4 M de paires par extrait :
négligeable.

L'option α rend le dendrogramme musical : la batterie se regroupe avant de rejoindre les synthés. Ce n'est
pas un classifieur, aucun nom n'est appris, et la règle « pas de spécialisation par type » est respectée.

### 4.2 Extraction

On tire jusqu'à Q = 4 requêtes par extrait, parmi :
- **une source présente** : la requête est le prototype oracle de la source, c'est-à-dire la moyenne des e
  pondérée par sa dominance, calculée **sur un autre extrait du même morceau**, avec du bruit (jetons
  retirés, mélange léger avec un prototype voisin) pour imiter les erreurs de découverte. La cible est la
  source ;
- **un groupe** (sous-arbre, ou feuille grossière dans MUSDB) : la requête est le prototype du groupe, la
  cible la somme de ses membres ;
- **une source absente de l'extrait mais présente ailleurs dans le morceau** (un break) : la cible est le
  silence. Le modèle apprend ainsi à se taire quand on lui demande une source qui ne joue pas.

La perte est −SNR, plafonné comme aujourd'hui, plus une perte STFT multi-résolution légère. La carte π est
calculée avec les prototypes de toutes les sources de l'extrait, comme à l'inférence.

On ajoute un **régulariseur de niveau** (Watcharasupat et Lerch, ICASSP 2026) : l'énergie de la sortie doit
suivre celle de la cible. Banquet s'effondrait parfois vers des sorties silencieuses, et un extracteur
conditionné y est sujet. Aux requêtes de groupe s'ajoute la **contrainte de monotonie** de Manilow et al.
(2020) : le masque d'un nœud n'est jamais plus petit que celui d'un de ses enfants.

### 4.3 Ancrage (distillation d'un enseignant à stems fixes)

Sur de vrais morceaux sans stems, un enseignant à stems fixes découpe chaque extrait. Des requêtes de
groupe en sont tirées (prototype pondéré par la dominance de chaque stem de l'enseignant), et la cible est
le stem de l'enseignant. L'extracteur ne peut donc pas descendre sous l'enseignant, ce qui règle l'oubli de
5,6 dB. Le code de `distill.py` sert presque tel quel.

**Mega 53 est un enseignant hiérarchique tout fait.** Ses stems se recouvrent : « drums » contient
« kick », « vocal » contient « lead-vocal ». Ils forment donc un pseudo-arbre, exactement ce que la perte
4.1 sait lire :
- les nœuds fiables (kick, snare, hh, bass, lead-vocal…) servent de feuilles ;
- « synth » sert de feuille **grossière**, dont l'intérieur est ignoré ;
- les stems presque vides sont écartés.

Il remplace la cascade SW → DrumSep, dont les liens de téléchargement MSST sont morts ; un miroir existe dans
`audio-separator`, à garder en secours. Sa qualité stem par stem n'est pas publiée : on la mesure en phase 0
sur MoisesDB avant de lui faire confiance, et seuls les stems où il bat SW servent d'ancre. Les pseudo-stems
sont filtrés par confiance, et la perte est tronquée (section 6, Distillation).

### 4.4 Le cas MixIT, en une ligne

MixIT, c'est la perte 4.2 avec deux feuilles grossières (mélange A, mélange B) et des requêtes de groupe. Il
ne demande donc aucun code à part. Son intérêt est discuté en section 6.

Au passage, la conception par requêtes supprime l'appariement combinatoire. Chaque requête désigne sa cible,
il n'y a donc plus de PIT ni de recherche de partition. Le `group_loss` actuel doit sinon choisir parmi 4¹⁶
façons de ranger 16 sorties dans 4 stems, et il s'en tire par une approximation.

---

## 5. Données

### 5.1 Quatre sources de vérité, une seule perte

| Niveau | Jeux | Vérité | Rôle |
|---|---|---|---|
| **A. Réel fin** | MoisesDB (train), MedleyDB, Slakh2100, StemGMD (éléments de batterie) | instrument par instrument (arbre à 2 niveaux) | timbres réels, la vérité la plus précieuse |
| **B. Générateur v3** | `priism gen`, réaliste (5.2) | arbre complet | électronique, granularité fine, cas difficiles |
| **C. Réel grossier** | MUSDB18-HQ train | 4 feuilles | voix, instruments acoustiques |
| **D. Réel sans stems** | bibliothèque de Lucas, FMA électronique | pseudo-arbre de l'enseignant (Mega 53, SW en secours) | **domaine cible**, ancrage |

Mélange de départ, par pas : 30 % A, 30 % B, 25 % D, 15 % C. On l'ajuste ensuite selon le banc d'essai réel,
jamais selon la validation synthétique. Règle : **au moins la moitié des pas sur du son réel**.

Deux faits publiés justifient ce mélange :
- **Slakh** (Manilow et al. 2019) : sur MUSDB test, entraîner sur le synthétique seul donne une basse à
  −2,4 dB, contre −0,5 dB avec MUSDB seul. Le synthétique **ajouté** au réel aide, mais à peu près autant
  qu'un simple remix des stems réels entre morceaux.
- **« Embracing Cacophony »** (Jeon et al., MERL 2024-2026) : mélanger au hasard des stems de morceaux
  différents bat les mix originaux de 2,1 dB, à condition d'avoir assez de morceaux distincts.

On garde donc environ 30 % de mélanges **incohérents** (stems de morceaux différents, sans calage) en plus des
morceaux arrangés. Pour les niveaux A et C, c'est de l'augmentation gratuite.

### 5.2 Générateur v3 : arrangement synthétique, timbres réels

Le moteur d'arrangement (genres, tempo, harmonie, sections, 2 à 16 sources) et le mastering additif restent.
On change ce qui sonne :

| Aujourd'hui | Plan v3 |
|---|---|
| Batterie synthétisée (balayages sinus, bruit filtré) | **Sampler** : vrais one-shots (909, 808, packs sous licence libre), mêmes motifs, vélocité, accordage, enveloppe, couches. Une boîte à rythmes *est* un sampler : c'est ainsi que la musique électronique est faite |
| Synthé maison + Surge XT | Surge XT + autres synthés libres rendus par DawDreamer (Vital, Dexed, OB-Xd, Odin 2), à valider en headless |
| Rien de réel | **Vrais stems et boucles** comme sources : stems MoisesDB, MedleyDB et Slakh, boucles FSL10K, recalés au tempo et à la tonalité du morceau (`tempo_key.py`) |
| Pas de voix | Voix et *vocal chops* découpées dans les stems de voix réels (MUSDB, MoisesDB) |

Le générateur garde sa garantie : la somme des pistes vaut exactement le mix, la vérité est complète, et tout
se régénère depuis la graine.

Pourquoi les timbres d'abord : **SynthSOD** (2025, orchestre rendu par une banque d'échantillons) marche
sur son propre test synthétique et sur aucun enregistrement réel. Ses auteurs l'attribuent au manque de
diversité des timbres. C'est le même symptôme que nos 5,6 dB. Aucun travail publié n'utilise de la musique
électronique générée de façon procédurale pour la séparation : le générateur de Priism est en soi une
contribution.

### 5.3 Évaluation : jamais entraînée, toujours réelle

| Jeu | Granularité | Ce qu'il mesure |
|---|---|---|
| **MoisesDB test** (le découpage de Banquet et MuS3D : 48 morceaux) | instrument | **boussole principale** : la séparation fine sur du vrai son, comparable aux 6,8 et 7,5 dB de MuS3D |
| **NI Free Stems** (65) | 4 groupes, électronique | le domaine cible au niveau grossier |
| MUSDB18-HQ test (50) | 4 stems | comparaison avec la littérature |
| Validation synthétique fixe (64) | arbre complet | diagnostic seulement, jamais critère de décision |
| **mshoxxDB** (Taenzer, ISMIR 2024 LBD : 18 morceaux électroniques, multipistes + MIDI), déjà lu par `eval-stems` | instrument, **électronique** | **le vrai juge** : seul jeu public électronique et fin |
| *Option :* 10 à 20 morceaux de plus (stems de concours de remix, projets d'amis producteurs) | instrument, électronique | élargir mshoxxDB |

**Attention à la contamination.** On ignore sur quelles données SW a été entraîné. Mega 53 reprend la
taxonomie de MoisesDB, et X-LANCE a été entraîné sur MoisesDB. Un tronc ou un enseignant venu de ces modèles
a donc pu voir MoisesDB test, et ses scores y seraient flatteurs. Deux règles en découlent :
- on compare les variantes **entre elles** sur MoisesDB, avec le même tronc ;
- on juge l'absolu sur NI et mshoxxDB, que ces modèles n'ont probablement pas vus.

---

## 6. Verdicts sur les stratégies d'entraînement

### MixIT : pas comme signal principal

- **Le raccourci.** Deux morceaux réels additionnés se séparent par ce qui les distingue globalement
  (tempo, tonalité, style, mixage), pas par instrument. `tempo_key.py` réduit le raccourci, sans le
  supprimer : il reste le style, la densité et le mastering.
- **Aucune pression vers le fin.** MixIT demande seulement que les sorties se regroupent en A et B. Rien
  n'oblige à séparer le kick des hats *à l'intérieur* de A. La granularité obtenue est arbitraire :
  c'est le problème de sur- et sous-séparation connu de MixIT.
- **Mélange irréaliste.** Deux morceaux complets, c'est deux fois la densité d'un vrai morceau, avec des
  sources désynchronisées. Or la difficulté réelle, ce sont des sources **synchrones** et **en harmonie** :
  kick et basse calés, sidechain.
- **Ce qui est mesuré.**
  - Dans l'article original (Wisdom et al. 2020), un seul exemple musical (annexe H) : le morceau se coupe
    en « rythmique » et « harmonique », et la voix est coupée en deux.
  - Sur FUSS, une entrée à une seule source tombe à 6,9 dB, contre 37,1 dB en supervisé : c'est la
    sur-séparation.
  - Le seul travail récent sur la musique (Saijo et Bando, arXiv 2505.07631, 2025) utilise MixIT en
    **pré-entraînement** sur 890 h de FMA, puis un fine-tune sur MUSDB. Le gain va de **+0,3 à +0,55 dB**,
    et passer d'un sous-ensemble de FMA à FMA entier n'ajoute qu'environ 0,1 dB.

Le verdict : quelques dixièmes de dB pour un coût de données et de calcul réel, et aucun signal vers la
granularité fine, qui est précisément notre problème.

Ce qu'on en garde : une variante ciblée, **MixIT à l'intérieur d'un stem**. On prend les stems « synth » de
l'enseignant pour deux morceaux calés en tempo et en tonalité, puis on demande de les redécouper. Cela vise
exactement le trou (séparer synthé contre synthé, sur de vrais timbres), avec les deux feuilles A et B comme
vérité. On y ajoute la perte de parcimonie de *Sparse MixIT* (Wisdom et al. 2021), qui limite la
sur-séparation. C'est une option de la phase 4, pas une fondation.

### Synthétique : indispensable, à condition de sonner vrai

C'est la seule source de vérité fine pour la musique électronique. L'écart de 5,6 dB vient surtout des
timbres (batterie procédurale, synthés qui se ressemblent) et de la part écrasante du synthétique dans les
pas. D'où le générateur v3 (5.2) et la règle « au moins 50 % de son réel ».

### Distillation : le pont vers le réel, sans trahir la règle « pas de classes »

Les modèles à stems fixes (SW, Mega 53, DrumSep…) ne sont jamais le produit : ils servent d'**enseignants** sur le
domaine cible. Le modèle final reste sans classes, mais il hérite de leur qualité au niveau grossier (perte
d'ancrage) et s'en libère là où ils sont aveugles, en découpant « synth » ou « other ».

Ce qui marche, d'après le défi SDX23 (étiquettes bruitées) et BSRNN semi-supervisé :
- **filtrer les pseudo-stems** : ne garder qu'un stem que l'enseignant isole avec confiance. Luo et Yu
  testent un écart d'énergie de 30 dB et passent de 8,24 à 8,97 dB sur MUSDB avec 1750 morceaux non
  étiquetés ;
- **tronquer la perte** : on écarte les exemples dont la perte est la plus haute, probablement des erreurs de
  l'enseignant. Seul, cela fait +0,4 dB chez kuielab.

Les deux s'appliquent directement à l'ancrage 4.3.

### Auto-entraînement (*RemixIT*) : la bonne façon d'utiliser du réel sans stems, plus tard

Une fois le pipeline correct (phase 3), un enseignant (la moyenne mobile, EMA, du modèle) découpe des
morceaux réels. On **remixe** ses pistes : gains changés, pistes coupées, pistes échangées entre morceaux
calés en tempo et en tonalité. L'élève apprend à retrouver les pistes de l'enseignant dans ce remix. C'est
plus sain que MixIT, puisque le mélange reste un vrai morceau, et cela adapte le modèle aux timbres de la
bibliothèque de Lucas. Les pistes de faible confiance sont filtrées.

Dans RemixIT (Tzinis et al. 2022), l'élève dépasse son enseignant hors domaine de plus de 1 dB, même avec
de mauvaises pseudo-cibles. Pour la voix, le remix calé en hauteur (Yuan et al. 2022) fait 10,3 → 11,7 dB,
et c'est la qualité de l'enseignant qui commande le gain : d'où cette étape en dernier, sur un pipeline
déjà bon.

Le même principe, en version **MixCycle** (Karamatlı et Kırbız 2022), donne une **auto-évaluation sans
référence**. On remixe les pistes sorties, on resépare, puis on compare. Elle permet de suivre la qualité sur
la bibliothèque de Lucas, qui n'a pas de stems.

---

## 7. Phases, critères de passage, coût

Prix RunPod constatés en 2026 : RTX 4090 environ 0,34 $/h en Community Cloud, L40S environ 0,79 $/h,
A100 80 Go environ 1,2 à 1,6 $/h.

### Phase 0 : banc d'essai réel (1 à 2 jours, ~5 $)

- `priism bench` : MoisesDB test, mshoxxDB, NI, MUSDB test et validation synthétique ; SNR strict, SNR groupé
  par niveau, pureté, complétude, comptage. Il écrit un JSON et une ligne de tableau dans
  `docs/bench.md`.
- Lignes de référence :
  1. SW seul ;
  2. **Mega 53 seul**, ses stems vides filtrés : c'est l'adversaire « classes fixes » le plus fort, et la
     mesure de sa fiabilité comme enseignant ;
  3. **cascade** SW → DrumSep sur la batterie (déjà dans `eval-stems --cascade`) ;
  4. meilleur checkpoint actuel (stage D ou G) avec `split.py` ;
  5. **oracles** : masque idéal sur la STFT du tronc (borne haute) et clusters oracle ;
  6. *option* : SAM Audio avec les noms d'instruments oracle comme requêtes texte, pour situer le
     génératif.
- Livrable : **le tableau qui sert de boussole**. Chaque expérience suivante y ajoute une ligne.

### Phase 1 : sonde d'identité sur tronc gelé (1 à 2 jours, ~15 h de GPU, ~5 à 10 $)

- On entraîne seulement la tête d'identité (4.1), pour **chacun des troncs** de 3.1 (SW, Mega 53,
  X-LANCE synth) et pour trois prises de représentation : couche finale, couche du milieu, concaténation.
- Données d'entraînement : MoisesDB train, Slakh et le générateur actuel.
- Mesures sur MoisesDB test et mshoxxDB : pureté et complétude des clusters de jetons, avec le nombre oracle puis le
  nombre automatique ; SNR obtenu en prenant la carte π comme masque, sans extracteur.
- **Passage** : les clusters séparent les instruments au sein de « other » (complétude ≥ 0,6 sur
  synthés / claviers / guitares, à ajuster sur l'oracle).
- On retient le meilleur couple tronc × prise pour la suite.
- **Sinon** : LoRA sur le tronc en phase 2, ou prise de représentation plus basse.
- Cette sonde dit en une journée si le tronc pré-entraîné « entend » les instruments, c'est-à-dire si
  tout le plan tient.

### Phase 2 : extracteur conditionné (3 à 5 jours, ~60 h de GPU, ~25 à 50 $)

- Extracteur 3.4, pertes 4.1 à 4.3, données A + B (générateur actuel au début) + C + D.
- Requêtes oracle (prototypes calculés sur un autre extrait, bruités).
- Mesures :
  - SNR d'extraction par instrument sur MoisesDB test, avec requêtes oracle ;
  - requêtes de groupe sur NI et MUSDB, comparées à SW ;
  - pureté et complétude.
- **Passage** : au niveau groupe, au moins SW − 0,5 dB sur NI (l'ancrage tient), et au niveau fin, mieux
  que la cascade là où elle est aveugle (les instruments rangés dans « other »).
- Repère externe : MuS3D obtient **7,5 dB** au niveau instrument sur MoisesDB test avec requêtes oracle.
  Viser au moins ce niveau, en gardant à l'esprit la contamination possible (5.3).

En parallèle, sur CPU : générateur v3 (sampler de batterie d'abord, puis vrais stems et boucles, puis
DawDreamer).

### Phase 3 : pipeline sur morceau entier (1 semaine, ~60 h de GPU, ~25 à 50 $)

- Découverte 3.3 et calibration de la coupe.
- Fine-tune avec des **requêtes issues de la vraie découverte** : on regroupe les e du modèle, puis on
  apparie aux sources vraies par recouvrement des cartes. On referme ainsi l'écart entre l'oracle et
  l'inférence.
- Wiener joint ; `priism split` v2 avec `--tracks`, `--granularity`, l'arbre en JSON et des pages
  `listen`.
- **Passage** : sur MoisesDB test avec comptage automatique, meilleur que le modèle actuel et que la
  cascade sur le F de pureté et complétude. Proche des **6,8 dB à l'aveugle** de MuS3D, mais sur des
  morceaux **entiers** et non sur des segments de 10 s. Meilleur que la cascade sur mshoxxDB. Écoute validée
  sur 10 morceaux de la bibliothèque.

### Phase 4 : adaptation au réel (option, ~40 h de GPU, ~15 à 40 $)

- RemixIT sur la bibliothèque de Lucas ; MixIT à l'intérieur de « synth » si la phase 3 laisse les synthés
  fusionnés.
- Suivi sans référence par MixCycle sur la bibliothèque.
- **Passage** : gain sur NI et mshoxxDB, sans perte sur MoisesDB.

### Phase 5 : finitions

- **Découper à la demande** (les jumeaux) : un séparateur à 2 sorties, appliqué à une piste choisie, entraîné
  par PIT sur les familles de jumeaux du générateur. Le travail déjà fait sur les jumeaux y trouve sa place.
- **Noms des pistes** : classer les pistes après coup, avec PE-A-Frame (Apache-2.0) ou CLAP en texte libre,
  ou avec un petit classifieur sur les centroïdes.
- Export `.stem.mp4` pour Mixxx : 4 pistes, par coupe de l'arbre.

**Total : 150 à 250 h de GPU, soit 60 à 200 $** selon la carte, plus le temps CPU du générateur.

---

## 8. Risques et plans B

| Risque | Signe | Plan B |
|---|---|---|
| Le tronc gelé ne distingue pas deux synthés | phase 1 : complétude faible dans « other » | LoRA sur le tronc ; prise de représentation plus basse ; en dernier recours, tronc entraînable avec ancrage fort |
| Une source discrète (pad sous tout le reste) ne domine aucun jeton | source absente des clusters, retrouvée dans le reste | seuil de dominance plus bas pour la découverte ; seconde découverte sur le reste |
| La coupe du dendrogramme est instable d'un genre à l'autre | comptage mauvais sur NI | seuil appris par un petit régresseur (statistiques des fusions) ; nombre de pistes fixé par l'utilisateur |
| L'écart entre requêtes oracle et découverte reste grand | phase 3 très en dessous de la phase 2 | requêtes bruitées plus tôt ; affinage de la découverte par quelques itérations d'attention entre q_k et les jetons, dans l'esprit de l'attention par slots mais au niveau du morceau |
| Licences | MoisesDB, MedleyDB, MUSDB, FMA : non commercial. Licence de SW **inconnue** (auteur inconnu, dépôt d'origine disparu). Poids de Mega 53 sans licence explicite (dépôt MIT) ; X-LANCE sous MIT mais entraîné sur MoisesDB | usage personnel et recherche pour l'instant. Avant toute distribution : Slakh, StemGMD (CC BY) et le générateur (libre), plus un tronc dont la licence est claire |
| Disponibilité des poids | le dépôt d'origine de SW renvoie 401, des miroirs existent | épingler le SHA-256 et garder une copie sur le volume du pod |
| Mécanismes qui s'empilent à nouveau | plus de 2 drapeaux expérimentaux actifs | une expérience = une hypothèse = une ligne du banc, consignée dans `docs/journal.md` |

---

## 9. Ce qu'on arrête, ce qu'on garde

**On arrête** (le code reste, il ne sort plus de la file) :
- les sondes et duels sur les jumeaux (v2, v3, facteurs, JEPA, mix-pitch, morceaux de 8 s) ;
- le séparateur à 16 emplacements comme direction principale : il devient une ligne de référence du banc ;
- la décision automatique sur 1000 pas de validation synthétique.

**On garde** :
- le générateur et son moteur d'arrangement ;
- le flux de morceaux, `distill.py` (l'enseignant), `eval_stems.py`, `split.py` (pour la ligne de référence du modèle actuel),
  `listen` ;
- `tempo_key.py`, l'outillage du pod.

**Méthode** :
- une hypothèse par expérience, jugée sur le banc **réel** ;
- au moins 5000 pas avant de conclure sur une architecture ;
- un journal d'une ligne par décision.

---

## 10. Pourquoi ces choix plutôt que d'autres

- **Pas de séparation récursive « une source + reste »** (OR-PIT, Takahashi 2019). Elle donne aussi un nombre
  arbitraire et un arbre, mais au prix d'une passe complète du tronc par source, avec des erreurs qui
  s'accumulent de niveau en niveau. Notre arbre vient du dendrogramme, en une passe.
- **Pas de modèle génératif** (diffusion, *flow matching*). Il est trop cher à entraîner, et il peut inventer
  du son, ce qu'un DJ ne veut pas. Le masquage reste fidèle au morceau.
- **Pas de 16 emplacements DETR en fondation.** C'est SepTDA, transposé à la musique. En parole, son
  comptage tombe déjà de 90 % à 83 % entre 4 et 5 locuteurs. À 16 sources corrélées, avec des extraits de
  4 s, les chiffres du repo (spécialisation par type, emplacements morts) montrent la même limite.
- **Pas de MixIT comme fondation** : voir la section 6.
- **Pas de nouveau tronc.** Le savoir de SW sur la musique réelle vaut des milliers d'heures de GPU qu'on n'a
  pas. On le recopie au lieu de le réapprendre.

---

## Références

Revue faite le 9 octobre 2026. Les chiffres sont ceux des articles ; les mentions « non vérifié » signalent
ce qu'on n'a pas pu recouper.

**Nombre de sources inconnu, attracteurs, requêtes**
- Hershey et al., *Deep clustering*, ICASSP 2016, arXiv 1508.04306 ; Luo et al., *Chimera* (musique),
  ICASSP 2017, arXiv 1611.06265.
- Chen, Luo, Mesgarani, *Deep Attractor Network*, ICASSP 2017, arXiv 1611.08930.
- Takahashi et al., *Recursive speech separation (OR-PIT)*, Interspeech 2019, arXiv 1904.03065.
- Dovrat, Nachmani, Wolf, *Many-speakers separation with Hungarian PIT*, Interspeech 2021, arXiv 2104.08955.
- Horiguchi et al., *EEND-EDA*, Interspeech 2020, arXiv 2005.09921 ; Kinoshita et al., *EEND-VC*,
  ICASSP 2021, arXiv 2010.13366 ; Zeghidour et Grangier, *Wavesplit*, arXiv 2002.08933.
- Lee et al., *SepTDA* (Transformer decoder attractors), ICASSP 2024, arXiv 2401.12473.
- Reddy et al., *AudioSlots*, 2023, arXiv 2305.05591.
- Watcharasupat et Lerch, *Banquet*, ISMIR 2024, arXiv 2406.18747 (code MIT, poids CC BY-NC-SA) ;
  *Hyperellipsoidal queries*, ICASSP 2026, arXiv 2501.16171.
- Saijo et al., *TUSS*, ICASSP 2025, arXiv 2410.23987 (AGPL).
- Kong et al., *Universal Source Separation*, arXiv 2305.07447.
- **Kallinen, Moliner, Juvela, Välimäki, *Music Source Separation via Stem Discovery (MuS3D)*,
  arXiv 2609.30912, sept. 2026.** C'est le travail le plus proche. Sur MoisesDB (découpage de Banquet),
  avec des segments de 10 s : F1 de découverte de 0,74 au niveau instrument ; BS-Locoformer à 7,5 dB avec
  requête oracle et 6,8 dB à l'aveugle. Le code est annoncé « à l'acceptation ».

**Hiérarchie**
- Manilow, Wichern, Le Roux, *Hierarchical Musical Instrument Separation*, ISMIR 2020.
- Petermann et al., *Hyperbolic Audio Source Separation*, ICASSP 2023.
- Pereira et al., *MoisesDB*, ISMIR 2023, arXiv 2307.15913.

**Génératif**
- Shi et al., *SAM Audio*, arXiv 2512.18099, déc. 2025 ; évaluation indépendante sur MUSDB,
  arXiv 2607.27828 : vocals 3,7, drums 5,6, bass 5,1, other −1,7 dB.
- Mariani et al., *MSDM*, ICLR 2024 ; Postolache et al., *GMSDI*, ICASSP 2024.

**Entraînement sans vérité complète**
- Wisdom et al., *MixIT*, NeurIPS 2020, arXiv 2006.12701 ; *Sparse, Efficient and Semantic MixIT*,
  WASPAA 2021, arXiv 2106.00847.
- Saijo et Bando, *Is MixIT Really Unsuitable for Correlated Sources?*, arXiv 2505.07631, 2025.
- Tzinis et al., *RemixIT*, IEEE JSTSP 2022, arXiv 2202.08862.
- Karamatlı et Kırbız, *MixCycle*, IEEE SPL 2022, arXiv 2202.03875.
- Luo et Yu, *BSRNN semi-supervised fine-tuning*, arXiv 2209.15174.
- Fabbro et al., *Sound Demixing Challenge 2023, Music Demixing Track*, TISMIR 2024, arXiv 2308.06979.
- Yuan et al., *Pitch-aware remixing*, arXiv 2203.15092.

**Données**
- Manilow et al., *Slakh2100*, WASPAA 2019, arXiv 1909.08494 (CC BY 4.0).
- Garcia-Martinez et al., *SynthSOD*, IEEE OJSP 2025, arXiv 2409.10995.
- Jeon et al., *Why does music source separation benefit from cacophony?*, arXiv 2402.18407 ;
  *Embracing Cacophony*, MERL TR2026-012.
- Défossez, *Hybrid Spectrogram and Waveform Source Separation*, arXiv 2111.03600 (remix calé en tempo et
  en hauteur).
- Mezza et al., *StemGMD / LarsNet*, arXiv 2312.09663 (StemGMD CC BY 4.0, poids LarsNet CC BY-NC).
- Taenzer, *mshoxxDB*, ISMIR 2024 LBD (18 morceaux électroniques ; licence non précisée).
- FSL10K (Freesound Loops), ISMIR 2020 (licence propre à chaque boucle, certaines non commerciales).

**Modèles pré-entraînés**
- BS-Roformer-SW : miroirs `huggingface.co/enerjazzer/BS-ROFO-SW-Fixed` et `lumabeat/bs-roformer-sw`.
  Auteur et licence inconnus. Scores publiés sur le Multisong de MVSep : bass 14,6, drums 14,1, other 8,7.
- MVSep Mega 53 stems : MSST release v1.0.21 (avril 2026),
  `mvsep_mega_model_bs_roformer_53_stems_v1.ckpt`. Pas de scores publiés ; au moins 16 Go de VRAM.
- X-LANCE MSR Challenge 2025 : `huggingface.co/chenxie95/xlance-msr-ckpt` (MIT), têtes *syn*, *perc*,
  *orch*.
- DrumSep MDX23C 6 stems (aufr33, jarredou), miroir dans la liste de modèles d'`audio-separator`.
- Représentations, pour nommer les pistes en phase 5 :
  - PE-A-Frame (Meta, Apache-2.0) ;
  - LAION-CLAP music ;
  - MuQ (CC BY-NC).
