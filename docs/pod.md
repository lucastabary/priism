# Training on a GPU pod (RunPod)

*Moved from the former README (in French). Commands are now `priism-lab ...`.*


Le pod fait tourner une file de jobs (`pod/jobs/*.sh`, exécutés dans l'ordre
des noms) pour que la carte enchaîne sans attendre personne. Deux voies
tournent en parallèle : un job marqué `# lane: cpu` (préparation de données)
passe dans la voie CPU, les autres dans la voie GPU. Une ligne
`# after: 0030` fait attendre un job jusqu'à la fin du job 0030.

1. Créer le pod avec comme commande de démarrage
   `bash -c "until curl -fsSL https://raw.githubusercontent.com/lucastabary/priism/main/pod/setup.sh -o /root/setup.sh; do sleep 5; done; bash /root/setup.sh"`,
   la variable `PRIISM_WORKER_TOKEN` (donnée par `priism-lab pod token`) et le
   port HTTP 8000 exposé.
2. Suivre et piloter depuis n'importe où :
   `priism-lab pod status --url https://<pod>-8000.proxy.runpod.net`, puis
   `log`, `add`, `cancel`.
3. Tout vit sous `/workspace/priism` : `/workspace` est le volume réseau
   RunPod de Lucas, partagé avec d'autres projets (ne jamais toucher aux
   autres dossiers). Le volume survit à l'arrêt du pod ; les résultats se
   rapatrient aussi par `priism-lab pod pull runs --url ... --dest <dossier>`.

Jobs actuels (le volume fait 25 Go, partagé) :
- 0200 (CPU) rend le jeu de validation fixe : 64 chansons de 30 s.
- 0300 (GPU) : 500 pas du séparateur à attracteurs sur le cœur
  BS-Roformer-SW pré-entraîné, pour vérifier la mémoire et le début
  d'apprentissage ; 0310 enchaîne le long entraînement (reprend depuis
  `last.pt` si le pod a été arrêté). Les chansons d'entraînement ne sont
  pas stockées : `train-sep --stream` les génère pendant l'entraînement
  sur le disque local du pod (réserve tournante de 300 chansons).
- 0230 (CPU) récupère un peu de FMA (100 morceaux par genre, plafonné à
  12 Go, arrêt s'il reste moins de 4 Go sur le volume).

Les jobs du premier plan (4 stems fixes, fine-tune acid, spécialistes LoRA)
sont archivés dans `pod/jobs-v1/` et ne sont plus mis en file.
