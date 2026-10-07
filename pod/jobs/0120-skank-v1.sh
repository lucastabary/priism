# after: 0106
# GPU: dub skank adapter specialist (skank / rest).
W=${PRIISM_WORKSPACE:-/workspace}
exec bash "$W/priism/pod/adapter.sh" skank-v1 skank "$W/data/train_skank" "$W/data/valid_skank" "$W/data/.ready-skank"
