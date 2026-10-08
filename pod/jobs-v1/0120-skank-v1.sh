# after: 0106
# GPU: dub skank adapter specialist (skank / rest).
W=${PRIISM_WORKSPACE:-/workspace}
git -C "$W/priism" pull -q --ff-only || true  # adapter.sh as of now, not as of the pod start
exec bash "$W/priism/pod/adapter.sh" skank-v1 skank "$W/data/train_skank" "$W/data/valid_skank" "$W/data/.ready-skank"
