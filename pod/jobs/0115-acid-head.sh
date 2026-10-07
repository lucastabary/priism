# after: 0105
# GPU: the 303 as a heads-only specialist (whole body frozen), same audio as
# acid-v1 and 0110: the three levels of fine-tuning compare on the acid SDR.
W=${PRIISM_WORKSPACE:-/workspace}
git -C "$W/priism" pull -q --ff-only || true  # adapter.sh as of now, not as of the pod start
exec bash "$W/priism/pod/adapter.sh" acid-head acid "$W/data/train_acid_fold" "$W/data/valid_acid_fold" "$W/data/.ready-acid-fold" head
