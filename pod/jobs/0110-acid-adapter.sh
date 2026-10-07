# after: 0105
# GPU: the 303 as an adapter specialist (frozen base + LoRA), on the same audio as
# acid-v1, to compare full fine-tune against adapter on the acid SDR.
W=${PRIISM_WORKSPACE:-/workspace}
exec bash "$W/priism/pod/adapter.sh" acid-adapter acid "$W/data/train_acid_fold" "$W/data/valid_acid_fold" "$W/data/.ready-acid-fold"
