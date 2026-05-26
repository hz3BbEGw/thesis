for ds in Amazon-lb Lastfm QK-video Jester ML-10M ML-20M; do
  uv run eval/eval_MME.py --dataset=$ds && \
  uv run eval/eval_MME_rerank.py --dataset=$ds
done