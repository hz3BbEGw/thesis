for ds in Amazon-lb Lastfm QK-video; do
  uv run pareto/generate_pareto.py --dataset=$ds
done