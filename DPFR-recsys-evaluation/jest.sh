for model in BPR ItemKNN MultiVAE NCL; do
  uv run RecBole/run_hyper_update.py --dataset=new_Jester --model=$model --params_file=RecBole/hyperchoice/$model.hyper
done