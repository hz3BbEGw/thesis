for model in BPR ItemKNN MultiVAE NCL; do
   uv run RecBole/run_hyper_update.py --dataset=new_Amazon-lb --model=$model --params_file=RecBole/hyperchoice/$model.hyper && \
   uv run RecBole/run_hyper_update.py --dataset=new_Lastfm   --model=$model --params_file=RecBole/hyperchoice/$model.hyper && \
   uv run RecBole/run_hyper_update.py --dataset=new_QK-video  --model=$model --params_file=RecBole/hyperchoice/$model.hyper
done