uv run RecBole/run_hyper_update.py --dataset=new_ML-20M --model=BPR      --params_file=RecBole/hyperchoice/BPR.hyper && \
uv run RecBole/run_hyper_update.py --dataset=new_ML-20M --model=ItemKNN  --params_file=RecBole/hyperchoice/ItemKNN-new_ML-20M.hyper && \
uv run RecBole/run_hyper_update.py --dataset=new_ML-20M --model=MultiVAE --params_file=RecBole/hyperchoice/MultiVAE.hyper && \
uv run RecBole/run_hyper_update.py --dataset=new_ML-20M --model=NCL      --params_file=RecBole/hyperchoice/NCL.hyper