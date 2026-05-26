for ds in Amazon-lb Lastfm QK-video; do
  uv run python -c "                                                        
from recbole.quick_start import run_recbole                                                                                                            
run_recbole(model='Pop', dataset='new_$ds')" 
done