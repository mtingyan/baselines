python main_link.py --mode=train --dataset='tgbl-coin' --interv_size_ratio 0.4  --device 1

python main_link.py --mode=train --dataset='tgbl-review' --interv_size_ratio 0.4 --n_factors 4 --delta_d 4 --device 1 --lr 0.1

python main_link.py --mode=train --dataset='tgbl-comment' --interv_size_ratio 0.4 --n_factors 4 --delta_d 8 --device 2

python main_link.py --mode=train --dataset='ogbl-collab' --interv_size_ratio 0.4 --n_factors 4 --delta_d 4 --device 3

python main_node.py --mode=train --dataset='ogbn-arxiv' --interv_size_ratio 0.1 --n_factors 5 --delta_d 8 --beta 0.01 --max_epoch 300 --device 0