python3 -m domainbed.scripts.train_tdg --algorithm IRM_GNN --dataset ogbn-arxiv --test_env 2 --config ./data_config/ogbn-arxiv.yaml
python3 -m domainbed.scripts.train_tdg --algorithm GroupDRO_GNN --dataset ogbn-arxiv --test_env 2 --config ./data_config/ogbn-arxiv.yaml
python3 -m domainbed.scripts.train_tdg --algorithm VREx_GNN --dataset ogbn-arxiv --test_env 2 --config ./data_config/ogbn-arxiv.yaml

python3 -m domainbed.scripts.train_tdg_lr --algorithm IRM_GNN_LR --dataset ogbl-collab --test_env 2 --config ./data_config/ogbl-collab.yaml
python3 -m domainbed.scripts.train_tdg_lr --algorithm GroupDRO_GNN_LR --dataset ogbl-collab --test_env 2 --config ./data_config/ogbl-collab.yaml
python3 -m domainbed.scripts.train_tdg_lr --algorithm VREx_GNN_LR --dataset ogbl-collab --test_env 2 --config ./data_config/ogbl-collab.yaml

python3 -m domainbed.scripts.train_tdg_lr --algorithm IRM_GNN_LR --dataset tgbl-review --test_env 2 --config ./data_config/tgbl-review.yaml
python3 -m domainbed.scripts.train_tdg_lr --algorithm GroupDRO_GNN_LR --dataset tgbl-review --test_env 2 --config ./data_config/tgbl-review.yaml
python3 -m domainbed.scripts.train_tdg_lr --algorithm VREx_GNN_LR --dataset tgbl-review --test_env 2 --config ./data_config/tgbl-review.yaml

python3 -m domainbed.scripts.train_tdg_lr --algorithm IRM_GNN_LR --dataset tgbl-coin --test_env 2 --config ./data_config/tgbl-coin.yaml
python3 -m domainbed.scripts.train_tdg_lr --algorithm GroupDRO_GNN_LR --dataset tgbl-coin --test_env 2 --config ./data_config/tgbl-coin.yaml
python3 -m domainbed.scripts.train_tdg_lr --algorithm VREx_GNN_LR --dataset tgbl-coin --test_env 2 --config ./data_config/tgbl-coin.yaml

python3 -m domainbed.scripts.train_tdg_lr --algorithm IRM_GNN_LR --dataset tgbl-comment --test_env 2 --config ./data_config/tgbl-comment.yaml
python3 -m domainbed.scripts.train_tdg_lr --algorithm GroupDRO_GNN_LR --dataset tgbl-comment --test_env 2 --config ./data_config/tgbl-comment.yaml
python3 -m domainbed.scripts.train_tdg_lr --algorithm VREx_GNN_LR --dataset tgbl-comment --test_env 2 --config ./data_config/tgbl-comment.yaml