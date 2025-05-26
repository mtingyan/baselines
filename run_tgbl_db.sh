python3 -m domainbed.scripts.train_tdg_lr --algorithm IRM_GNN_LR --dataset tgbl-review --test_env 2 --config ./data_config/tgbl-review.yaml
python3 -m domainbed.scripts.train_tdg_lr --algorithm GroupDRO_GNN_LR --dataset tgbl-review --test_env 2 --config ./data_config/tgbl-review.yaml
python3 -m domainbed.scripts.train_tdg_lr --algorithm VREx_GNN_LR --dataset tgbl-review --test_env 2 --config ./data_config/tgbl-review.yaml

# python3 -m domainbed.scripts.train_tdg_lr --algorithm IRM_GNN_LR --dataset tgbl-coin --test_env 2 --config ./data_config/tgbl-coin.yaml
# python3 -m domainbed.scripts.train_tdg_lr --algorithm GroupDRO_GNN_LR --dataset tgbl-coin --test_env 2 --config ./data_config/tgbl-coin.yaml
# python3 -m domainbed.scripts.train_tdg_lr --algorithm VREx_GNN_LR --dataset tgbl-coin --test_env 2 --config ./data_config/tgbl-coin.yaml

# python3 -m domainbed.scripts.train_tdg_lr --algorithm IRM_GNN_LR --dataset tgbl-comment --test_env 2 --config ./data_config/tgbl-comment.yaml
# python3 -m domainbed.scripts.train_tdg_lr --algorithm GroupDRO_GNN_LR --dataset tgbl-comment --test_env 2 --config ./data_config/tgbl-comment.yaml
# python3 -m domainbed.scripts.train_tdg_lr --algorithm VREx_GNN_LR --dataset tgbl-comment --test_env 2 --config ./data_config/tgbl-comment.yaml