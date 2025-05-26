# python predict_temporal_links.py --config data_config/arxiv_HepPh.yaml --device 0 --seed 0 
# python predict_temporal_links.py --config data_config/tgbl-wiki.yaml --device 0 --seed 0 
# # python predict_temporal_links.py --config data_config/tgbl-review.yaml --device 1 --seed 0 
# python predict_temporal_links.py --config data_config/tgbl-flight.yaml --device 2 --seed 0 
# python predict_temporal_links.py --config data_config/tgbl-coin.yaml --device 3 --seed 0 
# python predict_temporal_links.py --config data_config/tgbl-comment.yaml --device 0 --seed 0 

 
# python predict_temporal_nodes_seeds.py --config data_config/ogbn-arxiv.yaml --device 0

# python predict_temporal_links_seeds.py --config data_config/ogbl-collab.yaml --device 1
# python predict_temporal_links_seeds.py --config data_config/tgbl-review.yaml --device 2

# python predict_temporal_links_seeds.py --config data_config/tgbl-comment.yaml --device 3
# python predict_temporal_links_seeds.py --config data_config/tgbl-coin.yaml --device 1
# python predict_temporal_links_seeds.py --config data_config/tgbl-flight.yaml --device 2 
# python predict_temporal_links_seeds.py --config data_config/tgbl-wiki.yaml --device 0 

# python predict_temporal_nodes_prompt.py --config data_config/ogbn-arxiv.yaml --device 0

# python predict_temporal_links_prompt.py --config data_config/ogbl-collab.yaml --device 1
# python predict_temporal_links_prompt.py --config data_config/tgbl-review.yaml --device 2

# python predict_temporal_links_prompt.py --config data_config/tgbl-comment.yaml --device 3
# python predict_temporal_links_prompt.py --config data_config/tgbl-coin.yaml --device 1
# python predict_temporal_links_prompt.py --config data_config/tgbl-flight.yaml --device 2 
# python predict_temporal_links_prompt.py --config data_config/tgbl-wiki.yaml --device 0 

# python predict_temporal_nodes_aug.py --config data_config/ogbn-arxiv.yaml --device 0

python predict_temporal_nodes_seeds.py --config data_config/ogbn-arxiv.yaml --device 3 --model ERM
python predict_temporal_links_seeds.py --config data_config/ogbl-collab.yaml --device 1 --model ERM
python predict_temporal_links_seeds.py --config data_config/tgbl-review.yaml --device 2 --model ERM
python predict_temporal_links_seeds.py --config data_config/tgbl-comment.yaml --device 3 --model ERM
python predict_temporal_links_seeds.py --config data_config/tgbl-coin.yaml --device 2 --model ERM


python predict_temporal_nodes_seeds.py --config data_config/ogbn-arxiv.yaml --device 1 --model DegreeEncodingV1
python predict_temporal_nodes_seeds.py --config data_config/ogbn-arxiv.yaml --device 0 --model TimeSeriesEncoding
python predict_temporal_links_seeds.py --config data_config/ogbl-collab.yaml --device 1 --model TimeSeriesEncoding
python predict_temporal_links_seeds.py --config data_config/tgbl-review.yaml --device 2 --model TimeSeriesEncoding
python predict_temporal_links_seeds.py --config data_config/tgbl-comment.yaml --device 3 --model TimeSeriesEncoding
python predict_temporal_links_seeds.py --config data_config/tgbl-coin.yaml --device 3 --model TimeSeriesEncoding

python predict_temporal_nodes_seeds.py --config data_config/BA-random.yaml --device 0 --model ERM
python predict_temporal_nodes_seeds.py --config data_config/BA-random.yaml --device 0 --model DegreeEncoding

# python predict_temporal_nodes_seeds.py --config data_config/penn.yaml --device 0 --model ERM
# python predict_temporal_nodes_seeds.py --config data_config/Reed98.yaml --device 0 --model ERM
# python predict_temporal_nodes_seeds.py --config data_config/Johns_Hopkins55.yaml --device 0 --model ERM
# python predict_temporal_nodes_seeds.py --config data_config/Cornell5.yaml --device 0 --model ERM
# python predict_temporal_nodes_seeds.py --config data_config/Amherst41.yaml --device 0 --model ERM

python predict_temporal_nodes_seeds.py --config data_config/SBM.yaml --device 0 --model ERM


python predict_temporal_links_seeds.py --config data_config/tgbl-review.yaml --device 2 --model TSS 
python predict_temporal_links_seeds.py --config data_config/tgbl-comment.yaml --device 3 --model TSS
python predict_temporal_links_seeds.py --config data_config/tgbl-coin.yaml --device 1 --model TSS

