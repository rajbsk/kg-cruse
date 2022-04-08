This folder contains the codes for the EMNLP 2020 paper: Attnio: Knowledge graph exploration with in-and-out attention flow for knowledge-grounded dialogue

To train the AttnIO model, run the following:
python3 main_AttnIO.py --split_id=\$split_id\$ --data_directory=\$location of the AttnIO dataset\$


One might notice in attnio_mode_new.py file, we have used the same GNN layer for different timesteps. We observed that on using timestep specific GNN layer, the performance remains similar.
