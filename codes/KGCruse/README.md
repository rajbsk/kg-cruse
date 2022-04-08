The official implementation of the NLP4ConvAI 2022 Paper: 

To train the KG-Cruse model, run
python3 main.py --max_acts=100000 --split_id=\$split_id\$ --model_name=\$name of the model\$ --data_directory=\$directory of the dataset\$

To test the model, run
python3 tester.py --split_id=\$split_id\$ --model_name=\$path to the KG-Cruse model\$ --data_directory=\$directory of the dataset\$