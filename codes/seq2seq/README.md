Codes for the Seq2Seq model used in the paper: "KG-CRuSE: Recurrent Walks over Knowledge Graph for Explainable Conversation Reasoning using Semantic Embeddings"

One should note that we used the modality attention mechanism used in the paper "Opendialkg: Explainable conversational reasoning with attention-based walks over knowledge graphs" as our encoder output. The output of the modality attention mechanism gave us better results for recall@1 metrics.

Data Directory = kg-cruse/datasets/dataset_baseline/


To train the model, run
```python3 main_seq2seq.py --split_id=\$split id\$ --data_directory=\$directory of the dataset\$```

To test the model, run
```python3 tester.py --split_id=\$split id\$ --data_directory=\$directory of the dataset\$ --model_path=\$Path to the model being tested\$```

