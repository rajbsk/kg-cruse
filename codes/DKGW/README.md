Codes(re-implementation) for the ACL 2019 paper: Opendialkg: Explainable conversational reasoning with attention-based walks over knowledge graphs
Data Directory = kg-cruse/datasets/dataset_baseline/

To train the model, run
```python3 main_DialKgWalker.py --split_id=\$split_id\$ --data_directory=\$"directory of the dataset for DialKGWalker model"\$```

To test the model, run
```python3 tester.py --split_id=\$split_id\$ --data_directory=\$directory of the dataset for DialKGWalker model"\$ --model_path=\$"Path to the model file"\$```

