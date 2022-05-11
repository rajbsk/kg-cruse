# KG-CRuSE

KG-CRUSE is a simple, yet effective LSTM based decoder that leverage Sentence-BERT embeddings to capture the in the dialogue history and Knowledge Graph elements (KG) to generate walks over the KG for effective conversation explanation.

Rajdeep Sarkar, Mihael Arcan, John P. McCrae. ["KG-CRuSE: Recurrent Walks over Knowledge Graph for Explainable Conversation Reasoning using Semantic Embeddings"](https://openreview.net/pdf?id=B4eeVgx-xZq), Proceedings of the 4th Workshop on Natural Language Processing for Conversational AI. 2022.

## Data Format
We use the OpenDialKG dataset to evaluate our proposed approach. The dataset used for training KG-CRuSE and the other baseline models can be downloaded from this ["link"](https://drive.google.com/file/d/1pZlmqku2suO1xAlhiS8M2tBwzuF17f2t/view?usp=sharing). Once you have downloaded the zip file, place the downloaded zip file inside the \"datasets\" folder. Then run the following command

```cd datasets```
```unzip dataset_nlp4convai.zip```

## Code Structure

The codes for KG-CRuSE and the baseline methods are present in the codes folder. Once you have extracted the dataset, you can easliy train and test the model. The Readme of each model is present in their respective folder.

When running the codes, \$split_id\$ can take one of 1, 2, 3, 4 or 5.
## Citation
Please cite our paper if you find our code and paper helpful.


