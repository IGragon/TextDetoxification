**Name:** Igor Abramov

**Email:** ig.abramov@innopolis.university

**Group:** B21-DS-01

# Solution report

The final solution report can be found [here](reports/FinalSolutionReport.md) with more details on the model, training, evaluation, results, and e.t.c

# How to use

It is highly recommended running this repo in Colab or Kaggle notebooks. Therefore
commands will be described from that perspective. If you encounter problems refer to
[3.2 notebook](notebooks/3.2_Training_and_evaluating_paraphraser_within_module.ipynb) as it
contains installation, data loading, training, and evaluation done in Kaggle. (The same notebook can be run as is in the Colab)

``cd <destination>`` mostly indicates from which directory to run the script.

## Installation

Clone this repository to your environment with 

``git clone https://github.com/IGragon/TextDetoxification.git``

Then

``cd ./TextDetoxification/``

Run 

``pip3 install -r requirements.txt``

to install necessary dependencies.

## Loading data

Download datasets and generate datasets with

```bash
cd ./TextDetoxification/src/data
bash ./download_data.sh # loading data


python3 make_dataset.py # generating datasets
```

## Training 
If you want to save trained model to your HuggingFace run
```python
from huggingface_hub import notebook_login
notebook_login()
```
in a cell above

Then run 
```shell
wandb disabled
cd ./TextDetoxification/src/models
python3 train_model.py # add --store_locally to store model locally and not push to HuggingFace
```
Complete list of parameters can be found in [training script](src/models/train_model.py)

## Evaluation

For evaluation the final output will look as:
```
<a lot of garbage output with wild tqdm and stuff>
--------------------------------------------------
Evaluation results!
Mean metrics:
non-toxicity: 0.9697
fluency: 0.7799
similarity: 0.7283
```

Evaluation on 1000 random samples from eval.tsv

```shell
cd ./TextDetoxification/src/models
python3 evaluate_model.py
```

Evaluation on 500 toxic samples from toxicity_en.csv

```shell
cd ./TextDetoxification/src/models
python3 evaluate_model.py --use_toxicity_dataset
```

## Prediction

For prediction the final output will look as:
```
Original text: <toxic text>
Detoxified text: <detoxified text>
non-toxicity: 0.9567
fluency: 0.9672
similarity: 0.7729
```

Prediction on lines from [predict_texts.txt](src/models/predict_texts.txt) (you can insert your own lines)

```shell
cd ./TextDetoxification/src/models
python3 predict_model.py
```

Prediction on your own text
```shell
cd ./TextDetoxification/src/models
python3 predict_model.py -t "<your toxic input>"
```
