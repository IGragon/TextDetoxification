## data/raw/filtered.tsv

The dataset is a subset of the ParaNMT corpus (50M sentence pairs). The filtered ParaNMT-detox corpus (500K sentence pairs) can be downloaded from [here](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip). This is the main dataset for the assignment detoxification task.

The data is given in the `.tsv` format, means columns are separated by `\t` symbol.

| Column | Type | Description | 
| ----- | ------- | ---------- |
| reference | str | First item from the pair | 
| ref_tox | float | toxicity level of reference text | 
| translation | str | Second item from the pair - paraphrazed version of the reference|
| trn_tox | float | toxicity level of translation text |
| similarity | float | cosine similarity of the texts |
| lenght_diff | float | relative length difference between texts |

## data/interm/high_low_tox.tsv

The dataset is a result of transforming _data/raw/filtered.tsv_ to the following format

| Column | Type | Description
|--------|------|------------
| similarity | float | cosine similarity of the texts |
| lenght_diff | float | relative length difference between texts |
|tox_diff| float | difference between high toxicity example and low toxicity example
|tox_low| str | least toxic sample from pair
|tox_high| str | most toxic sample from pair
|score_low| float | toxicity level of the least toxic sample
|score_high| float | toxicity level of the most toxic sample

## data/external/toxicity_en.csv
https://github.com/surge-ai/toxicity/blob/main/toxicity_en.csv

Dataset with toxic and non-toxic comments. It will be primarily used to train toxicity classifier.

| Column | Type | Description |
|--------|------|-------------|
|text|str|comment sample|
|is_toxic|str|Whether the comment is toxic or not|

