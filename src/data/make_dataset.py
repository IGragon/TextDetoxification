import pandas as pd
from tqdm import tqdm

# This code reads and processes data related to toxicity,
# creates training and evaluation datasets, and saves the results to files.

raw_data = pd.read_csv("../../data/raw/filtered.tsv", delimiter="\t")

low_toxic = []
high_toxic = []
low_score = []
high_score = []
for sent1, sent2, score1, score2 in tqdm(raw_data[['reference', 'translation', 'ref_tox', 'trn_tox']].values):
    if score1 > score2:
        sent1, sent2 = sent2, sent1
        score1, score2 = score2, score1

    low_toxic.append(sent1)
    high_toxic.append(sent2)
    low_score.append(score1)
    high_score.append(score2)

raw_data['tox_low'] = low_toxic
raw_data['tox_high'] = high_toxic
raw_data['score_low'] = low_score
raw_data['score_high'] = high_score
raw_data['tox_diff'] = raw_data['score_high'] - raw_data['score_low']

raw_data.to_csv('../../data/interm/high_low_tox.tsv', sep='\t', index=False)
print("Saved intermediate data to '../../data/interm/high_low_tox.tsv', shape:", raw_data.shape)

high_toxicity_gap = raw_data[raw_data['tox_diff'] > 0.9][['tox_high', 'tox_low']]

train_part = high_toxicity_gap.iloc[: int(high_toxicity_gap.shape[0] * 0.8)]
eval_part = high_toxicity_gap.iloc[int(high_toxicity_gap.shape[0] * 0.8):]

train_part.to_csv('../../data/interm/train.tsv', sep='\t', index=False)
eval_part.to_csv('../../data/interm/eval.tsv', sep='\t', index=False)

print("Saved intermediate data to '../../data/interm/train.tsv', shape:", train_part.shape)
print("Saved intermediate data to '../../data/interm/eval.tsv', shape:", eval_part.shape)

print("Done with making training datasets!")

toxicity_dataset = pd.read_csv("../../data/external/toxicity_en.csv")

only_toxic = toxicity_dataset[toxicity_dataset['is_toxic'] == 'Toxic'][['text']]
only_toxic.to_csv('../../data/external/toxicity_en.csv', index=False)

print("Saved toxic data to '../../data/external/toxicity_en.csv', shape:", only_toxic.shape)

print("Done with making toxicity dataset!")
