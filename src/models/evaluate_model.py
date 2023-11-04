from collections import defaultdict

from utils import load_csv_dataset
from detoxificaton_model import DetoxificationModel

import argparse
import pandas as pd


def main(args):
    detoxification_model = DetoxificationModel(model_name=args.model_name, tokenizer_name=args.tokenizer_name)
    df = pd.read_csv('../../data/interm/eval.tsv', sep='\t')
    samples = df.sample(50)['tox_high'].values
    results = detoxification_model.predict(samples)

    scores = defaultdict(int)
    N = len(results)
    for result in results:
        for key, value in result['scores'].items():
            scores[key] += value
    for key, value in scores.items():
        scores[key] /= N

    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--model_name', type=str, default="IGragon/T5-detoxification")
    parser.add_argument('--tokenizer_name', type=str, default="Vamsi/T5_Paraphrase_Paws")

    args = parser.parse_args()
    main(args)
