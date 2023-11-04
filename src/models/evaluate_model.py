from collections import defaultdict

from detoxificaton_model import DetoxificationModel

import argparse
import pandas as pd


def main(args):
    detoxification_model = DetoxificationModel(model_name=args.model_name, tokenizer_name=args.tokenizer_name)

    if args.use_toxicity_dataset:
        df = pd.read_csv('../../data/external/toxicity_en.csv')
        samples = df['text'].values
    else:
        df = pd.read_csv('../../data/interm/eval.tsv', sep='\t')
        samples = df.sample(1000)['tox_high'].values
    results = detoxification_model.predict(samples)

    scores = defaultdict(int)
    N = len(results)
    for result in results:
        for key, value in result['scores'].items():
            scores[key] += value
    for key, value in scores.items():
        scores[key] /= N

    for _ in range(5):
        print()

    print('-' * 50)
    print("Evaluation results!")
    print("Mean metrics:")
    for key, value in scores.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--model_name', type=str, default="IGragon/T5-detoxification")
    parser.add_argument('--tokenizer_name', type=str, default="Vamsi/T5_Paraphrase_Paws")

    # evaluation arguments
    parser.add_argument('--use_toxicity_dataset', action='store_true')

    args = parser.parse_args()
    main(args)
