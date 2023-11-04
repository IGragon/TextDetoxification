from detoxificaton_model import DetoxificationModel

import argparse


def main(args):
    detoxification_model = DetoxificationModel(model_name=args.model_name, tokenizer_name=args.tokenizer_name)

    if args.text:
        texts = [args.text]
    else:
        with open(args.file, 'r') as f:
            texts = f.readlines()
            texts = [text.strip() for text in texts]

    results = detoxification_model.predict(texts)

    for _ in range(5):
        print()
    print('-' * 50)

    for original_text, result in zip(texts, results):
        print(f"Original text: {original_text}")
        print(f"Detoxified text: {result['output']}")
        for key, value in result['scores'].items():
            print(f"{key}: {value:.4f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="IGragon/T5-detoxification")
    parser.add_argument('--tokenizer_name', type=str, default="Vamsi/T5_Paraphrase_Paws")

    parser.add_argument('--file', type=str, default="./predict_texts.txt")
    parser.add_argument('-t', '--text', type=str, default="")

    args = parser.parse_args()
    main(args)
