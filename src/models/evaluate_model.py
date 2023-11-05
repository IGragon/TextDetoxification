# Import necessary modules and functions
from collections import defaultdict
from detoxificaton_model import DetoxificationModel
import argparse
import pandas as pd


# Define the main function to be executed
def main(args):
    # Initialize the DetoxificationModel with model and tokenizer names
    detoxification_model = DetoxificationModel(model_name=args.model_name, tokenizer_name=args.tokenizer_name)

    # Check if the evaluation should use an external toxicity dataset or an intermediate evaluation dataset
    if args.use_toxicity_dataset:
        # Load an external toxicity dataset and select text samples
        df = pd.read_csv('../../data/external/toxicity_en.csv')
        samples = df['text'].values
    else:
        # Load an intermediate evaluation dataset and select a random sample of 1000 'tox_high' texts
        df = pd.read_csv('../../data/interm/eval.tsv', sep='\t')
        samples = df.sample(1000)['tox_high'].values

    # Predict detoxified outputs for the selected samples
    results = detoxification_model.predict(samples)

    # Calculate mean scores for non-toxicity, fluency, and similarity
    scores = defaultdict(int)
    N = len(results)
    for result in results:
        for key, value in result['scores'].items():
            scores[key] += value
    for key, value in scores.items():
        scores[key] /= N

    # Add space for readability
    for _ in range(5):
        print()

    # Print a separator line and evaluation results
    print('-' * 50)
    print("Evaluation results!")
    print("Mean metrics:")
    for key, value in scores.items():
        print(f"{key}: {value:.4f}")


# Check if the script is executed directly (not imported as a module)
if __name__ == "__main__":
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser()

    # Define model arguments
    parser.add_argument('--model_name', type=str, default="IGragon/T5-detoxification")
    parser.add_argument('--tokenizer_name', type=str, default="Vamsi/T5_Paraphrase_Paws")

    # Define evaluation arguments to choose between toxicity dataset and evaluation dataset
    parser.add_argument('--use_toxicity_dataset', action='store_true')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args)
