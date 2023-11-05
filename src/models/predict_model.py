# Import necessary module and function
from detoxificaton_model import DetoxificationModel
import argparse


# Define the main function to be executed
def main(args):
    # Initialize the DetoxificationModel with model and tokenizer names
    detoxification_model = DetoxificationModel(model_name=args.model_name, tokenizer_name=args.tokenizer_name)

    # Check if text input is provided directly or from a file
    if args.text:
        # Use the provided text as input
        texts = [args.text]
    else:
        # Read text from a file and remove any leading/trailing whitespace
        with open(args.file, 'r') as f:
            texts = f.readlines()
            texts = [text.strip() for text in texts]

    # Predict detoxified outputs for the input texts
    results = detoxification_model.predict(texts)

    # Add space for readability
    for _ in range(5):
        print()
    # Print a separator line
    print('-' * 50)

    # Print original text, detoxified text, and associated scores for each result
    for original_text, result in zip(texts, results):
        print(f"Original text: {original_text}")
        print(f"Detoxified text: {result['output']}")
        for key, value in result['scores'].items():
            print(f"{key}: {value:.4f}")
        print()


# Check if the script is executed directly (not imported as a module)
if __name__ == "__main__":
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser()

    # Define model arguments
    parser.add_argument('--model_name', type=str, default="IGragon/T5-detoxification")
    parser.add_argument('--tokenizer_name', type=str, default="Vamsi/T5_Paraphrase_Paws")

    # Define input source arguments, either a file or direct text input
    parser.add_argument('--file', type=str, default="./predict_texts.txt")
    parser.add_argument('-t', '--text', type=str, default="")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args)
