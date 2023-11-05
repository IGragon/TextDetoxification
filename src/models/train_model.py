# Import necessary modules and functions
from utils import load_csv_dataset
from detoxificaton_model import DetoxificationModel

# Import argparse for command-line argument handling
import argparse


# Define the main function to be executed
def main(args):
    # Initialize the DetoxificationModel with model and tokenizer names
    detoxification_model = DetoxificationModel(model_name=args.model_name, tokenizer_name=args.tokenizer_name)

    # Load the training dataset from a CSV file
    dataset = load_csv_dataset('../../data/interm/train.tsv')

    # Add space for readability
    for _ in range(5):
        print()

    # Print the training arguments to the console
    dict_args = vars(args)
    print("Starting training with the following arguments:")
    for key, value in dict_args.items():
        print(f"{key}: {value}")

    # Start the training process
    print("Training...")
    detoxification_model.train(dataset, args)


# Check if the script is executed directly (not imported as a module)
if __name__ == "__main__":
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser()

    # Define model arguments
    parser.add_argument('--model_name', type=str, default="Vamsi/T5_Paraphrase_Paws")
    parser.add_argument('--tokenizer_name', type=str, default="Vamsi/T5_Paraphrase_Paws")

    # Define training arguments
    parser.add_argument('--save_name', type=str, default="IGragon/T5-detoxification")
    parser.add_argument('--store_locally', action='store_true')
    parser.add_argument('--output_dir', type=str, default="./results")
    parser.add_argument('--evaluation_strategy', type=str, default="epoch")
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--num_train_epochs', type=int, default=2)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args)
