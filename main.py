from argparse import ArgumentParser

from preprocess_data import preprocess_data_pipeline
from model import model_pipeline


def main(args) -> None:
    """
    Main function to train a model to predict if a client will subscribe to a term deposit.
    """

    # Load data
    data = preprocess_data_pipeline(load_data_from_local=args.load_data_from_local)

    # Train model
    model_pipeline(data=data,
                   save_model=args.save_model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-model", action='store_true', help="Save trained model.")
    parser.add_argument("--load-data-from-local", action='store_true', help="Load data from local.")
    args = parser.parse_args()

    main(args)
