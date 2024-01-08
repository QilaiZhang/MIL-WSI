import argparse
from milwsi.datasets import build_dataset


def parse_options():
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--dataset', default='WSIDataset')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--split_path', type=str, default='')
    parser.add_argument('--n_splits', type=int, default='10')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--patient_level', action='store_true')
    parser.add_argument('--monte_carlo', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_options()

    dataset = build_dataset(name=args.dataset, path=args.data_path)

    build_dataset(
        name="CrossValidationModule",
        dataset=dataset,
        split_path=args.split_path,
        n_splits=args.n_splits,
        seed=args.seed,
        patient_level=args.patient_level,
        monte_carlo=args.monte_carlo,
    )


if __name__ == '__main__':
    main()
