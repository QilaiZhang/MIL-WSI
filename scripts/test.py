import sys
import argparse
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.trainer.states import TrainerStatus
from lightning.pytorch.loggers import CSVLogger
from milwsi.utils import read_yaml
from milwsi.datasets import build_dataset
from milwsi.model import build_model
from milwsi.utils.callbacks import load_callbacks


def parse_options():
    parser = argparse.ArgumentParser(description='Train MIL model')
    parser.add_argument('--opt', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--folds', type=int, default=[1])
    args = parser.parse_args()
    opt = read_yaml(args.opt)
    return args, opt


def main():
    args, opt = parse_options()

    seed_everything(args.seed, workers=True)

    datamodule = build_dataset(**opt['DataModule'])

    folds = args.folds if args.folds is not None else range(opt['DataModule']['n_splits'])

    for fold in folds:
        # set data for current fold
        datamodule.set_fold(fold)

        # create model
        model = build_model(**opt['ModelModule'], hparams=opt)

        # create callbacks for early stop
        callbacks = load_callbacks(**opt['Callbacks'])

        # create logger
        logger = CSVLogger(save_dir=opt['log_path'], name=opt['name'], version=fold)

        # create trainer
        trainer = Trainer(
            accelerator=args.accelerator,
            devices=[args.device],
            logger=logger,
            callbacks=callbacks,
            **opt['Trainer'],
        )

        # start training
        trainer.fit(datamodule=datamodule, model=model)

        if trainer.state.status == TrainerStatus.INTERRUPTED:
            sys.exit(0)

        # start testing
        trainer.test(datamodule=datamodule, model=model)

        if trainer.state.status == TrainerStatus.INTERRUPTED:
            sys.exit(0)


if __name__ == "__main__":
    main()
