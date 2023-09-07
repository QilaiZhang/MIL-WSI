import os
from utils import load_callbacks
from model import TransMIL
from dataset import ProstateDataModule

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger


if __name__ == '__main__':
    results_dir = "./results"
    exp_name = "TransMIL_results"
    fold = 'fold_0'

    # load data
    data = ProstateDataModule(data_dir="/data_sdb/PRAD/feature/10x_resnet50_overlap/",
                              csv_dir="/data_sdb/PRAD/csv/label.csv")

    # load model
    model = TransMIL(n_classes=2)

    # load loggers
    csv_logger = CSVLogger(save_dir=results_dir, name=exp_name, version=fold)
    logger = [csv_logger]

    # load callbacks
    callbacks = load_callbacks(dirpath=os.path.join(results_dir, exp_name, fold))

    # load trainer
    trainer = Trainer(devices=[2], enable_progress_bar=True, num_sanity_val_steps=0, max_epochs=5,
                      accumulate_grad_batches=2, precision='16-mixed', logger=logger, callbacks=callbacks)
    trainer.fit(datamodule=data, model=model)
    trainer.test(datamodule=data, model=model)
