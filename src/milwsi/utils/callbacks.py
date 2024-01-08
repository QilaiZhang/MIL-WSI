from tqdm import tqdm
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar


class ProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        return tqdm(disable=True)

    def init_test_tqdm(self):
        return tqdm(disable=True)

    def on_train_epoch_start(self, trainer: pl.Trainer, *_) -> None:
        print("")
        super().on_train_epoch_start(trainer)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        print("")

    def on_train_end(self, *_) -> None:
        print("")


def load_callbacks(earlystop=True, monitor='val_loss', patience=10):

    callbacks = [ProgressBar()]

    if earlystop:
        callbacks.append(
            EarlyStopping(monitor=monitor, patience=patience, verbose=True)
        )
        callbacks.append(
            ModelCheckpoint(monitor=monitor, save_weights_only=True)
        )

    return callbacks

