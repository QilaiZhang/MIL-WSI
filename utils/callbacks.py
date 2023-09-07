from tqdm import tqdm
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar


class LitProgressBar(TQDMProgressBar):
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

    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


def load_callbacks(dirpath):
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=True)
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_weights_only=True, dirpath=dirpath,
                                       filename='checkpoint')
    progress_bar = LitProgressBar()
    callbacks = [earlystop, model_checkpoint, progress_bar]
    return callbacks


