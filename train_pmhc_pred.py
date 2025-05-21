# Training p-MHC presentation prediction model
import warnings
import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import setting_init
from utils.utils import set_random_seed
from model.classifier.pmhc_pred_model import PMHCPredModel
from utils.data.pmhc_datamodule import PMHCPredDataModule

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    set_random_seed(2024)
    args = setting_init()
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    args.out_checkpoint_dir = os.path.join(args.out_checkpoint_dir, 'pmhc_pred_model', current_date_time)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_checkpoint_dir,
        monitor='valid_loss',
        filename='ckpt-{epoch:02d}-{valid_loss:.4f}-{val_auroc:.4f}-{val_aupr:.4f}',
        save_top_k=args.save_top_k,
        mode='min',
    )

    if args.gpus == '-1': gpu_num = -1
    else: gpu_num = list(map(int, args.gpus.split(',')))
    
    trainer = pl.Trainer(
        max_epochs=args.n_max_epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=args.gpus
    )

    model = PMHCPredModel(learning_rate=args.learning_rate)
    dataModule = PMHCPredDataModule(batch_size=args.batch_sizes)
    trainer.fit(model, dataModule)