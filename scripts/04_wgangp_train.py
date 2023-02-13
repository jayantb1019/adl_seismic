import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time 
from threading import Thread

from datetime import datetime
from rich import traceback


import pytorch_lightning as pl 
from pytorch_lightning import Trainer 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler

# from pl_bolts.callbacks import TrainingDataMonitor, ModuleDataMonitor, BatchGradientVerificationCallback

from datamodules.mnist import MNISTDataModule
from models.wgan_gp import WGANGP

checkpoints_dir = '.'

def main() : 
    pl.seed_everything(42)
    traceback.install()

    datamodule = MNISTDataModule()
    model = WGANGP(*datamodule.dims)

    # logger 
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    experiment_version = f"w_gangp_{timestamp}"
    logger = TensorBoardLogger('lightning_logs', name='gan', log_graph=False, version = experiment_version) # log_graph = True throws an mps error - float64 not supported

    # profiler 
    profiler = AdvancedProfiler(dirpath = 'wgan_profiler', filename='logs')

    #callbacks 
    modelSummaryCb = RichModelSummary(max_depth=-1)
    tqdmProgressCb = TQDMProgressBar(refresh_rate=20)
    # datamonitorCb = TrainingDataMonitor(log_every_n_steps=50) # monitors input data 
    # activationsMonitorCb = ModuleDataMonitor(submodules=True) # monitors layer outputs , monitor the output of generator
    # batchMixingCb =  BatchGradientVerificationCallback() # verifies if the model is manipulating data in the batch dimension 

    trainer = Trainer(
        accelerator='mps',
        devices = 1, 
        max_epochs=250, 
        callbacks = [modelSummaryCb, tqdmProgressCb, 
                    # datamonitorCb,
                    # activationsMonitorCb,
                    # batchMixingCb
                    ],
        default_root_dir=checkpoints_dir, 
        logger = logger, 
        profiler=profiler,
        # resume_from_checkpoint='/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments/lightning_logs/gan/dc_gan_04_11_2022_10_32_01/checkpoints/epoch=41-step=18060.ckpt'
    )

    x = trainer.fit(model, datamodule)

def run_tensorboard() : 
    return os.system("tensorboard --logdir='lightning_logs' --bind_all")

if __name__ == '__main__' :
    # run tensorboard
    # thread = Thread(target=run_tensorboard)
    # thread.start()
    
    # wait for the thread to spawn 
    # time.sleep(2)

    # run main function
    main()