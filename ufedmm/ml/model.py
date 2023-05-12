import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping, Callback
from pytorch_lightning.loggers import NeptuneLogger

from network import FE_network
import numpy as np
import torch
from typing import List, Callable, Tuple

class Estimator(pl.LightningModule):
    r"""
    A CV model estimator which can be fit to data optimizing a coarse auto encoder model.

    Parameters
    ----------
    net : CV_net
        The Auto-encoder net which maps the features to the CV space and back onto the feature space.
    neptune_logger : NeptuneLogger
        For logging the experiment
    accelerator : str, optional
        The accelerator to use, by default 'gpu'
    devices : int, optional
        The number of devices to use, by default 1
    lr : float, optional
        The learning rate, by default 1e-3
    weight_decay : float, optional
        The weight decay, by default 1e-1
    """

    def __init__(
        self,
        net: FE_network,
        neptune_logger: NeptuneLogger,
        accelerator: str ='gpu',
        devices: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1e-1,
        norm: str = 'L2',
    ):
        super().__init__()
        self.accelerator=accelerator
        self.devices=devices
        self.net = net
        self.lr=lr
        self.weight_decay=weight_decay
        self.neptune_logger=neptune_logger
        self.norm = norm
        self.optimizer = None
        self.validation_step_loss = []
        print('Created estimator successfully!')

    def training_step(self, batch, batch_idx):
        r"""Performs a partial fit on data. This does not perform any batching.
        Parameters
        ----------
        batch : list
            The training data to feed through all the CVs, weights, and target data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        loss_value : torch.Tensor
        """
        target_forces = batch[1]
        pred_forces = self.net.forward(batch[0])
        loss_value = auto_loss(pred_forces, target_forces, norm=self.norm)
        loss_item = detach_numpy(loss_value)
        self.log("train_loss", float(loss_item))
        return loss_value
    
    def validation_step(self, batch, batch_idx):
        r"""Performs validation on data. This does not perform any batching.
        
        Parameters
        ----------
        batch : list
            The validation data to feed through all the CVs, weights, and target data.
        batch_idx : int
            The index of the batch.
        
        Returns
        -------
        loss_value : np.ndarray
            The total loss value
        loss_enc_item : np.ndarray
            The loss value of the autoencoder
        err_item_list : List[np.ndarray]
            The loss value for the mask weights.

        """
        torch.set_grad_enabled(True)
        target_forces = batch[1]
        pred_forces = self.net.forward(batch[0])
        loss_value = auto_loss(pred_forces, target_forces, norm=self.norm)
        loss_item = detach_numpy(loss_value)
        self.validation_step_loss.append(loss_item)
        return loss_item

    def on_validation_epoch_end(self):
        r"""Performs mean estimation and logging of validation scores at the end of epoch
        
        """
        loss_mean = np.mean(self.validation_step_loss)
        self.log("val_loss", loss_mean)
        self.validation_step_loss.clear()
        return loss_mean

    def configure_optimizers(self):
        r"""Configures the optimizer. This is called by pytorch lightning, so to use one optimizer for the whole pipeline.
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            # self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return self.optimizer
    
    def set_learning_rate(self, lr: float):
        r"""Sets the learning rate of the optimizer.
        
        Parameters
        ----------
        lr : float
            The learning rate to set.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def save(self, path: str):
        r"""Save the current estimator at path.

        Parameters
        ----------
        path: str
            The path where to save the model.

        """
        save_dict = {
            # "step": self._step,
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
        }
        torch.save(save_dict, path)

    def load(self, path: str):
        r"""Load the estimator from path.
         The architecture needs to fit!

        Parameters
        ----------
        path: str
             The path where the model is saved.
        """

        checkpoint = torch.load(path, map_location=self.device)
        if self.optimizer is None:
            self.configure_optimizers()
        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # self._step = checkpoint["step"]

def auto_loss(x, y, norm='L2'):
    diff = x+y # plus because gradient of A is negative forces
    if norm=='L2':
        loss = torch.mean(torch.sum((diff * diff), dim=1))
    else:
        loss = torch.sum(torch.abs(diff))
    return loss
def detach_numpy(x):
    return x.detach().cpu().numpy()