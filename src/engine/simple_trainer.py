import torch
from tqdm import tqdm
import numpy as np
from .build import TRAINER_REGISTRY
from torch.utils.tensorboard import SummaryWriter
import time


class Trainer:

    def train(self, data_loader, model, loss_func, optimizer, epoch_idx):
        pass

    def valid(self, data_loader, model, loss_func, epoch_idx):
        pass

    def test(self, data_loader, model):
        pass


@TRAINER_REGISTRY.register()
class SimpleTrainer(Trainer):
    """base trainer for bi-modal input"""
    def __init__(self, cfg, collector, logger):
        self.cfg = cfg
        self.clt = collector
        self.logger = logger
        # self.tb_writer = SummaryWriter(cfg.OUTPUT_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, data_loader, model, loss_func, optimizer, epoch_idx):
        lr = optimizer.param_groups[0]['lr']
        self.logger.info(f"Training: learning rate:{lr}")
        # self.tb_writer.add_scalar("lr", lr, epoch_idx)

        model.train()
        acc_avg_list, loss_list = [], []
        for i, data in enumerate(data_loader):
            inputs, labels = self.data_fmt(data)
            outputs = model(*inputs)
            optimizer.zero_grad()
            loss = loss_func(outputs.cpu(), labels.cpu())
            # self.tb_writer.add_scalar("loss", loss.item(), i)
            loss.backward()
            optimizer.step()

            acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean().clip(min=0)
            acc_avg = acc_avg.detach().numpy()
            acc_avg_list.append(acc_avg)
            loss_list.append(loss.item())

            # print loss and training info for an interval
            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                self.logger.info(
                    "Train: Epo[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] LOSS: {:.4f} ACC:{:.4f} ".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,   # Epo
                        i + 1, len(data_loader),                     # Iter
                        float(loss.item()), float(acc_avg),  # LOSS ACC ETA
                    )
                )

        self.clt.record_train_loss(loss_list)
        self.clt.record_train_acc(acc_avg_list)

    def valid(self, data_loader, model, loss_func, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            acc_batch_list = []
            acc_epoch = []
            for i, data in enumerate(data_loader):
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)
                loss = loss_func(outputs.cpu(), labels.cpu())
                loss_batch_list.append(loss.item())

                ocean_acc_batch = (
                    1 - torch.abs(outputs.cpu().detach() - labels.cpu().detach())
                ).mean(dim=0).clip(min=0)
                acc_epoch.append(ocean_acc_batch)
                acc_batch_avg = ocean_acc_batch.mean()
                acc_batch_list.append(acc_batch_avg)
            acc = torch.stack(acc_epoch, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            acc_avg = acc.mean()
            # self.tb_writer.add_scalar("valid_acc", ocean_acc_avg, epoch_idx)
        self.clt.record_valid_loss(loss_batch_list)
        self.clt.record_valid_acc(acc_batch_list)  # acc over batches
        self.clt.record_valid_ocean_acc(acc)
        if acc_avg > self.clt.best_valid_acc:
            self.clt.update_best_acc(acc_avg)
            self.clt.update_model_save_flag(1)
        else:
            self.clt.update_model_save_flag(0)

        self.logger.info(
            "Valid: Epoch[{:0>3}/{:0>3}] Train Mean_Acc: {:.2%} Valid Mean_Acc:{:.2%} OCEAN_ACC:{}\n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH,
                float(self.clt.epoch_train_acc),
                float(self.clt.epoch_valid_acc),
                self.clt.valid_ocean_acc)
        )

    def test(self, data_loader, model):
        mse_func = torch.nn.MSELoss(reduction="none")
        model.eval()
        with torch.no_grad():
            mse_ls = []
            acc = []
            label_list = []
            output_list = []
            for data in tqdm(data_loader):
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)  # for more then one input data

                outputs = outputs.cpu().detach()
                labels = labels.cpu().detach()
                output_list.append(outputs)
                label_list.append(labels)
                mse = mse_func(outputs, labels).mean(dim=0)
                ocean_acc_batch = (1 - torch.abs(outputs - labels)).mean(dim=0).clip(min=0)
                mse_ls.append(mse)
                acc.append(ocean_acc_batch)
            ocean_mse = torch.stack(mse_ls, dim=0).mean(dim=0).numpy()
            acc = torch.stack(acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            mse_mean = ocean_mse.mean()
            acc_avg = acc.mean()

        mse_mean_rand = np.round(mse_mean, 4)
        acc_avg_rand = np.round(acc_avg.astype("float64"), 4)
        # self.tb_writer.add_scalar("test_acc", ocean_acc_avg_rand)
        return acc_avg_rand,  mse_mean_rand

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        data_in, labels = data["data"], data["label"]
        return (data_in,), labels


