# -*- coding: utf-8 -*-

from torch.utils.tensorboard import writer

from token_error_rate import TokenErrorRate

# 平均と現在の値を計算して保存するクラス
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LossManager(object):

    def __init__(self, mode: str, writer: writer):
        self.loss = AverageMeter()
        self.mode = mode
        self.writer = writer

    def update(self, loss, batch_size):
        self.loss.update(loss, batch_size)

    def write(self, epoch):
        self.writer.add_scalars(f"loss", {self.mode: self.loss.avg}, epoch)

class TokenErrorRateManager():
    """
    Calculate TER

    Func:
        reset() : initialize ter
        update(ter: TokenErrorRate) : update ter
        write(epoch: int) : write ter to tensorboard
    """

    def __init__(self, mode: str, writer) -> None:
        self.reset()
        self.writer = writer
        self.mode = mode

    def reset(self):
        self.total_error = AverageMeter()
        self.substitute_error = AverageMeter()
        self.delete_error = AverageMeter()
        self.insert_error = AverageMeter()
        self.len_ref = AverageMeter()

    def update(self, ter: TokenErrorRate, batch_size: int) -> None:
        self.total_error.update(ter.total_error, batch_size)
        self.substitute_error.update(ter.substitute_error, batch_size)
        self.delete_error.update(ter.delete_error, batch_size)
        self.insert_error.update(ter.insert_error, batch_size)
        self.len_ref.update(ter.len_ref, batch_size)

    def write(self, epoch: int) -> None:
        self.writer.add_scalars(f'ter/total_error', {self.mode: self.total_error.avg}, epoch)
        self.writer.add_scalars(f'ter/substitute_error', {self.mode: self.substitute_error.avg}, epoch)
        self.writer.add_scalars(f'ter/delete_error', {self.mode: self.delete_error.avg}, epoch)
        self.writer.add_scalars(f'ter/insert_error', {self.mode: self.insert_error.avg}, epoch)
        self.writer.add_scalars(f'ter/len_ref', {self.mode: self.len_ref.avg}, epoch)

class DataManager():

    def __init__(self, mode: str, writer: writer) -> None:
        self.loss_manager = LossManager(mode, writer)
        self.acc_manager = TokenErrorRateManager(mode, writer)
        self.writer = writer

    def update_loss(self, loss: float, batch_size: int) -> None:
        self.loss_manager.update(loss, batch_size)

    def update_acc(self, acc: TokenErrorRate, batch_size: int) -> None:
        self.acc_manager.update(acc, batch_size)

    def reset(self) -> None:
        self.loss_manager.reset()
        self.acc_manager.reset()

    def write(self, epoch: int) -> None:
        self.acc_manager.write(epoch)
        self.loss_manager.write(epoch)
