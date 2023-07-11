# -*- coding: utf-8 -*-

import os

import torch

from dockhub import dockhub

args = dockhub.args

# save model parameters
def save_checkpoint(state, dir_path):
    filename = os.path.join(dir_path, 'epoch%03d_val%.3f.pth' % (state['epoch'], state['best_val']))
    torch.save(state, filename)

# load saved model parameterts
def load_checkpoint(best_epoch, best_val, model) -> None:
    filename = os.path.join(args.checkpoint, 'epoch%03d_val%.3f.pth' % (best_epoch, best_val))
    model.load_state_dict(torch.load(filename)['model_state_dict'])
