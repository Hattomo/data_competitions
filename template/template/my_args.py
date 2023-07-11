# -*- coding: utf-8 -*-

import os
import shutil
import multiprocessing
import argparse

def get_parser(time: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Set parameter')
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--dir', default=current_file_path, type=str, help='directory')
    parser.add_argument('--debug', default=False, type=bool, help="debug mode")
    parser.add_argument('--message', default='wav2vec2', type=str, help="message")
    # general
    parser.add_argument('--workers', default=int(multiprocessing.cpu_count() / 2), type=int, help="使用するCPUコア数")
    parser.add_argument('--benchmark', default=True, type=bool, help="torch cudnn benchmark")
    parser.add_argument('--multi-gpu', default=False, type=bool, help="multi gpu")
    parser.add_argument('--amp', default=False, type=bool, help="amp")
    # Dataset path
    parser.add_argument('--dataset', default='datasets/commonvoice_7_ja.py', type=str, help="train audio PATH")
    parser.add_argument('--train_audio_path',
                        default='../mozira/ready_yahoo/train.csv',
                        type=str,
                        help="train audio PATH")
    parser.add_argument('--valid_audio_path',
                        default='../mozira/ready_yahoo/test.csv',
                        type=str,
                        help="valid audio PATH")
    parser.add_argument('--test_audio_path', default='../mozira/ready_yahoo/dev.csv', type=str, help="test audio PATH")

    # result path
    parser.add_argument('--checkpoint', default=f'build/{time}/checkpoint/', type=str, help="checkpoint path")
    parser.add_argument('--tensorboard', default=f'build/{time}/tensorboard/', type=str, help="tensorboard path")
    parser.add_argument('--logger-config-path', default=f'assets/log_config.json', type=str, help="logger config path")
    parser.add_argument('--token', default=f"build/{time}/token.json", type=str, help="token path")
    parser.add_argument('--base-path', default=f"build/{time}/", type=str, help="logger build path")
    parser.add_argument('--args_path', default=f"build/{time}/args.json", type=str, help="args path")
    # model
    parser.add_argument('--batch-size', default=4, type=int, help="batch size")
    # training
    parser.add_argument('--train-size', default=45000, type=int, help="train dataset size")
    parser.add_argument('--end_epoch', default=100, type=int, help="epoch")
    parser.add_argument('--start_epoch', default=0, type=int, help="start epoch")
    parser.add_argument('--patience', default=70, type=int, help="patience")
    parser.add_argument('--resume', default='', type=str, help="resume checkpoint path")
    # Optimizer
    parser.add_argument('--lr', default=1e-4, type=float, help="学習率")
    # other
    parser.add_argument('--seed', default=0, type=int, help="randomのseed")
    parser.add_argument('--device', default="cuda:1", help="device")
    parser.add_argument('--line', default="", type=str, help="line")
    return parser

def set_model_parameters(model, opts, logger) -> None:
    """
    Set up turning parameters for model
    """

    for param in list(model.state_dict().keys()):
        logger.info(param)
    # off
    for param in model.parameters():
        param.requires_grad = False

    # on
    for param in model.lm_head.parameters():
        param.requires_grad = True
    for param in model.hubert.encoder.layers[-1].parameters():
        param.requires_grad = True
    for param in model.hubert.encoder.layers[-2].parameters():
        param.requires_grad = True
    for param in model.hubert.encoder.layers[-3].parameters():
        param.requires_grad = True
    for param in model.hubert.encoder.layers[-4].parameters():
        param.requires_grad = True


def set_debug_mode(opts: argparse.Namespace, log_conf: dict) -> None:
    """

    Turn on debug mode
    Don't make new tensorboard / checkpoint

    """
    shutil.rmtree("build/debug", ignore_errors=True)
    opts.base_path = "build/debug"
    opts.tensorboard = "build/debug/tensorboard"
    opts.checkpoint = "build/debug/checkpoint"
    opts.token = "build/debug/token.json"
    opts.args_path = "build/debug/args.json"
    opts.batch_size = 4
    os.makedirs(opts.tensorboard, exist_ok=True)
    os.makedirs(opts.checkpoint, exist_ok=True)
    opts.end_epoch = 1000
    log_conf["handlers"]["fileHandler"]["filename"] = f'build/debug/progress.log'
    log_conf["handlers"]["result_fileHandler"]["filename"] = f'build/debug/result.log'

def set_release_mode(opts: argparse.Namespace, log_conf: dict) -> None:
    """

    Turn on release mode
    Make new tensorboard / checkpoint

    """
    os.makedirs(opts.base_path, exist_ok=True)
    log_conf["handlers"]["fileHandler"]["filename"] = f'{opts.base_path}/progress.log'
    log_conf["handlers"]["result_fileHandler"]["filename"] = f'{opts.base_path}/result.log'
