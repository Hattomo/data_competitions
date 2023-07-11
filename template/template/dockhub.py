# -*- coding: utf-8 -*-

from datetime import datetime
import os
import json
import platform
from logging import getLogger, config
import subprocess

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import psutil

import my_args

class DockHub():
    """
    configreation and global instance such as logger and Tensorboard writer
    """

    def __init__(self) -> None:
        """
        create argparser, logger and tensorboard writer instance

        self.logger (dict): logger instance
        self.writer (SummaryWriter): tensorboard writer instance
        self.args (argparse.Namespace): arguments instance
        self.device (torch.device): device instance
        """
        start_time = datetime.now().strftime("%y%m%d-%H%M%S")

        parser = my_args.get_parser(start_time)
        self.args = parser.parse_args()

        # Initalize logger
        with open(self.args.logger_config_path, 'r') as f:
            log_conf = json.load(f)

        # Set Device
        if self.args.device == '':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.args.device)

        # set debug mode
        if self.args.debug:
            my_args.set_debug_mode(self.args, log_conf)
        if not self.args.debug:
            my_args.set_release_mode(self.args, log_conf)

        # save args
        with open(self.args.args_path, 'wt') as f:
            json.dump(vars(self.args), f, indent=4)

        config.dictConfig(log_conf)
        progress_logger = getLogger("Progress")
        result_logger = getLogger("Result")
        self.logger = {"progress": progress_logger, "result": result_logger}

        # Init tensorboard
        self.writer = SummaryWriter(log_dir=self.args.tensorboard)

    def write_machine_info(self) -> None:
        """
        write machine information to logger
        """
        progress_logger = self.logger["progress"]
        branch = subprocess.run(["git", "branch", "--contains=HEAD"], encoding='utf-8', stdout=subprocess.PIPE)

        commit_hash = subprocess.run(["git", "rev-parse", "HEAD"], encoding='utf-8', stdout=subprocess.PIPE)

        with open(f"{self.args.base_path}/environment.yml", mode='w', encoding="utf-8") as f:
            conda_env = subprocess.run(["conda", "env", "export"], encoding='utf-8',stdout=subprocess.PIPE)
            f.write(conda_env.stdout)

        with open(f"{self.args.base_path}/diff.patch", mode='w', encoding="utf-8") as f:
            diff_patch = subprocess.run(["git", "diff", "--diff-filter=d","./"], encoding='utf-8',stdout=subprocess.PIPE)
            f.write(diff_patch.stdout)
            
        progress_logger.info(f"python     : {platform.python_version()}")
        progress_logger.info(f"Pytorch    : {torch.__version__}")
        progress_logger.info(f"cuda version: {torch.version.cuda}")
        progress_logger.info(f"cudnn version: {torch.backends.cudnn.version()}")

        system_configs = f"system config \n {platform.uname()}\n\
                        {psutil.cpu_freq()}\n\
                        cpu_count: {psutil.cpu_count()}\n\
                        memory_available: {psutil.virtual_memory().available}"

        progress_logger.info(system_configs)
        progress_logger.info(f"branch : " + branch.stdout)
        progress_logger.info(f"commit hash : " + commit_hash.stdout)

    def setup(self) -> None:
        """
        setup train environment
        """
        torch.manual_seed(self.args.seed)
        cudnn.benchmark = self.args.benchmark
        # Init checkpoint
        os.makedirs(self.args.checkpoint, exist_ok=True)

dockhub = DockHub()
