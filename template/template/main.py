# -*- coding: utf-8 -*-

from datetime import datetime

from my_args import get_parser

start_time = datetime.utcnow().strftime("%y%m%d-%H%M%S")

# Load parameters
parser = get_parser(start_time)
opts = parser.parse_args()
