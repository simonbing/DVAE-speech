#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""

import sys
from dvae import LearningAlgorithm


if __name__ == '__main__':

    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        learning_algo = LearningAlgorithm(config_file=cfg_file)
        learning_algo.train()
        learning_algo.generate_synth(batch_size=10, seq_len=25)
    else:
        print('Error: Please indiquate config file')


