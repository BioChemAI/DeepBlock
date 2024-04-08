import argparse
import logging

from deepblock.evaluation import DockingToolBox
from deepblock.utils import init_logging, mix_config


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Initialize the docking toolbox')
    group = parser.add_argument_group('Docking Toolbox')
    DockingToolBox.argument_group(group)
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    init_logging()
    logging.info(f'opt: {opt}')

    DockingToolBox(**{k: opt[k] for k in DockingToolBox.opt_args if k in opt})

    logging.info("Done!")
