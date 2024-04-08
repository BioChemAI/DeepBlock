import argparse
import re
import sys
from typing import Dict
from ..utils import return_exception, time_limit

class RetroStarPlanner:

    opt_args = ('device', 'iterations', 'expansion_topk')

    @classmethod
    def argument_group(cls, group: argparse._ArgumentGroup):
        group.add_argument("--device", type=str, default='cpu')
        group.add_argument("--iterations", type=int, default=100)
        group.add_argument("--expansion-topk", type=int, default=50)
        return group

    def __init__(self, device: str='cpu', iterations: int=100, expansion_topk: int=50) -> None:
        """Secure loading load Retro Star planner
        
        device: 'cpu', 'cuda', 'cuda:0', etc. (According to experience, cuda has no acceleration effect)
        """
        old_argv = sys.argv
        if device == "cpu":
            gpu = -1
            sys.argv = old_argv[:1]
        elif device == "cuda":
            gpu = 0
            sys.argv = old_argv[:1] + ['--gpu', str(gpu)]
        else:
            m = re.search(r"^cuda:(?P<id>\d+)$", device)
            if m is not None:
                gpu = int(m['id'])
                sys.argv = old_argv[:1] + ['--gpu', str(gpu)]
            else:
                raise ValueError(f"Unknow device: {device}")
            
        try:
            from retro_star.api import RSPlanner # pyright: ignore[reportMissingImports]
        except ImportError as err:
            raise Exception(f"Please refer to "
                            f"https://github.com/binghong-ml/retro_star "
                            f"to configure the environment of Retro Star\n{repr(err)}")
        self.planner = RSPlanner(
            gpu=gpu,
            use_value_fn=True,
            iterations=iterations,
            expansion_topk=expansion_topk
        )
        sys.argv = old_argv
        
    @time_limit(120)
    def __call__(self, smi: str) -> Dict:
        return self.planner.plan(smi)
    