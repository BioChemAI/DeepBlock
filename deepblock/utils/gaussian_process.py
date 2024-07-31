"""
https://botorch.org/tutorials/vae_mnist
"""

import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize, Normalize
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement

def fitted_model_optimize_acqf(train_x, train_obj, state_dict,
                               q: int, num_restarts: int, raw_samples: int,
                               bounds: Tensor) -> tuple[SingleTaskGP, Tensor]:
    
    d = train_x.shape[-1]

    # initialize and fit model
    model = SingleTaskGP(
        train_X=normalize(train_x, bounds), 
        train_Y=train_obj,
        outcome_transform=Standardize(m=1)
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)
    fit_gpytorch_mll(mll)

    qEI = qExpectedImprovement(
        model=model, best_f=train_obj.max()
    )

    candidates, _ = optimize_acqf(
        acq_function=qEI,
        bounds=torch.stack([torch.zeros(d), torch.ones(d)]),
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    candidates = unnormalize(candidates.detach(), bounds=bounds)
    return model, candidates
