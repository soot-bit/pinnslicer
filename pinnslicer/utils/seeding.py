# ----------------------------------------------------------------------------
# Description: one place to seed every random number generator this package
#              uses, so that a run can be reproduced.
#
# Three generators are in play:
#
#   random        the Python standard library generator,
#   numpy         used by UniformSample and by the training notebook,
#   torch         used for weight initialisation, shuffling and subsampling.
#
# A fourth, the Sobol scrambler in scipy.stats.qmc, does NOT read the global
# numpy seed: it builds its own np.random.default_rng() unless a seed is
# handed to it. SobolSample therefore takes its own `seed` argument; pass one
# whenever the training points have to be reproducible.
# ----------------------------------------------------------------------------
import random

import numpy as np
import torch
# ----------------------------------------------------------------------------
def set_seed(seed, deterministic=False, verbose=1):
    """
    Seed the Python, numpy and torch generators.

    Parameters:
        seed (int): the seed.
        deterministic (bool): also ask torch for deterministic algorithms and
            switch off the cuDNN autotuner. Slower, and only meaningful on a
            GPU; a run can still differ across torch versions and hardware.
        verbose (int): print what was seeded.

    Returns:
        seed (int): the seed, so it can be stored in the run configuration:

            config('seed', set_seed(config('seed')))
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"/!\\ seed must be an int, got {type(seed).__name__}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

    if verbose:
        print(f"  Seeded random, numpy and torch with seed = {seed}")
        print(f"  (Sobol sampling is seeded separately: SobolSample(seed={seed}))")

    return seed
# ----------------------------------------------------------------------------
def torch_generator(seed=None):
    """
    Returns a torch.Generator seeded with `seed`, or None if seed is None.

    A local generator keeps a draw (a shuffle, a subsample) reproducible
    without disturbing the global torch generator that drives training.
    """
    if seed is None:
        return None

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"/!\\ seed must be an int, got {type(seed).__name__}")

    return torch.Generator().manual_seed(seed)
