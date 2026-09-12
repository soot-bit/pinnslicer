<div align="center">

<pre style="font-size: 5pt; line-height: 1;">

            ███                                ████   ███                             
           ░░░                                ░░███  ░░░                              
 ████████  ████  ████████   ████████    █████  ░███  ████   ██████   ██████  ████████ 
░░███░░███░░███ ░░███░░███ ░░███░░███  ███░░   ░███ ░░███  ███░░███ ███░░███░░███░░███
 ░███ ░███ ░███  ░███ ░███  ░███ ░███ ░░█████  ░███  ░███ ░███ ░░░ ░███████  ░███ ░░░ 
 ░███ ░███ ░███  ░███ ░███  ░███ ░███  ░░░░███ ░███  ░███ ░███  ███░███░░░   ░███     
 ░███████  █████ ████ █████ ████ █████ ██████  █████ █████░░██████ ░░██████  █████    
 ░███░░░  ░░░░░ ░░░░ ░░░░░ ░░░░ ░░░░░ ░░░░░░  ░░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░░     
 ░███                                                                                 
 █████                                                                                
░░░░░                                                                              
</pre>

<p>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.9%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
  <a href="https://numpy.org/"><img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"></a>
  <a href="https://scipy.org/"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"></a>
</p>
</div>


## Introduction
This module can be used to train a Physics-Informed Neural Network (PINN) [1, 2] to solve the following nonlinear ordinary differential equation (ODE):
```math
\overset{\textstyle\cdot\cdot}{u}  \: + \: u - \: 3 \: \frac{u^2}{2}  =  0 ,
```
which describes the orbit of photons in a Schwarzschild spacetime about a spherically symmetric body of mass $M$.

### Notation
The variable of interest here is
```math
  u = \frac{r_s }{ r},
```
where $r_s$, the Schwarzschild radius, is defined as $r_s =  2 G M  /  c^2,$ where $G$ is Newton's gravitational constant and $c$ is the speed of light in vacuum. If $C$ is the proper circumference of a circle centered at the center of mass,
in a Schwarzschild spacetime, the radial coordinate is *defined by* $r \equiv C  /  (2\pi)$ and differs from the proper radial distance.
The overdot here ($\overset{\textstyle\cdot\cdot}{u}$) indicates differentiation with respect to $\phi$, the azimuthal angle in a spherical polar coordinate system, $(r, \theta, \phi)$. Here $\theta$ is set to $\pi  /  2$ without loss of generality.

The initial conditions are
```math
u\,(0) = u_0
```
and
```math
\overset{\textstyle\cdot}{u}\,(0) = v_0.
```

### Approach
The ODE is solved using a PINN following the approach in [3]. The neural network is described by the function $g_\beta(\phi, u_0, v_0),$ where $\beta$ are the network's trainable weights.  
  
We use the following Ansatz from the theory of connections (ToC) [4] that incorporates the initial conditions explicitly:

```math
    u(\phi, u_0, v_0)  = u_0 + g_\beta(\phi, u_0, v_0) - g_\beta(0, u_0, v_0) + \phi \left[ v_0 - \dot{g}_\beta(0, u_0, v_0) \right],
```
and
```math
    \dot{u}(\phi, u_0, v_0) = v_0 + \dot{g}_\beta(\phi, u_0, v_0) - \dot{g}_\beta(0, u_0, v_0),
```
so that the initial conditions hold by construction and the loss reduces to the ODE residual alone, with no weighting coefficients to tune.

### Slicing
The ODE is autonomous: it does not depend on $\phi$ explicitly, so the coordinate system can be rotated to any starting azimuthal angle. PINN-Slicer exploits this by training the network only on a thin slice $\phi \in [0, \Delta\phi]$, with $(u_0, v_0)$ Sobol-sampled over their range. The full orbit is then mapped out recursively: the solution $u(\Delta\phi)$ and its derivative $\dot{u}(\Delta\phi)$ at the end of one slice become the initial conditions of the next, until the photon escapes ($u \to 0$), crosses the event horizon ($u = 1$), or the requested $\phi$ range is reached. A single trained network therefore generalizes to any initial condition and to a solution domain that is not known a priori.

### References
[1] B. Moseley, [Deep Learning in Scientific Computing (2023)](https://camlab.ethz.ch/teaching/deep-learning-in-scientific-computing-2023.html), ETH Zürich, Computational and Applied Mathematics Laboratory (CAMLab)  
[2] S. Cuomo *et al*., *Scientific Machine Learning through Physics-Informed Neural Networks: Where we are and What's next*, [arXiv:2201.05624](https://doi.org/10.48550/arXiv.2201.05624)  
[3] Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, Michael W. Mahoney, *Characterizing possible failure modes in physics-informed neural networks*, NIPS'21: Proceedings of the 35th International Conference on Neural Information Processing Systems; [arXiv:2109.01050](https://arxiv.org/abs/2109.01050)  
[4] D. Mortari, *The Theory of Connections: Connecting Points*, Mathematics, vol. 5, no. 57, 2017.

## Package layout

```
pinnslicer/
    nn.py                  FCNN, the ToC Solution, the Objective (loss),
                           train_pinn, Config, save_checkpoint/load_checkpoint
    orbits/
        orbit_setup.py     boomerang orbit parameters, (r0, delta) -> (u0, v0)
        solvers.py         FDPhotonOrbitSolver, PinnSlicer, OrbitSolution,
                           SolverStatus
        plotting.py        OrbitPlotXY, OrbitPlotUPhi, figure builders, palette
    utils/
        data.py            Sobol/uniform sampling, Dataset, DataLoader
        loss_vs_dphi.py    minimum validation loss vs slice width
        monitoring.py      live cost curves during training
        seeding.py         set_seed: one call seeds python, numpy and torch
```

### Solving an orbit

Both solvers are callables with the same signature, and both return an
`OrbitSolution`:

```python
import numpy as np
import pinnslicer.nn as mlp
from pinnslicer.orbits.solvers import FDPhotonOrbitSolver, PinnSlicer

phi = np.linspace(0, 5*np.pi/2, 400)      # where the solution is wanted
y0  = (0.2, 0.34)                         # (u0, v0)

# finite-difference reference
fd = FDPhotonOrbitSolver()(y0, phi)

# PINN trained on the slice [0, dphi], deployed recursively
pinn  = mlp.load_checkpoint(mlp.Solution(mlp.FCNN()), 'runs/dphi010/dphi010_params.pth')
sol   = PinnSlicer(pinn, dphi=0.1)(y0, phi)

sol.phi, sol.u, sol.v     # the orbit; phi is trimmed to what was computed
sol.status                # why the loop stopped: COMPLETED, ESCAPED,
                          # ABSORBED, MAX_STEPS or PHI_GUARD
sol.n_iterations          # FD steps, or phi-segments the recursion visited
sol.complete              # True if the requested phi range was reached
```

`PinnSlicer` requires the points in `phi` to be spaced no more widely than
`dphi`: a coarser grid leaves whole segments with no requested point, which
the recursion would skip silently. It warns when that happens, and raises
instead if constructed with `strict=True`.

### Reproducibility

`set_seed` seeds python, numpy and torch in one call. The Sobol scrambler
keeps its own generator and is seeded where the sample is drawn:

```python
from pinnslicer.utils.seeding import set_seed
from pinnslicer.utils.data import SobolSample

set_seed(1234)
sample = SobolSample(lower, upper, num_points_exp=16, seed=1234)
```

Notebook `01_pinn_training.ipynb` records both the seed and the network
architecture in the run's yaml configuration, so a run can be repeated and its
weights reloaded into a network of the right shape.

## Getting Started

Clone the repository:

```

git clone https://github.com/soot-bit/pinnslicer.git
cd pinnslicer

```

inside a virtual environment:
Install the project in editable mode (installs dependencies too):

```

pip install -e .

```

Verify the installation:

```

python -c "from pinnslicer import nn"

```

### Installing with conda

On macOS with an Intel CPU (`uname -m` reports `x86_64`), PyPI has no torch
wheel newer than 2.2.2 — PyTorch stopped building macOS x86 wheels after that
version. conda-forge does still build them, so install torch with conda first
and let pip install the rest:

```

conda create -n slicer python=3.12
conda activate slicer
conda install -c conda-forge pytorch numpy
pip install -e .

```

The dependency floors in `pyproject.toml` are deliberately loose, so pip
accepts the torch that conda already installed instead of trying to fetch a
newer one that does not exist for this platform. To install the notebook
extras as well:

```

pip install -e ".[notebooks]"

```

## Google Colab installation `pinnslicer`
  1. Assign Colab working folder to string `COLAB_FOLDER` in notebook.
  2. Execute the code below in a notebook cell before your imports (see, for example, `01_pinn_training.ipynb`).
```python
COLAB_FOLDER = 'AIMS' # change as needed
GITHUB_USER  = 'soot-bit'
GITHUB_REPO  = 'pinnslicer'
GITHUB_FOLDERS = ['pinnslicer']
#------------------------------------------------------
MYDRIVE      = '/content/gdrive/MyDrive'
GITHUB_BASE  = 'https://raw.githubusercontent.com'
GITHUB_PATH  = f'{MYDRIVE}/{COLAB_FOLDER}'
#------------------------------------------------------
try:
    from google.colab import drive
    drive.mount('/content/gdrive')
    print('\nGoogle Drive mounted\n')
    IN_COLAB = True
except:
    print('\nRunning locally\n')
    IN_COLAB = False
 
if IN_COLAB:
    %cd {GITHUB_PATH}
    %rm -f {GITHUB_PATH}/clone2colab.ipynb
    !wget -q {GITHUB_BASE}/{GITHUB_USER}/{GITHUB_REPO}/refs/heads/main/clone2colab.ipynb
    %run {GITHUB_PATH}/clone2colab.ipynb
    %ls
```

