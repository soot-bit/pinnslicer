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
This module can be used to train a Physics-Informed Neural Network (PINN) to solve the following nonlinear ordinary differential equation (ODE),
```math
\overset{\textstyle\cdot\cdot}{u}  \: + \: u - \: 3 \: \frac{u^2}{2}  =  0 ,
```
which describes the orbit of photons in a Schwarzschild spacetime about a spherically symmetric body, where 
```math
  u = \frac{r_s }{ r},
```
and $r_s$ is the Schwarzschild radius.

## Installation

The notebooks can run within Colab and as well as locally  using the following commands:
```
conda update conda
conda create -n slicer python=3.11
conda activate slicer
conda install jupyterlab notebook
conda install pytorch
conda install matplotlib
conda install pandas
```

Then clone the repository in the activated environment, here called *slicer*,

```
git clone https://github.com/soot-bit/pinnslicer.git
cd pinnslicer
```

then install the project in editable mode (installs dependencies too) using,

```
pip install -e .

```
