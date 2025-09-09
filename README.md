# csdl_fwh: Ffowcs Williams–Hawkings (FWH) acoustic analogy

**csdl_fwh** is a package written by Computational System Design Language (CSDL) implementing **Farassat’s Formulation 1A** of the Ffowcs Williams–Hawkings (FWH) acoustic analogy.  
It takes time-domain pressure (and surface kinematics, when applicable) on **impermeable (solid)** or **permeable** data surfaces and returns the **acoustic pressure** in both the **time** and **frequency** domains.

## Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
pip install "git+https://github.com/cfdkang/csdl_fwh.git"

## Installation instructions for developers
To install **csdl_fwh**, first clone the repository and install using pip. On the terminal or command line, run
git clone https://github.com/cfdkang/csdl_fwh.git
pip install -e ./csdl_fwh

## Test the package
Example codes (ex_1_vlm.py and ex_2_analytic.py) are included in the examples folder.