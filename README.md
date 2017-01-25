# hnbr-generation

Python routines to hydrogenate and cross-link nitrile butadiene rubber 

## lmp2nx.py
turns LAMMPS data file containing polymer into a network 

## nx2opls.py
infers OPLS atom types as well as bonded/non-bonded force-field information from network, e.g., mass of 1 -> hydrogen etc.

## gmx/
contains OPLS force-field information, reproduced from GROMACS v5.0.1

## cross_link.py
hydrogenates/cross-links NBR into HNBR, based on user-defined cross-link fraction
