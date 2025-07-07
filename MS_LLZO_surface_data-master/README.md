# Repository of computed data  [![DOI](https://zenodo.org/badge/128926251.svg)](https://zenodo.org/badge/latestdoi/128926251)


## Summary 
This repository contains the relevant surface example input files and structures computed in the paper Particle Morphology and Lithium Segregation to Surfaces of the Li<sub>7</sub>La<sub>3</sub>Zr<sub>2</sub>O<sub>12</sub> Solid Electrolyte, P. Canepa *et al.*, *Chem. Mater.* (2018) DOI: [10.1021/acs.chemmater.8b00649](https://doi.org/10.1021/acs.chemmater.8b00649).

### Only the most stable orderings are archived in this repository.

Two type of surface models are available per each distinct chemical termination: 

* Stoichiometric 
* Off-stoichiometric or nonstoichiometric 

Finally, for certain surface models the effect of Li converage is also explored. 

Additional surface structures (orderings) are available upon request. 

## Calculations setup
### DFT environment setup, INCAR file 
All structures have been carefully optimized (at the bulk volume) utilizing the [Vienna Ab initio Simulation Package](http://www.vasp.at). The following INCAR file is based on the [**MaterialsProject**](https://www.materialsproject.org) and the **MVLSlabSet** standards defined via [pymatgen](http://pymatgen.org).  

```
EDIFF  = 1e-06
EDIFFG = -0.01
ENCUT  = 520
ALGO   = Fast
AMIN   = 0.01
AMIX   = 0.2
BMIX   = 0.001
IBRION = 2
ICHARG = 1
ISIF   = 2
ISMEAR = 0
ISPIN  = 1
LORBIT = 11
LPLANE = True
LREAL  = Auto
LWAVE  = False
NELM   = 100
NELMIN = 6
NPAR   = 12
NSW    = 99
PREC   = Accurate
SIGMA  = 0.05
SYMPREC= 1e-8
```

###  Setup of *k*-space integration, KPOINT file 
The *k*-point integration mesh was based on the attached KPOINT file. Gamma centering was invoked whenever needed. 

```
Automatic mesh
0
M
2 2 1
```
### Potential utilized, POTCAR file
The PAW potentials at the Generalized Gradient level of approximation used were: 

|  **Element**      |  **Potential Name**            |  **Electronic configuration** | **ENMAX (eV)** |
| ------------- |:-------------              | :-----       | :-----       |
| Li            | PAW\_PBE Li\_sv 23Jan2001  | 1s2s2p       | 271.649 |
| La            | PAW\_PBE La 06Sep2000      | [core=Kr4d]  | 219.313 |
| Zr            | PAW\_PBE Zr\_sv 07Sep2000  | 4s4p5s4d     | 229.839 |
| O             | PAW\_PBE O 08Apr2002       | s2p4         | 400.000 |
