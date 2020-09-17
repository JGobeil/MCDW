# Monte Carlo Domain Walls (MCDW)

Code in Python and Cython for Kinetic Monte Carlo simulations of suface growth and reconstruction Cu2N/Cu(111).

The program implemented a Kinetic Monte Carlo approach where a surface is divided into multiple sites. Atoms can bound on those sites and jump between each site. At each step, an atom will be added to the surface or will try to jump between to sites.  The resulting configuration will be kept if it lowers the energy of the system. There is a small probability of keeping a configuration that raises the energy is kept to allow the system to go out of local minima. This probability depends on the temperature. 

The probability of jumping between the sites are determined by the relative energies between the neighbouring sites.

See files example1/conf.py and example2/conf.py for example on how to run the code.

## Features

- Create/save/load surface configurations automatically
- Nearest neighbors pre calculation and save/load
- Separate process for images creation
- High-performance with the inner loop implemented in Cython

## Nearest neighbors shells

| NN Shell |  Count |  Transition  | Array  | Notes  |
|----------|--------|--------------|--------|--------|
|  NN 1    |   3    |  fcc -> hcp  |  0..3  |        |
|  NN 2    |   6    |  fcc -> fcc  |  3..9  |        |     
|  NN 3    |   3    |  fcc -> hcp  |  9..12 |        |
|  NN 4    |   6    |  fcc -> hcp  | 12..18 |        |
|  NN 5    |   6    |  fcc -> fcc  | 18..24 | (sqrt(3) x sqrt(3) R30 |
|  NN 6    |   6    |  fcc -> fcc  | 24..30 |        |
|  NN 7    |   6    |  fcc -> hcp  | 30..36 |        |
|  NN 9    |   3    |  fcc -> hcp  | 36..39 |        |


