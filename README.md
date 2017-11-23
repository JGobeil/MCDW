# Monte Carlo Domain Walls (MCDW)

Code for Monte Carlo simulations of domain walls formation on Cu2N/Cu(111).

See file run1/mcdw.py for example on how to run the code.

## Features

- Create/save/load surface configurations automatically
- Nearest neighbors pre calculation and save/load
- Separate process for images creation
- Other stuff


## Nearest neighbors shells

 NN Shell |  Count |  Transition  | Array  | Notes
----------+--------+--------------+--------+--------
  NN 1    |   3    |  fcc -> hcp  |  0..3  |
  NN 2    |   6    |  fcc -> fcc  |  3..9  |
  NN 3    |   3    |  fcc -> hcp  |  9..12 |
  NN 4    |   6    |  fcc -> hcp  | 12..18 |
  NN 5    |   6    |  fcc -> fcc  | 18..24 | (sqrt(3) x sqrt(3) R30
  NN 6    |   6    |  fcc -> fcc  | 24..30 |
  NN 7    |   6    |  fcc -> hcp  | 30..36 |
  NN 9    |   3    |  fcc -> hcp  | 36..39 |


