import subprocess
from timing import Timing

t = Timing('Building cython extension')
rc = subprocess.run("python setup.py build_ext --inplace".split()).returncode
t.finished()
if rc != 0:
    t.prt('Error when building extension')
    exit(rc)

import numpy as np
import cmontecarlo
import montecarlo

from hexsurface import HexagonalDirectPosition
from sites import SitesGroup
from visualize import CreateImages

pot = cmontecarlo.PotentialFuncNN(
    conf = [
         0.050,  # nn 1 ( x3 /  0: 3 ) fcc -> hcp
         0.010,  # nn 2 ( x6 /  3: 9 ) fcc -> fcc
        -0.050,  # nn 3 ( x3 /  9:12 ) fcc -> hcp
        -0.090,  # nn 4 ( x6 / 12:18 ) fcc -> hcp
        -0.103,  # nn 5 ( x6 / 18:24 ) fcc -> fcc (sqrt(3) x sqrt(3) R30)
        -0.060,  # nn 6 ( x6 / 24:30 ) fcc -> fcc
        -0.010,  # nn 7 ( x6 / 30:36 ) fcc -> hcp
         0.100,  # nn 8 ( x3 / 36:39 ) fcc -> hcp
        #-0.004,  # nn 9   fcc -> hcp
        # nn 10+ ==> energy = 0
    ],
)

s = HexagonalDirectPosition(
    a = 0.362 / np.sqrt(2) / 2,
    radius=70,
    nn_radius=3,
    sites=SitesGroup.fcc_hcp_111(),
    load=True,
    save=True,
    load_and_save_dir='../sdb'
)

ci = CreateImages(read_path='cmc-test', write_path='cmc-test/img')

mc = montecarlo.MonteCarloSimulator(
    surface=s,
    nb_binding_sites=s.stlen,
    lap_max=100,
    steps_per_lap=500,
    moves_per_step=5,
    target_coverage=0.155,
    potential_func=pot,
    output_path='cmc-test',
    temperature_func=lambda lap: 300 - lap*2,
    create_image_queue=ci.queue,
)

pot.set_surface(s)
pot.set_simulator(mc)

if __name__ == '__main__':
    ci.process.start()
    mc.run()
    print('Waiting for images processing...')
    ci.process.join()


