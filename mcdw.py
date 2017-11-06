import argparse
import os
import subprocess
from timing import Timing
import numpy as np

from hexsurface import HexagonalDirectPosition
from sites import SitesGroup
from visualize import CreateImages

if __name__ == "__main__":
    t = Timing('Building cython extension')
    rc = subprocess.run("python setup.py build_ext --inplace".split()).returncode
    t.finished()
    if rc != 0:
        t.prt('Error when building extension')
        exit(rc)

    import cmontecarlo
    import montecarlo

    parser = argparse.ArgumentParser(description='Monte Carlo Domain Walls')
    parser.add_argument(
        'directories',
        type=str,
        nargs='+'
    )
    args = parser.parse_args()

    for path in args.directories:
        conf_file = os.path.join(path, 'conf.py')
        if not os.path.isfile(conf_file):
            print("File %s not found in %s" % ('conf.py', path))
            continue

        t = Timing("Running MCDW in '%s'" % path)
        exec(open(conf_file).read())

        # ... quick and dirty

        if pot_type == 'NN':
            pot = cmontecarlo.PotentialFuncNN(conf=pot_conf)
        else:
            t.prt("Potential type unknown '%s'" % pot_type)

        surface = HexagonalDirectPosition(
            a = 0.362 / np.sqrt(2) / 2,
            radius=surface_radius,
            nn_radius=nn_radius,
            sites=SitesGroup.fcc_hcp_111(
                fcc_energy=sites_energy_conf['fcc'],
                hcp_energy=sites_energy_conf['hcp'],
            ),
            load=True,
            save=True,
            load_and_save_dir=surface_db
        )

        states_path = os.path.join(path, 'states')
        images_path = os.path.join(path, 'images')

        if use_image_process == True:
            ci = CreateImages(read_path=states_path, write_path=images_path)
            ci_queue = ci.queue
            ci.process.start()
        else:
            ci = None
            ci_queue = None

        mc = montecarlo.MonteCarloSimulator(
            surface=surface,
            nb_binding_sites=surface.stlen,
            lap_max=lap_max,
            steps_per_lap=steps_per_lap,
            moves_per_step=moves_per_step,
            target_coverage=target_coverage,
            potential_func=pot,
            output_path=states_path,
            temperature_func=temperature,
            create_image_queue=ci_queue,
        )

        pot.set_surface(surface)
        pot.set_simulator(mc)

        t.prt("Loading complete. Stating simulation. sites = %d" % surface.stlen)

        mc.run()

        if ci is not None:
            print('Waiting for images processing...')
            ci.process.join()

        t.finished()

        del pot, mc, surface

