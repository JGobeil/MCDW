import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numexpr as ne
import pandas as pd

from timing import Timing
from timing import TimingWithBatchEstimator
from hexsurface import HexagonalDirectPosition
from sites import SitesGroup
from utils import sizeof_fmt

import cmontecarlo

kB = 8.6173303e-5 # boltzmann contant in eV/K


class MonteCarloSimulator(cmontecarlo.CMonteCarloSimulator):
    def __init__(self,
                 surface,
                 temperature_func,
                 output_path,
                 create_image_queue=None,
                 *args, **kwargs
                 ):

        super().__init__(
            surface=surface,
            *args, **kwargs
        )

        os.makedirs(output_path, exist_ok=True)
        self.temperature_func = temperature_func
        self.temperature = temperature_func(0)
        self.surface = surface

        self.create_image_queue = create_image_queue
        self.outpath = output_path

    @property
    def occupancy(self):
        return self.occupancy_int64 > 0

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, t):
        self._temperature = t
        self.kBT = kB * t

    @property
    def lap_info(self):
        return {
            'Energy': self.get_total_energy(),
            'Temperature': self.temperature,
            'Coverage': self.coverage,
            'Atom on surface': self.n_used,
            'Attempted moves': self.attempted_moves,
            'Sucessful moves': self.successful_moves,
            'Not moved moves': self.not_moved_moves,
            #'Blocked moves': self.blocked_moves,
            'Lap': self.lap,
            #'Total steps': self.total_steps,
        }

    @property
    def boltzmann_T(self):
        #return 1/(8.6173303e-5 * self.temperature)
        return 11604.5221105 / self.temperature

    def run(self):
        t = Timing('Running MC', nbsteps=self.lap_max,
                   tic_custom_fields=[
                       ('Energy', 'g', 10),
                       ('Temperature', 'g', 12),
                       ('Coverage', 'g', 10),
                       ('Atom on surface', 'd', 16),
                       ('Attempted moves', 'd', 16),
                       ('Sucessful moves', 'd', 16),
                       ('Not moved moves', 'd', 16),
        #               #('Blocked moves', 'd', 16),
                   ])
        self.save_init_state()
        for i in range(self.lap_max):
            self.temperature = self.temperature_func(self.lap)
            self.run_lap()
            self.save_state()
            if self.create_image_queue is not None:
                self.create_image_queue.put(self.lap)
            stat = self.lap_info
            t.tic(**stat)
        self.save_state()
        t.finished()

        if self.create_image_queue is not None:
            self.create_image_queue.put('QUIT')

    def save_init_state(self):
        fn = os.path.join(self.outpath, "init.npz")
        #t = Timing('Saving initial state to %s' % fn)
        np.savez_compressed(fn,
                            sites_x=self.surface.stx,
                            sites_y=self.surface.sty,
                            sites_name=[self.surface.sites[i].name
                                        for i in self.surface.sts],
                            a=self.surface.a,
                            )

        #t.prt('File saved ( %s )' % sizeof_fmt(fn))
        #t.finished()

    def save_state(self):
        fn = os.path.join(self.outpath, "occ_%.10d.npy" % self.lap)
        #t = Timing('Saving to %s' % fn)
        np.save(fn, self.occupancy)
        #t.prt('File saved ( %s )' % sizeof_fmt(fn))
        #t.finished()

    def show(self,
             savefig=None,
             show=False,
             symbol=None,
             symbol_scale=0.9,
             ):
        #plt.scatter(
        #    self.sites.uc.x,
        #    self.sites.uc.y,
        #    30, 'black',  marker=(6, 0, 0))

        if symbol is None:
            symbol = 'hexagon'

        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(111)

        for i in reversed(range(len(self.surface.stidx))):
            stidx = self.surface.stidx[i]
            if symbol == 'scatter':
                idx = stidx[self.index[stidx]]
                plt.scatter(
                    self.surface.stx[idx],
                    self.surface.sty[idx],
                    s=10, c=self.surface.sites[i].color, marker='.')
            else:
                if symbol == 'hexagon':
                    for idx in stidx[self.index[stidx]]:
                        ax.add_patch(
                            RegularPolygon(
                                (self.surface.stx[idx], self.surface.sty[idx]),
                                6, # number of vertices
                                self.surface.a*symbol_scale, # radius
                                #orientation=np.pi/6,
                                color=self.surface.sites[i].color,
                            )
                        )


        axmin = min((self.surface.stx.min(), self.surface.sty.min())) * 1.05
        axmax = min((self.surface.sty.max(), self.surface.sty.max())) * 1.05

        plt.axis((axmin, axmax, axmin, axmax))

        if savefig is not None:
            plt.savefig(os.path.join(self.outpath, "%.10i.png" % savefig))
            plt.close()
        if show:
            self.show()
