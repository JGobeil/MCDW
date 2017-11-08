import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import json

from timing import Timing
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
            'T': self.temperature,
            'Coverage': self.coverage * 6,
            'Atoms': self.n_used,
            'Moves tried': self.attempted_moves,
            'Moved': self.successful_moves,
            'Not moved': self.not_moved_moves,
            'Error moves': self.error_moves,
            'Error adds': self.error_adds,
            #'Blocked moves': self.blocked_moves,
            'Lap': self.lap,
            #'Total steps': self.total_steps,
        }


    def run(self):
        t = Timing('Running MC', nbsteps=self.lap_max,
                   tic_custom_fields=[
                       ('Lap', '%5d', 6),
                       ('Time left', '%12s', 14),
                       ('Elapsed time', '%12s', 14),
                       ('Energy', '%8g', 12),
                       ('T', '%.2f', 8),
                       ('Coverage', '%.4g', 10),
                       ('Atoms', '%6d', 12),
                       ('Moves tried', '%10d', 12),
                       ('Moved', '%10d', 12),
                       ('Not moved', '%10d', 12),
                       ('Error moves', '%6d', 12),
                       ('Error adds', '%6d', 12),
                   ])
        self.save_init_state()
        for i in range(self.lap_max):
            self.temperature = self.temperature_func(self.lap)
            self.run_lap()

            stat = self.lap_info
            self.save_state(stat)

            t.tic(**stat)

            if self.create_image_queue is not None:
                self.create_image_queue.put(self.lap)


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

        self.save_stat()
        #t.prt('File saved ( %s )' % sizeof_fmt(fn))
        #t.finished()

    def save_stat(self, lap_info=None):
        if lap_info is None:
            lap_info = self.lap_info

        fn = os.path.join(self.outpath, "lap_%.10d.json" % self.lap)
        with open(fn, 'w') as f:
            json.dump(lap_info, f)

    def save_state(self, lap_info=None):
        fn = os.path.join(self.outpath, "occ_%.10d.npy" % self.lap)
        #t = Timing('Saving to %s' % fn)
        np.save(fn, self.occupancy)
        #t.prt('File saved ( %s )' % sizeof_fmt(fn))
        #t.finished()

        self.save_stat(lap_info)



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
