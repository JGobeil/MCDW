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


class MonteCarloSimulation:
    def __init__(self,
                 surface,
                 max_laps,
                 steps_per_lap,
                 moves_per_step,
                 potential_func,
                 potential_params,
                 temperature_func,
                 max_coverage,
                 output_path,
                 precalculate_potential_func=None,
                 create_image_queue=None
                 ):

        os.makedirs(output_path, exist_ok=True)

        self.create_image_queue = create_image_queue
        self.surface = surface
        self.V = potential_func
        self.V_params = potential_params

        if precalculate_potential_func is not None:
            precalculate_potential_func(surface, **potential_params)

        #self.stats = pd.DataFrame(
        #    columns=[
        #        'temperature',
        #        'energy',
        #        'coverage',
        #    ],
        #    index=np.arange(max_laps))
        #})

        self.max_laps = max_laps
        self.steps_per_lap = steps_per_lap
        self.moves_per_step = moves_per_step
        self.target_coverage = max_coverage
        self.outpath = output_path

        self.temperature_func = temperature_func
        self.temperature = temperature_func(0)

        self.lap = 0

        self.add_prepopulated = 0
        self.add_accepted = 0
        self.add_rejected = 0
        self.move_accepted = 0
        self.move_rejected = 0

        self.stlen = surface.stlen

        self.occ = np.zeros(self.stlen, dtype=bool)
        self.previous_occ = np.zeros(self.stlen, dtype=bool)

        self.energies = np.zeros(self.stlen)
        self.previous_energies = np.zeros(self.stlen)

        self.energy = 0
        self.previous_energy = 0

        self.landing_prob = np.zeros(self.stlen, dtype=float)
        for i, s in enumerate(self.surface.sites):
            self.landing_prob[self.surface.stidx[i]] = s.prob

        # caching variable
        self._nni = np.zeros(4, dtype=int)
        self._occ = np.zeros(4, dtype=bool)
        self._nne = np.zeros(4, dtype=float)
        self._p = np.zeros(4, dtype=float)

    @property
    def lap_info(self):
        per100 = 100 / self.atoms_on_surface
        return {
            'Atom on surface (counted)': np.count_nonzero(self.occ),
            'Atom on surface': self.atoms_on_surface,
            'Coverage': self.coverage,
            'Add prepopulated': self.add_prepopulated,
            'Add prepopulated (%)': self.add_prepopulated * per100,
            'Add accepted': self.add_accepted,
            'Add accepted/rejected': self.add_accepted / self.add_rejected,
            'Add rejected': self.add_rejected,
            'Move accepted': self.move_accepted,
            'Move accepted/rejected': self.move_accepted / self.move_rejected,
            'Move rejected': self.move_rejected,
            'Energy': self.energy,
            'Temperature': self.temperature,
        }

    @property
    def boltzmann_T(self):
        #return 1/(8.6173303e-5 * self.temperature)
        return 11604.5221105 / self.temperature

    @property
    def atoms_on_surface(self):
        return self.add_prepopulated + self.add_accepted

    @property
    def coverage(self):
        return self.atoms_on_surface / len(self.occ)

    def run(self):
        t = Timing('Running MC', nbsteps=self.max_laps)
        self.save_init_state()
        while t.done < self.max_laps:
            self.lap = t.done
            self.temperature = self.temperature_func(self.lap)
            self.run_lap()
            self.save_state()
            if self.create_image_queue is not None:
                self.create_image_queue.put(self.lap)
            t.tic()
        self.save_state()
        t.finished()

        if self.create_image_queue is not None:
            self.create_image_queue.put('QUIT')

    def run_lap(self):
        N = self.steps_per_lap
        for step in range(N):
            if self.coverage < self.target_coverage:
                self.add_atom()
            for i in range(self.moves_per_step*self.atoms_on_surface):
                self.move_atom()

        #self.show(savefig=self.lap)

    def add_atom(self):

        not_occ = np.logical_not(self.occ)
        prob = np.cumsum(self.landing_prob[not_occ])
        prob /= prob[-1]

        i = np.searchsorted(prob, np.random.uniform())
        i = np.arange(self.stlen)[not_occ][i]

        nni = self.surface.nni[i]

        if self.occ[i]:
            print(i, "already occupied")

        self.occ[i] = True
        self.update_energy(nni)
        self.update_energy([i,])

        if self.accept():
            self.keep_modification(nni)
            self.keep_modification([i,])
            self.add_accepted += 1
        else:
            self.reverse_modification(nni)
            self.reverse_modification([i,])
            self.add_rejected += 1

    def move_atom(self):
        #prob = np.cumsum(self.occ, dtype=float)
        #prob /= prob[-1]

        #i = np.searchsorted(prob, np.random.uniform())
        i = np.random.choice(self.surface.sti[self.occ])
        self.occ[i] = False
        self._nni[0] = i
        self._nni[1:] = self.surface.nni[i][:3]
        self._nne[:] = [0 if self.occ[ii] else self.get_energy(ii)
                        for ii in self._nni]

        self._occ[:] = self.occ[self._nni]
        self._p[:] = np.exp(-self._nne * self.boltzmann_T)
        self._p[self._occ] = 0
        self._p[:] /= np.sum(self._p)

        j = np.random.choice(self._nni, p=self._p)
        #j = not_occ_i[np.searchsorted(prob, np.random.uniform())]


        if self.occ[j]:
            print('!!!!Moving to an occupied position!!!!')

        self.occ[j] = True


        self.update_energy(self._nni)
        self.update_energy([i,j])

        self.keep_modification(self._nni)
        self.keep_modification([i,j])
        self.move_accepted += 1

        #
        #if self.accept():
        #    self.keep_modification(nni)
        #    self.keep_modification([i,j])
        #    self.move_accepted += 1
        #else:
        #    self.reverse_modification(nni)
        #    self.reverse_modification([i,j])
        #    self.move_rejected += 1

    def prepopulate(self, fraction):
        t = Timing('Prepopulating (%g%%)' % (fraction * 100))
        total = self.stlen
        all = np.arange(total)
        c = np.random.choice(all, size=int(fraction * total), replace=False)
        t.prt("Prepopulating %i sites on %i." % (len(c), total ))
        self.add_prepopulated = len(c)
        self.occ[c] = True
        t.prt("Updating energy")
        self.update_energy(all)
        self.keep_modification(all)
        t.finished()

    def prepopulate_sites(self, i):
        s = self.surface.sites[i]
        t = Timing('Prepopulating sites %s' % s.name)
        all = np.arange(total)
        c = self.surface.stidx[i]
        t.prt("Prepopulating %i sites on %i." % (len(c), total ))
        self.add_prepopulated = len(c)
        self.occ[c] = True
        t.prt("Updating energy")
        self.update_energy(all)
        self.keep_modification(all)
        t.finished()

    def prepopulate_split(self, i, j, gap):
        s = self.surface
        s1 = s.sites[i]
        s2 = s.sites[j]

        idx1 = self.surface.stidx[i]
        idx2 = self.surface.stidx[j]

        t = Timing('Prepopulating split')
        total = self.stlen
        all = np.arange(total)

        c = np.union1d(
            idx1[s.stx[idx1] < -gap/2],
            idx2[s.stx[idx2] >  gap/2],
        )


        t.prt("Prepopulating %i sites on %i." % (len(c), total ))
        self.add_prepopulated = len(c)
        self.occ[c] = True
        t.prt("Updating energy")
        self.update_energy(all)
        self.keep_modification(all)
        t.finished()

    def update_energy(self, indices):
        idx = np.array(indices)
        occ = self.occ[idx]

        # put to zero all unoccupied sites
        self.energies[idx[np.logical_not(occ)]] = 0

        # calculate energy for occupied sites
        for i in idx[occ]:
            self.energies[i] = self.get_energy(i)
        self.energy = np.sum(self.energies)

    def get_energy(self, i):
        nni = self.surface.nni[i]
        return self.V(i, self.occ[nni], self.surface, **self.V_params)

    def keep_modification(self, indices):
        self.previous_energies[indices] = self.energies[indices]
        self.previous_energy = self.energy
        self.previous_occ[indices] = self.occ[indices]

    def reverse_modification(self, indices):
        self.energies[indices] = self.previous_energies[indices]
        self.energy = self.previous_energy
        self.occ[indices] = self.previous_occ[indices]

    def accept(self):
        dE = self.energy - self.previous_energy
        if dE < 0:
            return True
        else:
            return np.random.uniform() < np.exp(-(dE*self.boltzmann_T))

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
        np.save(fn,self.occ)
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
                idx = stidx[self.occ[stidx]]
                plt.scatter(
                    self.surface.stx[idx],
                    self.surface.sty[idx],
                    s=10, c=self.surface.sites[i].color, marker='.')
            else:
                if symbol == 'hexagon':
                    for idx in stidx[self.occ[stidx]]:
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
