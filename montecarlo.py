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


class MonteCarloSimulation:
    def __init__(self,
                 surface,
                 max_laps,
                 steps_per_lap,
                 potential_func,
                 potential_params,
                 temperature_func,
                 max_coverage,
                 output_path,
                 precalculate_potential_func=None,
                 ):

        os.makedirs(output_path, exist_ok=True)

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
        while t.done < self.max_laps:
            self.lap = t.done
            self.temperature = self.temperature_func(self.lap)
            self.run_lap()
            t.tic()
        t.finished()

    def run_lap(self):
        N = self.steps_per_lap
        for step in range(N):
            if self.coverage < self.target_coverage and step < (N / 10):
                self.add_atom()

            else:
                self.move_atom()

        self.show(savefig=self.lap)

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
        i = (np.arange(self.stlen)[self.occ])[np.random.randint(0, self.atoms_on_surface)]

        nnr = self.surface.nnr[i]
        nni = self.surface.nni[i]

        occ = self.occ[nni]

        not_occ = np.logical_not(occ) & (nnr < 0.3)
        not_occ_i = nni[not_occ]
        not_occ_r = nnr[not_occ]

        prob = np.cumsum(1/not_occ_r)
        prob /= prob[-1]

        j = not_occ_i[np.searchsorted(prob, np.random.uniform())]

        self.occ[i] = False
        self.occ[j] = True

        self.update_energy(nni)
        self.update_energy([i,j])

        if self.accept():
            self.keep_modification(nni)
            self.keep_modification([i,j])
            self.move_accepted += 1
        else:
            self.reverse_modification(nni)
            self.reverse_modification([i,j])
            self.move_rejected += 1

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
        total = self.stlen
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
            nni = self.surface.nni[i]
            self.energies[i] = self.V(i, self.occ[nni], self.surface,
                                      **self.V_params)
        self.energy = np.sum(self.energies)

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

    def show(self, savefig=None, show=False):
        #plt.scatter(
        #    self.sites.uc.x,
        #    self.sites.uc.y,
        #    30, 'black',  marker=(6, 0, 0))

        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(111)

        for i, stidx in enumerate(self.surface.stidx):
            #idx = stidx[self.occ[stidx]]
            #plt.scatter(
            #    self.surface.stx[idx],
            #    self.surface.sty[idx],
            #    s=10, c=self.surface.sites[i].color, marker='.')
            for idx in stidx[self.occ[stidx]]:
                #print((self.surface.stx[idx], self.surface.stx[idx]))
                #print(self.surface.a)

                ax.add_patch(
                        RegularPolygon(
                            (self.surface.stx[idx], self.surface.sty[idx]), # (x,y)
                            6, # number of vertices
                            self.surface.a*0.9, # radius
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
