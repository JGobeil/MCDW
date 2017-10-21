#!/bin/python3

import os
import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
import pandas as pd

from timing import Timing
from hexsurface import HexagonalSurface
from sites import SitesGroup


class MonteCarloSimulation:
    def __init__(self,
                 surface,
                 max_laps,
                 steps_per_lap,
                 potential,
                 potential_params,
                 temperature_func,
                 target_coverage,
                 outpath,
                 precalculate_potential=None,
                 ):

        os.makedirs(outpath, exist_ok=True)

        self.s = surface
        self.V = potential
        self.V_params = potential_params

        if precalculate_potential is not None:
            precalculate_potential(surface, **potential_params)


        self.max_laps = max_laps
        self.steps_per_lap = steps_per_lap
        self.target_coverage = target_coverage
        self.outpath = outpath

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
        for i, s in enumerate(self.s.sites):
            self.landing_prob[self.s.stidx[i]] = s.prob

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
        t = Timing('Running MC',
                   total_steps=self.max_laps,
                   batch_size=1,
                   )
        while t.batch_size > 0:
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

        nni = self.s.nni[i]

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

        nnr = self.s.nnr[i]
        nni = self.s.nni[i]

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
        s = self.s.sites[i]
        t = Timing('Prepopulating sites %s' % s.name)
        total = self.stlen
        all = np.arange(total)
        c = self.s.stidx[i]
        t.prt("Prepopulating %i sites on %i." % (len(c), total ))
        self.add_prepopulated = len(c)
        self.occ[c] = True
        t.prt("Updating energy")
        self.update_energy(all)
        self.keep_modification(all)
        t.finished()

    def prepopulate_split(self, i, j, gap):
        s = self.s
        s1 = s.sites[i]
        s2 = s.sites[j]

        idx1 = self.s.stidx[i]
        idx2 = self.s.stidx[j]

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
            nni = self.s.nni[i]
            self.energies[i] = self.V(i, self.occ[nni], self.s,
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

        plt.figure(figsize=(10,10))

        for i, stidx in enumerate(self.s.stidx):
            idx = stidx[self.occ[stidx]]

            plt.scatter(
                self.s.stx[idx],
                self.s.sty[idx],
                s=10, c=self.s.sites[i].color, marker='.')

        axmin = min((self.s.stx.min(), self.s.sty.min()))*1.05
        axmax = min((self.s.sty.max(), self.s.sty.max()))*1.05

        plt.axis((axmin, axmax, axmin, axmax))

        if savefig is not None:
            plt.savefig(os.path.join(self.outpath, "%.10i.png" % savefig))
            plt.close()
        if show:
            self.show()

def LJ_ne(i, mask, surface, epsilon, r_m):
    r = surface.nnr[i][mask]
    return ne.evaluate('sum(epsilon*((r_m/r)**12 - 2*(r_m/r)**6))')


def LJ_np(i, mask, surface, epsilon, r_m):
    rmr = (r_m/(surface.nnr[i][mask]))**6
    return epsilon*np.sum(rmr**2 - 2*rmr) + surface.ste[i]

def LJ_np2(i, mask, surface, epsilon, r_m):
    rm6 = r_m**6
    r6 = surface.nnr[i][mask]**6
    return epsilon*rm6*np.sum((rm6 - 2*r6)/(r6*r6)) + surface.ste[i]

def LJ_np3_prepared(i, mask, surface, epsilon, r_m):
    return np.sum(surface.LJ[i][mask]) + surface.ste[i]

def LJ_np3_prepare(surface, epsilon, r_m):
    t = Timing('Precalculating LJ potential.')
    rm6 = r_m**6
    def pre(i):
        r6 = surface.nnr[i]**6
        return np.array(epsilon*rm6*((rm6-2*r6)/(r6*r6)))

    surface.LJ =  [pre(i) for i in range(surface.stlen)]

    t.finished()

def Paper1(i, mask, surface):
    return np.sum(surface.Paper1[i][mask]) + surface.ste[i]

def Paper1_prepare(surface, show=False):
    t = Timing('Precalculating Paper1 potential.')

    fit_PBE = np.array([
        (3.50, -1.724),
        (3.94, -1.804),
        (4.43, -1.805),
        (5.14, -1.793),
        (5.26, -1.789),
        (5.94, -1.795),
    ])
    r = fit_PBE[:,0] / 10
    E = fit_PBE[:,1] - np.array([
        (-1.790 + -1.790)/2,
        (-1.803 + -1.790)/2,
        (-1.803 + -1.803)/2,
        (-1.803 + -1.803)/2,
        (-1.803 + -1.790)/2,
        (-1.803 + -1.790)/2,
    ])

    poly = np.polyfit(r, E, 20)
    pre_poly = np.poly1d(poly)

    def pre_r(r):
        v = pre_poly(r)
        #v[r > 0.6] = 0
        #v[r < .35] = 0.1
        return v

    surface.Paper1 =  [pre_r(surface.nnr[i]) for i in range(surface.stlen)]

    if show:
        rmax = np.max([np.max(nnr) for nnr in surface.nnr])
        nnr = np.random.choice(surface.nnr)
        x = np.linspace(0, np.max(r), 1000)
        plt.plot(x, pre_r(x), label='Potential')
        plt.scatter(nnr, pre_r(nnr), label='nnr')
        plt.scatter(r, E, label='Fitted point')
        plt.legend()
        plt.show()

    t.finished()


if __name__ == '__main__':
    np.random.seed(201)

    s = HexagonalSurface(
        a=0.362 / np.sqrt(2) / 2,
        radius=25,
        nn_radius=4,
        sites=SitesGroup.fcc_hcp_111(),
        load=True,
        save=True,
        load_and_save_dir='sdb'
        )

    #s.save_json()

    mc = MonteCarloSimulation(
        surface=s,
        max_laps=20,
        steps_per_lap=1000,
        potential=Paper1,
        potential_params={},
        precalculate_potential=Paper1_prepare,
        temperature_func=lambda l: 300 - 0.1*l,
        target_coverage=0,
        outpath='test3',
    )

    mc.prepopulate(0.1)
    mc.run()

