#!/bin/python3

import os
import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
import pandas as pd

import sys
sys.path.append('..')
from timing import Timing
from timing import TimingWithBatchEstimator
from montecarlo import MonteCarloSimulation
from hexsurface import HexagonalDirectPosition
from sites import SitesGroup


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

    s = HexagonalDirectPosition(
        a=0.362 / np.sqrt(2) / 2,
        radius=30,
        nn_radius=4,
        sites=SitesGroup.fcc_hcp_111(),
        load=True,
        save=True,
        load_and_save_dir='../sdb'
        )

    #s.save_json()

    mc = MonteCarloSimulation(
        surface=s,
        max_laps=20,
        steps_per_lap=1000,
        potential_func=Paper1,
        potential_params={},
        precalculate_potential_func=Paper1_prepare,
        temperature_func=lambda l: 300 - 0.1*l,
        max_coverage=0.15,
        output_path='img',
    )

    mc.prepopulate(0.1)
    mc.run()

