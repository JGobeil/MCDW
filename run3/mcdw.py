#!/bin/python3

import os
import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
import pandas as pd
from types import SimpleNamespace

import sys
sys.path.append('..')
from timing import Timing
from montecarlo import MonteCarloSimulation
from hexsurface import HexagonalDirectPosition
from sites import SitesGroup
from visualize import CreateImages


def LJ_ne(i, mask, surface, epsilon, r_m):
    """ Lennard-Jones potential using numexp"""
    r = surface.nnr[i][mask]
    return ne.evaluate('sum(epsilon*((r_m/r)**12 - 2*(r_m/r)**6))')


def LJ_np(i, mask, surface, epsilon, r_m):
    """ Lennard-Jones potential using numpy"""
    rmr = (r_m/(surface.nnr[i][mask]))**6
    return epsilon*np.sum(rmr**2 - 2*rmr) + surface.ste[i]


def LJ_np3_prepared(i, mask, surface, epsilon, r_m):
    """ Lennard-Jones potential using numpy and precalculation"""
    return np.sum(surface.LJ[i][mask]) + surface.ste[i]


def LJ_np3_prepare(surface, epsilon, r_m):
    """ Lennard-Jones potential using numpy precalculation"""
    t = Timing('Precalculating LJ potential.')
    rm6 = r_m**6
    def pre(i):
        r6 = surface.nnr[i]**6
        return np.array(epsilon*rm6*((rm6-2*r6)/(r6*r6)))

    surface.LJ =  [pre(i) for i in range(surface.stlen)]
    t.finished()


def NN_prepare(surface):
    """ Potential using the nearest neighbors"""
    conf = [ # (energy, number of nn)
        ( 0.100, 3),  # nn 1   fcc -> hcp
        ( 0.050, 6),  # nn 2   fcc -> fcc
        ( 0.010, 3),  # nn 3   fcc -> hcp
        (-0.019, 6),  # nn 4   fcc -> hcp
        (-0.020, 6),  # nn 5   fcc -> fcc (sqrt(3) x sqrt(3) R30)
        (-0.019, 6),  # nn 6   fcc -> fcc
        (-0.010, 6),  # nn 7   fcc -> hcp
        (-0.008, 3),  # nn 8   fcc -> hcp
        (-0.004, 6),  # nn 9   fcc -> hcp
        # > nn 9 ==> energy = 0
    ]

    nnlen = surface.nnr.shape[-1]
    surface._pre_nn_pot = np.zeros(nnlen)

    i = 0
    for c in conf:
        j = i+c[1]
        surface._pre_nn_pot[i:j] = c[0]
        i = j

def NN(i, mask, surface):
    return np.sum(surface._pre_nn_pot*mask) + surface.ste[i]



if __name__ == '__main__':

    t = Timing('Running MCDW')
    np.random.seed(201)

    t.prt("Creating surface")
    s = HexagonalDirectPosition(
        a = 0.362 / np.sqrt(2) / 2,
        radius=50,
        nn_radius=3,
        sites=SitesGroup.fcc_hcp_111(),
        load=True,
        save=True,
        load_and_save_dir='../sdb'
    )

    #s.save_json()
    t.prt("Creating images creator process")
    ci = CreateImages('states', 'img')
    q = ci.queue
    ci.start()

    t.prt("Creating Monte Carlo simulator")
    mc = MonteCarloSimulation(
        surface=s,
        max_laps=200,
        steps_per_lap=200,
        moves_per_step=10,
        potential_func=NN,
        potential_params={},
        precalculate_potential_func=NN_prepare,
        temperature_func=lambda l: 205 - 1*l,
        max_coverage=0.15,
        output_path='states',
        create_image_queue=q,
    )

    #mc.prepopulate(0.0)
    t.prt("Running Monte Carlo simulator")
    mc.run()

    t.prt("Waiting for images creator process to complete")
    ci.join()




