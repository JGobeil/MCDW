""" Functions and class to deal with hexagonal lattices"""
import os
import numpy as np
import matplotlib.pyplot as plt
import json

from timing import Timing
from surfacegeo import create_grid
from surfacegeo import calculate_sites
from surfacegeo import geometric_cut_mask
from surfacegeo import calculate_nn
from utils import sizeof_fmt

surface_filename = 'surface'

class HexagonalDirectPosition:
    default_surface_db_dir = "surface_db"

    savelist = ['ucx',  # unit cell x position
                'ucy',  # unit cell y position
                'uci',  # unit cell i index
                'ucj',  # unit cell j index
                'stx',  # sites x position
                'sty',  # sites y position
                'sts',  # sites index in sites specification list
                'sti',  # sites index in big list
                'nni',  # nn index
                'nnr',  # nn distance
                ]

    """ Hexagonal surface lattice. """
    def __init__(self,
                 a: float,  # scaling
                 radius: float,  # size of the hexagon
                 nn_radius: float,  # radius for nearest-neighbors (unit of 'a')
                 sites,  # list of sites or name of sites list
                 periodic=True,  # use hexagonal periodic boundaries
                 load=True,
                 save=True,
                 load_and_save_dir=None,
                 ):

        t = Timing('Hexagonal surface creation r=%g' % radius)

        # first use a radius that is that big before cutting
        uc_over = 5

        # use 'surface_db' as default load_and_save_dir
        if load_and_save_dir is None:
            self.wd = HexagonalDirectPosition.default_surface_db_dir
        else:
            self.wd = load_and_save_dir

        # create the directory if it doesn't exist
        os.makedirs(self.wd, exist_ok=True)

        # hexagonal specifications
        grid = (np.sqrt(3.)/2, 1)
        corrections = [
            [(2, (.0, 0.5)), ],  # each 2 rows shift position by 0.5 in y
            [],  # no modification for columns
        ]

        self.a = a  # scaling
        self.radius = radius  # radius of the hexagon
        self.nn_radius = nn_radius  # radius of the nearest neighbors search
        self.periodic = periodic  # if the surface is periodic
        self.sites = sites  # sites specifications

        self.ucx = None  # unitcells x position
        self.ucy = None  # unitcells y position
        self.uci = None  # unitcells i index (rows)
        self.ucj = None  # unitcells j index (cols)
        self.stx = None  # sites x position
        self.sty = None  # sites y postions
        self.sts = None  # sites index in site specifications
        self.sti = None  # sites index
        self.nni = None  # sites nearest neighbors index
        self.nnr = None  # sites nearest neighbors distance
        self.nna = None  # sites nearest neighbors angle
        self.ste = None  # sites energy

        if load and self.load():
            # surface loaded from file
            pass
        else:
            # calculate the x, y, i and j of the unit cells
            self.create_uc(radius + uc_over, grid, corrections)

            # calculate the positions of the sites
            self.create_sites(sites)

            # cut sites
            self.cut_st(radius)
            self.cut_uc(radius + 0.5)  # keep the rim

            if self.periodic:
                # add the periodicity
                self.add_periodicity()
                self.cut_st(radius + nn_radius + 0.5)

            # calculate nn
            self.create_nn_list(nn_radius)
            self.cut_st(radius)

            if self.periodic:
                self.uniformize_nn()

            if save:
                self.save()

        # scale to a
        self.scale(a)

        t.finished()

    def save(self):
        path = os.path.join(self.wd, self.id_str)
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, surface_filename) + '.npz'

        t = Timing('Saving to %s' % path)
        np.savez_compressed(fn, **{key: getattr(self, key)
            for key in self.savelist})
        t.prt('File saved ( %s )' % sizeof_fmt(fn))
        t.finished()

    def save_json(self):
        path = os.path.join(self.wd, self.id_str)
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, surface_filename) + '.json'

        t = Timing('Saving json to %s' % fn)
        with open(fn, 'w') as f:
            json.dump({key: getattr(self, key).tolist()
                for key in self.savelist},
                f)
        t.prt('File saved ( %s )' % sizeof_fmt(fn))
        t.finished()

    def load(self):
        path = os.path.join(self.wd, self.id_str)
        fn = os.path.join(path, surface_filename) + '.npz'

        t = Timing('Trying to load %s' % fn)
        try:
            with np.load(fn) as loaded:
                t.prt('File opened. Reading.')

                # checking for missing field
                mf =[key for key in self.savelist if key not in loaded]
                if len(mf) > 0:
                    t.finished('Missing field %s in file.' % mf)
                    return False

                for key in self.savelist:
                    setattr(self, key, loaded[key])

            self.update_sites_infos()
            t.finished('File loaded.')
            return True
        except FileNotFoundError:
            t.finished('File not found or invalid.')
            return False
        except Exception as e:
            t.finished(e)
            return False

    @property
    def uclen(self):
        return len(self.ucx)

    @property
    def stlen(self):
        return len(self.stx)

    @property
    def id_str(self):
        """ Unique identification"""
        return ('Hex-P' if self.periodic else 'Hex') + '-'.join([
            'R%g' % self.radius,
            'NN%g' % self.nn_radius,
            "S%s" % self.sites.id_str,
        ])

    def create_uc(self, radius, grid, corrections):
        t = Timing('Creating unit cell (radius=%g)' % radius)
        n = int(radius)*2 + 1

        (self.ucx,
         self.ucy,
         self.uci,
         self.ucj) = create_grid(n, n, grid, corrections)

        self.cut_uc(radius)
        t.finished()

    def cut_uc(self, radius):
        mask = geometric_cut_mask(self.ucx, self.ucy, radius)
        self.ucx = self.ucx[mask]
        self.ucy = self.ucy[mask]
        self.uci = self.uci[mask]
        self.ucj = self.ucj[mask]

    def create_sites(self, sites):
        t = Timing('Creating sites')
        (self.stx,
         self.sty,
         self.sts) = calculate_sites(self.ucx, self.ucy,
                                     self.uci, self.ucj, sites)

        self.sti = np.arange(self.stlen)
        self.update_sites_infos()
        t.finished()

    def update_sites_infos(self):
        # site energies
        t = Timing('Updating sites info (n=%g)' % self.stlen)
        self.stidx = [np.argwhere(self.sts == i)
                      for i in range(len(self.sites))]
        self.ste = np.zeros(self.stx.shape)
        for i, idx in enumerate(self.stidx):
            self.ste[idx] = self.sites[i].energy

        t.finished()

    def cut_st(self, radius):
        t = Timing('Cutting sites')
        mask = geometric_cut_mask(self.stx, self.sty, radius)
        self.stx = self.stx[mask]
        self.sty = self.sty[mask]
        self.sts = self.sts[mask]

        if self.nni is not None and self.nnr is not None:
            self.nni = self.nni[mask]
            self.nnr = self.nnr[mask]

        sti_temp = self.sti[mask]
        u, self.sti = np.unique(sti_temp, return_inverse=True)

        self.update_sites_infos()
        t.finished()

    def add_periodicity(self, radius=None, side=6, offangle=0):
        t = Timing('Adding symmetry')
        if radius is None:
            radius = min(
                (np.max(self.ucx) - np.min(self.ucx)),
                (np.max(self.ucy) - np.min(self.ucy))
            )

        theta = np.linspace(0, 2*np.pi*(side-1)/side, side) + offangle
        dx = radius*np.cos(theta)
        dy = radius*np.sin(theta)

        self.stx = np.concatenate((self.stx, *[self.stx + x for x in dx]))
        self.sty = np.concatenate((self.sty, *[self.sty + y for y in dy]))
        self.sts = np.concatenate([self.sts]*(side + 1))
        self.sti = np.concatenate([self.sti]*(side + 1))

        self.update_sites_infos()
        t.finished()

    def create_nn_list(self, nn_radius):
        (self.nni,
         self.nnr) = calculate_nn(self.stx, self.sty, nn_radius, self.sti)

    def uniformize_nn(self):
        """ Cut the NN list of all atoms to the same lenght """
        t = Timing('Uniformizing the number of NN')

        nn_len = np.array([nnr.size for nnr in self.nnr])
        unique = np.unique(nn_len)
        t.prt("%4s: %7s" % ('Nb', 'Count'))
        for u in unique:
            t.prt("%4d: %7d" % (u, np.count_nonzero(nn_len == u)))
        lmin = np.min(nn_len)
        lmax = np.max(nn_len)
        t.prt('Cutting NN to %d closest sites' % lmin)
        t.prt('Max cut count: %d -> %d)' % (lmax, lmin))

        self.nni = np.array([nn[:lmin] for nn in self.nni])
        self.nnr = np.array([nn[:lmin] for nn in self.nnr])

        t.finished()

    def scale(self, a):
        t = Timing('Scaling')
        self.a = a

        self.ucx *= a
        self.ucy *= a
        self.stx *= a
        self.sty *= a

        self.nnr *= a

        t.finished()

    def show(self, grid=True, sites=True,
             highlight=None,
             highlight_lst=None,
             ):
        if grid:
            plt.scatter(
                self.ucx,
                self.ucy,
                30, 'black',  marker=(6, 0, 0))

        if sites:
            for i, idx in enumerate(self.stidx):
                color = self.sites[i].color
                plt.scatter(
                    self.stx[idx],
                    self.sty[idx],
                    s=10, c=color, marker='.')

        if highlight_lst is not None:
            for h in highlight_lst:
                color = 'green'
                plt.scatter(
                    self.stx[h],
                    self.sty[h],
                    s=80, c=color, marker='.')

        if highlight is not None:
            color = 'yellow'
            plt.scatter(
                self.stx[highlight],
                self.sty[highlight],
                s=120, c=color, marker='.')

        plt.axis('equal')
        plt.show()


