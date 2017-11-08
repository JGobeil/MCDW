#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as np


cimport cython

np.import_array()

from libc.stdio cimport printf
from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libc.stdlib cimport rand
from libc.stdlib cimport srand
from libc.stdlib cimport RAND_MAX
from libc.math cimport exp
from libc.math cimport log

ctypedef unsigned char bool_t
ctypedef long long int_t
ctypedef double float_t


cdef class PotentialFunc:
    cdef float_t get_energy(self, int_t index):
        return 0.0

    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self, *args, **kwargs):
        pass

cdef class PotentialFuncNN(PotentialFunc):
    cdef int_t [:, :] nni  # list of nn index for each position
    cdef int_t [:] occ  # occupancy of the surface
    cdef float_t [:] nn_energy  # nn energies
    cdef int_t size   # number of nn to consider
    cdef int_t [:] nn_count  # number of nn at each level
    cdef float_t [:] binding_energy  # number of nn at each level

    cdef object sim

    def __cinit__(self, conf, int_t size = 39, *args, **kwargs):

        self.nn_count = np.array([3, 6, 3, 6, 6, 6, 6, 3, 6, 12, 3],
                                 dtype=np.int64)
        self.size = size
        self.nn_energy = np.zeros(self.size)

        cdef int_t i, j, n=0
        cdef float_t e
        for i, energy in enumerate(conf):
            for j in range(self.nn_count[i]):
                if n < self.size:
                    self.nn_energy[n] = energy
                    n += 1

    def set_surface(self, surface):
        self.nni = surface.nni
        self.binding_energy = np.zeros(surface.stlen)
        cdef int_t i
        for i in range(surface.stlen):
            self.binding_energy[i] = surface.sites[surface.sts[i]].energy

    def set_simulator(self, simulator):
        self.occ = simulator.occupancy_int64

    cdef float_t get_energy(self, int_t index):
        cdef float_t energy = 0.0

        cdef int_t* nni = &self.nni[index, 0]
        cdef int_t i, nn_index, count = 0

        for i in range(self.size):
            if self.occ[nni[i]] > 0:
                energy += self.nn_energy[i]
                count += 1
            if count == 6:
                break
        return energy + self.binding_energy[index]


cdef class CMonteCarloSimulator:
    # from initialisation
    cdef readonly int_t lap_max
    cdef readonly int_t steps_per_lap
    cdef readonly int_t moves_per_step
    cdef readonly float_t target_coverage
    cdef readonly int_t size
    cdef PotentialFunc potential_func

    # python object
    cdef public object occupancy_int64

    # c accessor
    cdef float_t [:] x
    cdef float_t [:] y
    cdef int_t [:, :] nni

    # simulation progress
    cdef readonly int_t n_max
    cdef readonly int_t n_used
    cdef readonly int_t n_free
    cdef public float_t kBT
    cdef readonly float_t energy
    cdef readonly float_t coverage
    cdef readonly int_t lap
    cdef readonly int_t attempted_moves
    cdef readonly int_t successful_moves
    cdef readonly int_t not_moved_moves
    cdef readonly int_t error_moves
    cdef readonly int_t error_adds

    # sites informations
    cdef int_t* filled
    cdef int_t* occ

    # cache (I'm not sure if this help)
    cdef int_t* _move_cache_nni
    cdef float_t* _move_cache_p

    cdef float_t* _energies_buffer

    def __cinit__(self,
                  int_t nb_binding_sites,
                  int_t lap_max,
                  int_t steps_per_lap,
                  int_t moves_per_step,
                  float_t target_coverage,
                  PotentialFunc potential_func,
                  *args, **kwargs
                  ):

        self.n_max = nb_binding_sites
        self.n_used = 0
        self.n_free = self.n_max
        self.coverage = 0.0
        self.target_coverage = target_coverage

        self.lap_max = lap_max
        self.steps_per_lap = steps_per_lap
        self.moves_per_step = moves_per_step

        self.energy = 0.0
        self.temperature = 0.0
        self.kBT = 0.0
        self.potential_func = potential_func

        self.attempted_moves = 0
        self.successful_moves = 0
        self.not_moved_moves = 0
        self.error_moves = 0
        self.error_adds = 0

        self.filled = <int_t*>malloc(self.n_max * sizeof(int_t))
        self.occ = <int_t*>malloc(self.n_max * sizeof(int_t))
        self._energies_buffer = <float_t*>malloc(self.n_max * sizeof(float_t))

        cdef int_t i
        for i in range(self.n_max):
            self.filled[i] = i
            self.occ[i] = -i - 1

        self._move_cache_nni = <int_t*>malloc(4 * sizeof(int_t))
        self._move_cache_p = <float_t*>malloc(4 * sizeof(float_t))


        # fixed seed for testing
        srand(201)

    def __dealloc__(self):
        free(self.filled)
        free(self.occ)
        free(self._move_cache_nni)
        free(self._move_cache_p)
        free(self._energies_buffer)

    def __init__(self,
                 surface,
                 *args, **kwargs):
        self.x = surface.stx
        self.y = surface.sty
        self.nni = surface.nni

        self.occupancy_int64 = np.PyArray_SimpleNewFromData(
            1, [self.n_max, ], np.NPY_INT64, self.occ)

    cpdef run_lap(self):
        cdef int_t step, move
        for step in range(self.steps_per_lap):
            if self.coverage < self.target_coverage:
                self.add_random_atom()
            for move in range(self.moves_per_step*self.n_used):
                self.move_random_atom()
        self.lap += 1

    cdef int_t filled_indexof(self, int_t site_index):
        return abs(self.occ[site_index]) - 1

    cdef void add_atom(self, int_t site_index):
        cdef int_t tmp_site_index = self.filled[self.n_used]
        cdef int_t filled_add_index = self.filled_indexof(site_index)

        self.occ[site_index] = (self.n_used + 1)
        self.occ[tmp_site_index] = -(filled_add_index + 1)

        self.filled[self.n_used] = site_index
        self.filled[filled_add_index] = tmp_site_index

        self.n_used += 1
        self.n_free -= 1
        self.coverage =  self.n_used / <float_t>self.n_max

    cdef void move_atom(self, int_t site_src, int_t site_dst):
        cdef int_t filled_src_index = self.filled_indexof(site_src)
        cdef int_t filled_dst_index = self.filled_indexof(site_dst)

        self.occ[site_src] = -(filled_dst_index + 1)
        self.occ[site_dst] =  (filled_src_index + 1)

        self.filled[filled_src_index] = site_dst
        self.filled[filled_dst_index] = site_src

    cdef void add_random_atom(self):
        # get a list of landing position available
        if self.n_free == 0:
            printf("!!! Surface totally saturated. Something is wrong !!!")
            self.error_adds += 10000
            return

        # chose a free site
        cdef int_t i, j
        cdef int_t site
        cdef int_t can_land = True
        for i in range(self.n_free):
            site = self.filled[(rand() % self.n_free) + self.n_used]
            for j in range(18):
                if self.occ[self.nni[site, j]] > 0:
                    can_land = False
                    break
            if can_land:
                self.add_atom(site)
                return
            can_land = True
        printf('!!! Cannot find a nice spot for landing. Atom rejected.\n !!!')
        self.error_adds += 1

    cdef int_t move_random_atom(self):
        self.attempted_moves += 1

        # chose an atom to move
        cdef int_t src = self.filled[rand() % self.n_used]

        # remove source temporary
        self.occ[src] = -self.occ[src]


        # local information about the atom
        cdef int_t* nni = self._move_cache_nni
        cdef float_t* p = self._move_cache_p

        # create a list of nn including origin
        nni[0] = src
        nni[1] = self.nni[src, 0]
        nni[2] = self.nni[src, 1]
        nni[3] = self.nni[src, 2]

        cdef int_t i, j
        # calculate the probability

        p[0] = self.potential_func.get_energy(src)
        for i in [1, 2, 3]:
            if self.occ[nni[i]] > 0:
                # site is occupied
                p[i] = 0
            else:
                # site is free
                #p[i] = exp(-self.potential_func.get_energy(j)/self.kBT)
                p[i] = exp(-(self.potential_func.get_energy(nni[i]) - p[0])
                    / self.kBT)

        #p[0] = 1
        p[1] += 1
        p[2] += p[1]
        p[3] += p[2]

        if p[3] == 0:
            printf('!!! No place to move !!!\n')
            self.error_moves += 1
            return -1


        p[0] = 1 / p[3]
        p[1] /= p[3]
        p[2] /= p[3]
        #p[3] /= p[3]

        # chose destination
        # get a random value between [0, 1]
        cdef float_t r = <float_t>rand() / <float_t>RAND_MAX
        cdef int_t dst
        if r > p[1]:
            if r > p[2]:
                dst = nni[3]
            else:
                dst = nni[2]
        else:
            if r > p[0]:
                dst = nni[1]
            else:
                dst = nni[0]

        if self.occ[dst] > 0:
            printf('!!! Moving to an occupied position !!!\n')
            printf("  %5d -> %5d\n", src, dst)
            printf("  p=[%7g %7g %7g %7g]\n", p[0], p[1], p[2], p[3])
            printf("  nni=[%5d %5d %5d %5d]\n", nni[0], nni[1], nni[2], nni[3])
            printf("  r=%g\n", r )
            for i in range(4):
                p[i] = self.potential_func.get_energy(nni[i])
            printf("  e=[%7g %7g %7g %7g]\n", p[0], p[1], p[2], p[3])
            self.error_moves += 1

        # put back the atom
        self.occ[src] = -self.occ[src]

        if src == dst:
            self.not_moved_moves += 1
        else:
            self.successful_moves += 1
            # move the atom
            self.move_atom(src, dst)

    cpdef float_t get_total_energy(self):
        cdef int_t size = self.n_used
        cdef float_t* buff = self._energies_buffer
        cdef int_t left = 0

        cdef int_t i
        for i in range(size):
            buff[i] = self.potential_func.get_energy(self.filled[i])

        # fancy sum for limiting rounding error
        while size > 1:
            left = size % 2
            size = size // 2
            for i in range(size):
                buff[i] += buff[i+size]
            if left:
                buff[size-1] += buff[2*size]
        return buff[0]

        #return np.sum(buff)

