import numpy as np
import numexpr as ne
from timing import Timing
from timing import TimingWithBatchEstimator


def create_grid(nx, ny, grid_def, corrections, center=True):
    with Timing(name="Creating base grid %dx%d" % (nx, ny)):
        n = nx * ny

        # create a square grid based on sizes of grid definition
        ind = np.indices((nx, ny))
        grid = ind * np.array(grid_def).reshape((2, 1, 1))

        # correct row/col to create special grid
        for dim, correction in enumerate(corrections):
            for c in correction:
                condition = (ind[dim] % c[0]) == (c[0]-1)
                grid += np.array((condition, condition)) * \
                    np.array(c[1]).reshape((2, 1, 1))

        # transform everything to a 1D array
        x = grid[0].reshape(n)
        y = grid[1].reshape(n)
        i = ind[0].reshape(n)
        j = ind[1].reshape(n)

        if center:
            x -= np.mean(x)
            y -= np.mean(y)

        return x, y, i, j


def calculate_sites(ucx, ucy, uci, ucj, sites_desc):
    with Timing('Adding sites'):

        # find the periodicity for each sites
        prd = [np.any([
            ((uci % rcmod[1]) == (rcmod[0] - 1))
            &
            ((ucj % rcmod[3]) == (rcmod[2] - 1))
            for rcmod in s.rcmod], axis=0).flatten()
               for s in sites_desc]

        # calculte x, y for each sites and put them in a 1D array
        x = np.concatenate(
            [ucx[p] + s.rel_x for p, s in zip(prd, sites_desc)])
        y = np.concatenate(
            [ucy[p] + s.rel_y for p, s in zip(prd, sites_desc)])

        # make a list with the indices of the corresponding sites
        last_i = 0
        idx = np.zeros(x.shape, dtype='uint8')
        for i, p in enumerate(prd):
            this_i = np.count_nonzero(p) + last_i
            idx[last_i:this_i] = i
            last_i = this_i

        return x, y, idx


def geometric_cut_mask(px, py, radius, side=6, offangle=np.pi/2):
    """ Work only for convex geometry """
    t = Timing('Cutting (N=%d, r=%g)' % (len(px), radius))
    theta = np.linspace(0, 2*np.pi, side+1) + offangle
    gx = radius*np.cos(theta)
    gy = radius*np.sin(theta)

    dx = np.expand_dims(px, 1) - np.expand_dims(gx, 0)
    dy = np.expand_dims(py, 1) - np.expand_dims(gy, 0)

    angles = np.diff(np.unwrap(np.arctan2(dy, dx), axis=1), axis=1)
    pt_in = np.invert(np.any(angles < 0.0, axis=1))

    n1 = len(px)
    n2 = np.count_nonzero(pt_in)

    t.prt('%d -> %d (%g%%)' % (n1, n2, n2/n1 * 100))

    t.finished()
    return pt_in


def geometric_cut_mask_old(px, py, radius, side=6, offangle=np.pi/2):
    """ special case need more work and condition"""
    with Timing('Cut'):
        theta = np.linspace(0, 2*np.pi, side+1) + offangle
        gx = radius*np.cos(theta)
        gy = radius*np.sin(theta)

        # gy = ga*gx + gy0
        ga = np.diff(gy)/np.diff(gx)
        gy0 = gy[:-1] - ga*gx[:-1]

        # py = pa*px + py0
        pa = 0.0
        py0 = py - pa*px

        # interception
        # pa*x' + py0 = ga*x' + gy0
        itrx = [(gy0[i] - py0)/(pa - ga[i]) for i in range(side)]
        itry = [pa*itrx[i] + py0 for i in range(side)]

        p_valid = [(itrx[i] >= px) & (itry[i] >= py)
                   for i in range(side)]

        crossing = []
        for i in range(side):
            ix = itrx[i]
            iy = itry[i]
            gx1, gx2 = sorted((gx[i], gx[i+1]))
            gy1, gy2 = sorted((gy[i], gy[i+1]))

            crossing.append(
                p_valid[i] &
                (ix > gx1) & (ix < gx2) &
                (iy > gy1) & (iy < gy2)
            )

        return (np.count_nonzero(crossing, axis=0) % 2) == 1


def calculate_nn(stx, sty, nn_radius, idx_list=None,
                 time_per_batch=10,
                 time_before_print=1):
    n = len(stx)

    nni = [None] * n  # indices of nn
    nnr = [None] * n  # distances of nn

    t = TimingWithBatchEstimator(name='NN calculation',
                                 nbsteps=n,
                                 target_batch_time=time_per_batch,
                                 time_before_first_print=time_before_print
                                 )

    outr = np.zeros(n, dtype=float)
    outless = np.zeros(n, dtype=bool)
    while t.batch_size > 0:
        for i in t.get_range():
            # calculate distance between i and all (numexp scale better)
            ne.evaluate('((x - stx)**2 + (y - sty)**2)**(0.5)',
                        local_dict={
                            'x': stx[i],
                            'y': sty[i],
                            'stx': stx,
                            'sty': sty,
                        }, out=outr)

            # keep only those inside nn_radius
            np.less(outr, nn_radius, out=outless)
            nn_in_id = np.flatnonzero(outless)
            nn_in_radius = outr[nn_in_id]

            # sort with radius (do not include i)
            argsort = np.argsort(nn_in_radius)[1:]

            if idx_list is None:
                nni[i] = nn_in_id[argsort]
            else:
                nni[i] = idx_list[nn_in_id[argsort]]
            nnr[i] = nn_in_radius[argsort]

        t.tic()
    t.finished()
    return np.array(nni), np.array(nnr)
