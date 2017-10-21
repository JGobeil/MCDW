import hashlib
import numpy as np

from lazy import lazy_property

class SitesGroup(list):

    @lazy_property
    def id_str(self):
        return hashlib.md5(
            ''.join([s.id_str for s in self]).encode()
        ).hexdigest()

    @classmethod
    def fcc_hcp_111(cls):
        fcc_conf = dict(
            rel_pos=(1/np.sqrt(3.), 0,),
            prob=1,
            energy=-1.803,
        )

        hcp_conf = dict(
            rel_pos=(.5/np.sqrt(3.), -0.5),
            prob=1,
            energy=-1.790,
        )

        fcc_name = ['fcc1', 'fcc2', 'fcc3']
        hcp_name = ['hcp1', 'hcp2', 'hcp3']

        fcc_rcmod = [
            [(1, 2, 3, 3), (2, 2, 1, 3), ],
            [(1, 2, 1, 3), (2, 2, 2, 3), ],
            [(1, 2, 2, 3), (2, 2, 3, 3), ],
        ]

        hcp_rcmod = fcc_rcmod

        fcc_color = [
            '#2874A6',
            '#3498DB',
            '#85C1E9',
        ]

        hcp_color = [
            '#922B21',
            '#C0392B',
            '#D98880',
        ]

        fcc_sites = [UnitCellSite(name=fcc_name[i],
                                  color=fcc_color[i],
                                  rcmod=fcc_rcmod[i],
                                  **fcc_conf,
                                  ) for i in range(3)]

        hcp_sites = [UnitCellSite(name=hcp_name[i],
                                  color=hcp_color[i],
                                  rcmod=hcp_rcmod[i],
                                  **hcp_conf,
                                  ) for i in range(3)]

        g = cls(fcc_sites + hcp_sites)
        g.id_str = 'fcc_hcp_111'
        return g


class UnitCellSite:
    def __init__(self, name, rel_pos, prob, energy,
                 rcmod=None, color=None):
        self.name = name
        self.rel_x = float(rel_pos[0])
        self.rel_y = float(rel_pos[1])
        self.prob = float(prob)
        self.energy = float(energy)
        self.color = color

        if rcmod is None:
            self.rcmod = [(1, 1, 1, 1), ]
        elif isinstance(rcmod[0], int):
            self.rcmod = [rcmod, ]
        else:
            self.rcmod = rcmod

        self.sites_per_uc = np.sum([1/(rc[1]*rc[3]) for rc in self.rcmod])

    @property
    def id_str(self):
        return "%f%f%s" % (self.rel_x, self.rel_y, self.rcmod)
