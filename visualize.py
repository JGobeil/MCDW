from multiprocessing import Process
from multiprocessing import Queue
from types import SimpleNamespace
import numpy as np
import pandas as pd
import os
from glob import glob

import json

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import RegularPolygon

from timing import Timing

default_colors_conf = {
    'fcc1': '#2874A6',
    'fcc2': '#3498DB',
    'fcc3': '#85C1E9',
    'hcp1': '#922B21',
    'hcp2': '#C0392B',
    'hcp3': '#D98880',
}

class CreateImages:
    def __init__(self, read_path, write_path,
                 symbol=None,
                 symbol_scale=0.9,
                 colors_conf=None,
                 ):
        self.process = Process(
            target=CreateImages._wait_for_message,
            args=(self, ))

        self.queue = Queue()

        self.surface = None

        self.read_path = read_path
        self.write_path = write_path
        os.makedirs(write_path, exist_ok=True)

        self.symbol = symbol or 'hexagon'
        self.symbol_scale = symbol_scale
        self.colors_conf = colors_conf or default_colors_conf

        self.figure = Figure(figsize=(10,10))
        self.canvas = FigureCanvasAgg(self.figure)
        self.axis = self.figure.add_subplot(111)

    def _wait_for_message(self):
        # canvas don't work in multiprocess !!
        self.canvas = FigureCanvasAgg(self.figure)

        running = True
        while running:
            msg = self.queue.get()
            if msg == 'QUIT':
                running = False
            else:
                self.create_image(msg)

    def do_all(self):
        files = sorted(glob(os.path.join(self.read_path, 'occ_*.npy')))

        t = Timing('Creating images from files in %s' % self.read_path,
                   nbsteps=len(files))
        for f in files:
            self.create_image(int(os.path.basename(f)[4:-4]))
            t.tic()
        t.finished()

    def load_surface(self):
        with np.load(os.path.join(self.read_path, 'init.npz')) as loaded:
            surface = SimpleNamespace(
                x=loaded['sites_x'],
                y=loaded['sites_y'],
                names=loaded['sites_name'],
                a=loaded['a'],
            )
        surface.colors = [
            self.colors_conf[name] for name in surface.names
        ]

        if self.symbol == 'hexagon':
            surface.symbols = np.array([
                RegularPolygon(
                    (surface.x[i], surface.y[i]),
                    6, # number of vertices
                    surface.a*self.symbol_scale, # radius
                    color=surface.colors[i],
                    #orientation=np.pi/6,
                    )
                for i in range(len(surface.names))])

        s = 1.1
        surface.xlim = (surface.x.min()*s, surface.x.max()*s)
        surface.ylim = (surface.y.min()*s, surface.y.max()*s)

        self.surface = surface

    def create_image(self, i):
        if self.surface is None:
            self.load_surface()

        occ = np.load(os.path.join(self.read_path, 'occ_%.10d.npy' % i))
        with open(os.path.join(self.read_path, "lap_%.10d.json" % i), 'r') as f:
            stat = json.load(f)

        if self.symbol == 'scatter':
            x = self.surface.x[occ]
            y = self.surface.y[occ]
            c = self.surface.colors[occ]
            self.axis.scatter(x, y, s=10, c=c, marker='.')
        elif self.symbol == 'hexagon':
            for symbol in self.surface.symbols[occ]:
                self.axis.add_patch(symbol)
        else:
            print('Unknown symbol: %s' % self.symbol)
            return

        self.axis.set_xlim(self.surface.xlim)
        self.axis.set_ylim(self.surface.ylim)

        fmt = '{:15s} {:>12s}'.format
        values_left = [
            ('Lap:', '%-d' % stat['Lap']),
            ('Temperature:', '%-.2f K' % stat['T']),
            ('Energy:', '%-.6g eV' % stat['Energy']),
            ('Coverage:', '%-.4g' % stat['Coverage']),
        ]

        values_right = [
            ('Atoms:', '%-d' % stat['Atoms']),
            ('Moves tried:', '%-d' % stat['Moves tried']),
            ('Moved:', '%-d' % stat['Moved']),
            ('Not moved:', '%-d' % stat['Not moved']),
        ]

        if stat['Error adds'] > 0:
            values_right.append(('Error adds', '-%d' % stat['Error adds']))

        if stat['Error moves'] > 0:
            values_right.append(('Error moves', '-%d' % stat['Error moves']))

        self.axis.set_title(
            '\n'.join([fmt(*v) for v in values_left]),
            fontname='monospace',
            loc='left')

        self.axis.set_title(
            '\n'.join([fmt(*v) for v in values_right]),
            fontname='monospace',
            loc='right')

        fn = os.path.join(self.write_path, "%.10i.png" % i)
        self.figure.savefig(fn)
        self.axis.clear()


