from time import perf_counter
import numpy as np


def nice_s(*seconds):
    if len(seconds) > 1:
        return [nice_s(s) for s in seconds]
    else:
        s = seconds[0]
    if s < 60:
        return "%.3gs" % s
    m, s = divmod(s, 60)
    if m < 60:
        return "%2dm %.3fs" % (m, s)
    h, m = divmod(m, 60)
    return "%dh %2dm %.1fs" % (h, m, s)


class Timing:
    level = 0

    def __init__(self,
                 name,
                 total_steps=None,
                 tic_custom_fields=None,
                 time_before_print=5,
                 target_batch_time=10,
                 first_batch=1,
                 batch_size=None,
                 ):
        self.name = name
        self.N = total_steps
        self.tbp = time_before_print
        self.tbt = target_batch_time
        self.bs = batch_size

        self.tpu = 1

        self.first_print = True
        self.done = 0
        self.todo = self.N
        self.elapsed = 0
        self.left = 0
        self.total = 0

        if self.N is None:
            tic_format = [
                ('Done', 'd', 8),
                ('Elapsed', 's', 14),
                ('TPU', 's', 10),
            ]
        else:
            tic_format = [
                ('Todo', 'd', 8),
                ('Done', 'd', 8),
                ('Elapsed', 's', 14),
                ('Left', 's', 14),
                ('Total', 's', 14),
                ('TPU', 's', 10)
            ]

        if tic_custom_fields is not None:
            tic_format += tic_custom_fields

        self.header = ' '.join([(('%%%ds' % size) % name)
                                for name, typpe, size in tic_format])

        self.tic_fmt = ' '.join([('{%s:>%d%s}' % (name, size, typpe))
                                 for name, typpe, size in tic_format]).format

        self.tics = [perf_counter(), ]
        self.batchs = [first_batch, ]

        if self.N is None:
            self.prt('%s' % self.name, start=True)
        else:
            self.prt('%s (N=%d)' % (self.name, self.N), start=True)
        Timing.level += 1

    @classmethod
    def prt(cls, msg, start=False, end=False):
        if start:
            if cls.level == 0:
                print('─┬─ ' + msg, flush=True)
            else:
                print(' │ ' * (cls.level-1) + ' ├──┬─ ' + msg,
                      flush=True)
            return
        if end:
            if cls.level < 2:
                print('─┴─ ' + msg, flush=True)
            else:
                print(' │ ' * (cls.level-2) + ' ├──┴─ ' + msg,
                      flush=True)
            return
        print(' │ ' * cls.level + msg, flush=True)

    def get_range(self):
        return range(self.done, self.done + self.batch_size)

    @property
    def batch_size(self):
        return self.batchs[-1]

    def estimate_next_batch_size(self):
        if self.bs is not None:
            est = self.bs
        else:
            est = int(np.ceil(self.tbt/self.tpu))
        if self.N is None:
            self.batchs.append(est)
        else:
            self.batchs.append(min((est, self.todo)))

    def tic(self, **kwargs):
        self.tics.append(perf_counter())

        self.done += self.batch_size
        self.elapsed = self.tics[-1] - self.tics[0]
        self.tpu = (self.tics[-1] - self.tics[-2]) / self.batch_size

        if self.N is not None:
            self.todo -= self.batch_size
            self.left = self.tpu * self.todo
            self.total = self.elapsed + self.left

        self.estimate_next_batch_size()

        if self.elapsed > self.tbp:
            if self.first_print:
                self.first_print = False
                self.prt(self.header)
            d = {'Done': self.done,
                 'Todo': self.todo,
                 'Elapsed': nice_s(self.elapsed),
                 'Left': nice_s(self.left),
                 'Total': nice_s(self.total),
                 'TPU': nice_s(self.tpu),
                 }
            d.update(**kwargs)
            self.prt(self.tic_fmt(**d))

    def finished(self, msg=None):
        if msg is None:
            self.prt('Done in %s' % nice_s(perf_counter() - self.tics[0]),
                     end=True)
        else:
            self.prt(msg + ' ' + nice_s(perf_counter() - self.tics[0]),
                     end=True)
        Timing.level -= 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finished()
        return True
