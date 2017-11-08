from time import perf_counter
import numpy as np


def nice_s(*seconds):
    """Nice format for seconds with hours and minutes values if needed.

    Example:
        12.2 -> "12.2s"
        78.4 -> "1m 18.400s"
        4000 -> "1h  6m 40.0s"

    Parameters
    ----------
    *seconds: float
        value to convert

    Return
    ------
    single string or list of strings
    """
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
    """ Timing of excution time with multiple tic. Useful for logging also.
    After initialisation, you Timing.tic to print usefull informations. Can be use
    as a context manager.
    """
    level = 0  # depth of the timing (for display)

    def __init__(self,
                 name: str,
                 nbsteps: int=None,
                 tic_custom_fields: list=None):
        """ A new timing. Increase the level by 1 until finished is called.

        Parameters
        ----------
        name: str
            A name for this timer
        nbsteps: int
            If set, will be use to track the number of steps left todo with
            time estimation.
        tic_custom_fields: list
            If set, define new fields to be print for each tic. The format of
            each input is (field_name, format, lenght). The value must be pass
            when called tic.
        """
        self.name = name
        self.first_print = True

        if nbsteps is None:
            self.nbsteps = 0
            tic_format = [
                ('Done', '%6d', 8),
                ('TPU', '%9s', 10),
                ('Average TPU', '%9s', 10),
                ('Elapsed time', '%12s', 14), ]
            self.prt('%s' % self.name, start=True)
        else:
            self.nbsteps = nbsteps
            tic_format = [
                ('Done', '%6d', 8),
                ('Time left', '%12s', 14),
                ('Elapsed time', '%12s', 14),
                ('Total time', '%12s', 14),
                ('TPU', '%8s', 10),
                #('Average TPU', 's', 10),
            ]
            self.prt('%s (steps = %d)' % (self.name, self.nbsteps),
                     start=True)

        if tic_custom_fields is not None:
            tic_format = tic_custom_fields

        self.header = ' '.join([(('%%%ds' % size) % name)
                                for name, fmt, size in tic_format])

        self.tic_format = tic_format

        self.tics = [perf_counter(), ]
        Timing.level += 1

    def get_tic_str(self, params):
        return ' '.join([
            ('%%%ds' % size) % (fmt % params[name]) for
            name, fmt, size in self.tic_format
        ])

    @property
    def done(self):
        """ Number of steps completed. """
        return len(self.tics) - 1

    @property
    def todo(self):
        """ Number of steps left to do."""
        return self.nbsteps - self.done

    @property
    def elapsed_time(self):
        """ Time elapsed since the timer start."""
        return self.tics[-1] - self.tics[0]

    @property
    def last_tpu(self):
        """ Time elapsed since the last tic. (tpu=time per unit)"""
        return self.tics[-1] - self.tics[-2]

    @property
    def average_tpu(self):
        """ Average time per unit."""
        return self.elapsed_time / self.done

    @property
    def tpu(self):
        """ The tpu median value."""
        return np.median(np.diff(self.tics))

    @property
    def est_time_left(self):
        """ Estimation of the time left."""
        return self.tpu * self.todo

    @property
    def est_total_time(self):
        """ Estimation of the total time."""
        return self.est_time_left + self.elapsed_time

    def tic(self, **kwargs):
        """ Clock tic. Print info about the timing."""
        self.tics.append(perf_counter())

        stat = self.stat
        stat.update(**kwargs)

        if self.first_print:
            self.first_print = False
            self.prt(self.header)

        self.prt(self.get_tic_str(stat))

    @property
    def stat(self):
        """ Statistics about the current step."""
        return {
            'Done': self.done,
            'Todo': self.todo,
            'Time left': nice_s(self.est_time_left),
            'Elapsed time': nice_s(self.elapsed_time),
            'Total time': nice_s(self.est_total_time),
            'TPU': nice_s(self.last_tpu),
            'Average TPU': nice_s(self.average_tpu)
        }

    @classmethod
    def prt(cls, msg, start=False, end=False):
        """ Print function with visual indentation corresponding to the level."""
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

    def finished(self, msg=None):
        """ Call this when finished."""
        if msg is None:
            self.prt('Done in %s' % nice_s(perf_counter() - self.tics[0]),
                     end=True)
        else:
            self.prt(msg + 'Done in %s' % nice_s(perf_counter() - self.tics[0]),
                     end=True)
        Timing.level -= 1

    def __enter__(self):
        """ Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Exit the context manager."""
        self.finished()
        return True


class TimingWithBatchEstimator(Timing):
    """ Timing with estimator of batch size. Use to keep regular update on
    the calculation, but not too much."""

    def __init__(self,
                 name,
                 nbsteps=None,
                 tic_custom_fields=None,
                 time_before_first_print=5,
                 target_batch_time=10,
                 first_batch_size=1,
                 batch_size=None,
                 ):
        self.tbfp = time_before_first_print
        self.tbt = target_batch_time
        self.bs = batch_size
        self.batchs = [first_batch_size, ]

        super().__init__(
                name=name,
                nbsteps=nbsteps,
                tic_custom_fields=tic_custom_fields)

    @property
    def done(self):
        return sum(self.batchs[:-1])

    @property
    def batch_size(self):
        return self.batchs[-1]

    @property
    def last_tpu(self):
        return (self.tics[-1] - self.tics[-2])/self.batchs[-2]

    def get_range(self):
        return range(self.done, self.done + self.batch_size)

    def tic(self, **kwargs):
        """ Clock tic. Print info about the timing."""
        self.tics.append(perf_counter())

        if self.bs is not None:
            est = self.bs
        else:
            est = int(np.ceil(
                self.tbt/((self.tics[-1] - self.tics[-2])/self.batchs[-1])))

        self.batchs.append(est)
        if self.nbsteps is not None and self.batch_size > self.todo:
            self.batchs[-1] = self.todo

        stat = self.stat
        stat.update(**kwargs)

        if self.elapsed_time > self.tbfp:
            if self.first_print:
                self.first_print = False
                self.prt(self.header)

            self.prt(self.get_tic_str(stat))
