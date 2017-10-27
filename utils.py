from os.path import getsize

def sizeof_fmt(fn, suffix='B'):
    """ Return the size of a file 'fn' in a human readeable format.

     From https://stackoverflow.com/questions/1094841 """

    num = getsize(fn)
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
