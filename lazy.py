"""Decorator for lazy evaluated properties"""

class lazy_property:
    """ A decorator to create a lazy evaluated property, i.e. only evaluated
    on the first call.

    When used as a function decorator, this will create an attribute that
    will be evaluated on the first call and returned. On the next calls the
    previously evaluated value will be directly return without reevaluation.
    The value can be deleted. In this case, the property will be reevaluated
    on the next call.
    """

    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value
