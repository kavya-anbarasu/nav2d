

registry = {}


def make(id, *args, **kwargs):
    return registry[id](*args, **kwargs)


def register(id, obj=None):
    if obj is None:
        def wrap(obj):
            registry[id] = obj
            return obj

        return wrap
    else:
        registry[id] = obj
