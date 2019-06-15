"""Backend module. Wraps dynet and pytorch."""

def get_backend_from_params(parameters):
    """Retrieve backend based on parameters. Returns wrapped backend class."""
    try:
        import dynet as dy
        if isinstance(parameters, dy.ParameterCollection):
            return get_backend_from_string("dynet")
    except ImportError:
        pass

    try:
        import torch
        if inspect.isgenerator(parameters):
            return get_backend_from_string("pytorch")
    except ImportError:
        pass

    raise ValueError("couldn't infer backend")


def get_backend_from_string(backend_name):
    """Retrieve backend based on parameters. Returns wrapped backend class."""
    if backend_name == "dynet":
        from uniparse.backend.dynet_backend import DynetBackend
        backend = DynetBackend()
    elif backend_name == "pytorch":
        from uniparse.backend.pytorch_backend import PyTorchBackend
        backend = PyTorchBackend()
    else:
        raise ValueError("backend doesn't exist: %s" % backend_name)
    return backend
