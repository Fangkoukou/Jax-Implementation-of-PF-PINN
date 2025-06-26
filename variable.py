import jax
import jax.numpy as jnp
from types import SimpleNamespace
from jax import config
config.update("jax_enable_x64", True)

@jax.tree_util.register_pytree_node_class
class Variable(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__()
        for key, val in kwargs.items():
            if val is not None:
                setattr(self, key, jnp.asarray(val, dtype=jnp.float64))
            else:
                setattr(self, key, val)

    def getval(self, *names):
        if len(names) == 0:
            values = self.__dict__.values()
        else:
            values = [getattr(self, name, None) for name in names]
        values_1D = [jnp.ravel(val) for val in values if val is not None]
        return jnp.concatenate(values_1D) if values_1D else jnp.array([], dtype=jnp.float64)

    def getname(self):
        return list(self.__dict__.keys())

    def getsize(self, *names):
        if len(names) == 0:
            values = self.__dict__.values()
        else:
            values = [getattr(self, name, None) for name in names]
        return jnp.asarray([0 if val is None else val.size for val in values], dtype=jnp.int32)

    def tree_flatten(self):
        children = tuple(self.__dict__.values())
        aux_data = tuple(self.__dict__.keys())
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls()
        for key, val in zip(aux_data, children):
            if val is not None:
                setattr(obj, key, jnp.asarray(val, dtype=jnp.float64))
            else:
                setattr(obj, key, val)
        return obj