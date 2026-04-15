from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .encoding import SparseEncoding


@dataclass(frozen=True)
class SparseTensorSpec:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    encoding: SparseEncoding

    def __post_init__(self):
        if len(self.shape) == 0:
            raise ValueError("shape must be non-empty")
        if not all(isinstance(dim, int) and dim >= 0 for dim in self.shape):
            raise ValueError("shape must be a tuple of non-negative ints")
        object.__setattr__(self, "dtype", jnp.dtype(self.dtype))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseTensor:
    buffers: tuple[jax.Array, ...]
    spec: SparseTensorSpec
    buffer_kinds: tuple[str, ...]

    def __post_init__(self):
        if len(self.buffers) == 0:
            raise ValueError("buffers must be non-empty")
        if len(self.buffers) != len(self.buffer_kinds):
            raise ValueError("buffers and buffer_kinds lengths must match")

    def tree_flatten(self):
        return self.buffers, (self.spec, self.buffer_kinds)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        spec, buffer_kinds = aux_data
        return cls(buffers=tuple(children), spec=spec, buffer_kinds=buffer_kinds)

    def buffer(self, name: str):
        for kind, value in zip(self.buffer_kinds, self.buffers, strict=True):
            if kind == name:
                return value
        raise KeyError(f"buffer kind '{name}' not found")
