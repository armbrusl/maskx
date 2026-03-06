# maskx docs

`maskx` is a minimal library for inspecting JAX PyTree leaf paths and building boolean masks from those paths.

## Current API

- `maskx.leaf_paths(tree)` returns `(path, leaf)` pairs
- `maskx.select(tree, ...)` builds a `Mask`
- `mask.paths()` returns matched leaf paths
- `mask.count()` returns the number of matches
- masks support `|`, `&`, `^`, `+`, `-`, and `~`

## Example

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import maskx

class Model(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

model = Model(weight=jnp.ones((2, 2)), bias=jnp.zeros((2,)))
weight_mask = maskx.select(model, target=r"weight", leaf_type=jax.Array)
bias_mask = maskx.select(model, target=r"bias", leaf_type=jax.Array)
combined = weight_mask | bias_mask
paths = combined.paths()
```
