# maskx


```bash
pip install maskx
```

Mask algebra for selecting and combining JAX PyTree leaves.

`maskx` builds `Mask` objects from PyTree leaves and supports selection by path, type, shape, dtype, ndim, exact path membership, and custom predicates.

```python
import jax
import maskx

weight = maskx.select(model, target=r".*/weight", leaf_type=jax.Array)
decoder = maskx.select(model, target=r"decoder/.*", leaf_type=jax.Array)

mask = decoder & weight
paths = mask.paths()
count = mask.count()
```

Selectors can be based on `target`, `path_prefix`, `path_in`, `leaf_type`, `shape`, `dtype`, and `ndim`.

Mask operators: `|`, `&`, `^`, `+`, `-`, `~`

Works with Optax:

```python
import jax
import optax
import maskx

weight = maskx.select(model, target=r".*/weight", leaf_type=jax.Array)
optimizer = optax.masked(optax.adam(1e-3), weight.tree)
```

Works with Paramax:

```python
import jax
import jax.tree_util as jtu
import maskx
import paramax

weight_mask = maskx.select(model, target="weight", leaf_type=jax.Array)

frozen = jtu.tree_map(
    lambda leaf, selected: paramax.NonTrainable(leaf) if selected else leaf,
    model,
    weight_mask.tree,
)
```

## Example notebook

See `docs/notebooks/equinox_optax_demo.ipynb` for a small Equinox MLP example that uses `maskx` to train only selected parameters with Optax.
