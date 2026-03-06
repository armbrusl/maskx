# maskx

Minimal path-based masking for JAX PyTrees.

`maskx` builds `Mask` objects from pytree paths and simple mask algebra.

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

The library is intentionally small: it only builds and combines masks.
