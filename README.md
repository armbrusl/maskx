# maskx


```bash
pip install maskx
```

Mask algebra for selecting and combining JAX PyTree leaves. Backed by flat NumPy arrays for fast operations on large trees.

```python
import jax
import maskx

weight = maskx.select(model, target=r".*/weight", leaf_type=jax.Array)
decoder = maskx.select(model, target=r"decoder/.*", leaf_type=jax.Array)

mask = decoder & weight
mask.paths()    # selected leaf paths
mask.count()    # number of selected leaves
mask.summary()  # "2/348 leaves selected"
```

Selectors: `target`, `path_prefix`, `path_in`, `leaf_type`, `shape`, `dtype`, `ndim`, `where`.

Operators: `|`, `&`, `^`, `+`, `-`, `~`

```python
a = maskx.select(model, target=r"decoder/.*", leaf_type=jax.Array)
b = maskx.select(model, target=r".*/weight", leaf_type=jax.Array)

a | b   # union — decoder leaves OR weights
a & b   # intersection — decoder weights only
a ^ b   # symmetric difference — in one but not both
a + b   # alias for union (a | b)
a - b   # difference — decoder leaves that are NOT weights
~a      # complement — everything except decoder leaves

# chain freely
trainable = (a | b) - maskx.select(model, target=r".*norm.*")

# cumulative: build up from multiple masks
masks = [maskx.select(model, path_prefix=p) for p in prefixes]
combined = masks[0]
for m in masks[1:]:
    combined = combined | m

# or via combine_masks
combined = maskx.combine_masks(*masks, op="or")   # "and", "xor" also supported
```

Apply a function to selected leaves only:

```python
mask.apply(model, fn=lambda x: x * 0)
```

Works with Optax:

```python
weight = maskx.select(model, target=r".*/weight", leaf_type=jax.Array)
optimizer = optax.masked(optax.adam(1e-3), weight.tree)
```

Works with Paramax:

```python
weight_mask = maskx.select(model, target="weight", leaf_type=jax.Array)
frozen = weight_mask.apply(model, fn=paramax.NonTrainable)
```

## Example notebook

See `docs/notebooks/equinox_optax_demo.ipynb` for a small Equinox MLP example that uses `maskx` to train only selected parameters with Optax.
