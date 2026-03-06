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

<<<<<<< HEAD
=======
## Install

Requires Python 3.10 or newer.

```bash
pip install maskx
```

>>>>>>> c804a833c72f66054c74d1d5f9dab8f425c295a1
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

## Example notebook

See `docs/notebooks/equinox_optax_demo.ipynb` for a small Equinox MLP example that uses `maskx` to train only selected parameters with Optax.

## Release

Build and validate locally:

```bash
python -m build
python -m twine check dist/*
```

Manual upload:

```bash
uv run --active twine upload dist/*
```

GitHub Actions can also publish via trusted publishing using `.github/workflows/publish.yml` once PyPI is configured to trust this repository.
