from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np

PathPredicate = Callable[[str, Any], bool]
LeafType = type[Any] | tuple[type[Any], ...] | Callable[[Any], bool]


class Mask:
    """Boolean mask over pytree leaves backed by a flat NumPy bool array.

    Algebra operators (``|``, ``&``, ``^``, ``+``, ``-``, ``~``) execute as
    vectorised NumPy operations on the flat array, avoiding per-leaf Python
    overhead.  The full boolean pytree is only materialised when ``.tree`` is
    accessed.
    """

    __slots__ = ("_treedef", "_flat")

    def __init__(self, tree: Any) -> None:
        leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(tree)
        flat = np.array([bool(leaf) for _, leaf in leaves_with_paths], dtype=np.bool_)
        object.__setattr__(self, "_treedef", treedef)
        object.__setattr__(self, "_flat", flat)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Mask objects are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Mask objects are immutable")

    @classmethod
    def _from_flat(cls, treedef: Any, flat: np.ndarray) -> Mask:
        obj = object.__new__(cls)
        object.__setattr__(obj, "_treedef", treedef)
        object.__setattr__(obj, "_flat", flat)
        return obj

    # -- properties ----------------------------------------------------------

    @property
    def tree(self) -> Any:
        """Reconstruct the boolean pytree (materialised on each access)."""
        return self._treedef.unflatten(self._flat.tolist())

    # -- query ---------------------------------------------------------------

    def _all_paths(self) -> list[str]:
        dummy = self._treedef.unflatten(self._flat.tolist())
        return [
            _path_to_string(p)
            for p, _ in jax.tree_util.tree_flatten_with_path(dummy)[0]
        ]

    def paths(self) -> list[str]:
        """Return the paths of all selected leaves."""
        all_p = self._all_paths()
        return [all_p[i] for i in np.flatnonzero(self._flat)]

    def matches(self) -> list[str]:
        """Alias for :meth:`paths`."""
        return self.paths()

    def count(self) -> int:
        """Return the number of selected leaves."""
        return int(np.count_nonzero(self._flat))

    def summary(self) -> str:
        """Short description, e.g. ``'12/348 leaves selected'``."""
        return f"{np.count_nonzero(self._flat)}/{len(self._flat)} leaves selected"

    # -- transform -----------------------------------------------------------

    def apply(
        self,
        tree: Any,
        fn: Callable[[Any], Any],
        default: Callable[[Any], Any] = lambda x: x,
    ) -> Any:
        """Apply *fn* to selected leaves and *default* to the rest."""
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        new_leaves = [
            fn(leaf) if sel else default(leaf) for leaf, sel in zip(leaves, self._flat)
        ]
        return treedef.unflatten(new_leaves)

    # -- algebra -------------------------------------------------------------

    def _coerce(self, other: Any) -> Mask:
        if isinstance(other, Mask):
            return other
        return Mask(other)

    def _check_compat(self, other: Mask) -> None:
        if self._treedef != other._treedef:
            raise ValueError("Cannot combine masks from differently structured pytrees")

    def __or__(self, other: Any) -> Mask:
        other = self._coerce(other)
        self._check_compat(other)
        return Mask._from_flat(self._treedef, self._flat | other._flat)

    def __and__(self, other: Any) -> Mask:
        other = self._coerce(other)
        self._check_compat(other)
        return Mask._from_flat(self._treedef, self._flat & other._flat)

    def __xor__(self, other: Any) -> Mask:
        other = self._coerce(other)
        self._check_compat(other)
        return Mask._from_flat(self._treedef, self._flat ^ other._flat)

    def __add__(self, other: Any) -> Mask:
        return self | other

    def __sub__(self, other: Any) -> Mask:
        other = self._coerce(other)
        self._check_compat(other)
        return Mask._from_flat(self._treedef, self._flat & ~other._flat)

    def __invert__(self) -> Mask:
        return Mask._from_flat(self._treedef, ~self._flat)

    def __repr__(self) -> str:
        return f"Mask({self.summary()})"


# -- helpers -----------------------------------------------------------------


def _key_str(key: Any) -> str:
    if hasattr(key, "key"):
        return str(key.key)
    if hasattr(key, "idx"):
        return str(key.idx)
    if hasattr(key, "name"):
        return str(key.name)
    return str(key)


def _path_to_string(path: Sequence[Any]) -> str:
    return "/".join(_key_str(key) for key in path)


def _matches_leaf_type(leaf: Any, leaf_type: LeafType | None) -> bool:
    if leaf_type is None:
        return True
    if isinstance(leaf_type, type):
        return isinstance(leaf, leaf_type)
    if isinstance(leaf_type, tuple) and all(
        isinstance(item, type) for item in leaf_type
    ):
        return isinstance(leaf, leaf_type)
    return bool(leaf_type(leaf))


def leaf_paths(tree: Any) -> list[tuple[str, Any]]:
    """Return all leaf paths in a pytree as slash-joined strings."""
    if isinstance(tree, Mask):
        tree = tree.tree
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [(_path_to_string(path), leaf) for path, leaf in leaves_with_paths]


def select(
    tree: Any,
    target: str | None = None,
    *,
    where: PathPredicate | None = None,
    leaf_type: LeafType | None = None,
    shape: tuple[int, ...] | None = None,
    dtype: Any | None = None,
    ndim: int | None = None,
    path_prefix: str | Sequence[str] | None = None,
    path_in: Sequence[str] | None = None,
) -> Mask:
    """Build a boolean mask selecting pytree leaves by path or predicate.

    Args:
        tree: Input pytree.
        target: Regex pattern matched with ``re.search`` against leaf paths.
        where: Custom predicate receiving ``(path, leaf)``.
        leaf_type: Optional type filter or predicate applied before matching.
        shape: Optional exact shape filter for leaves with a ``shape`` attribute.
        dtype: Optional exact dtype filter for leaves with a ``dtype`` attribute.
        ndim: Optional exact ndim filter for leaves with an ``ndim`` attribute.
        path_prefix: Optional string or list of strings matched against path prefixes.
        path_in: Optional collection of exact paths.
    """
    if (
        target is None
        and where is None
        and leaf_type is None
        and shape is None
        and dtype is None
        and ndim is None
        and path_prefix is None
        and path_in is None
    ):
        raise ValueError(
            "select requires at least one selector: "
            "target, where, leaf_type, shape, dtype, ndim, path_prefix, or "
            "path_in"
        )

    matcher = re.compile(target) if target is not None else None
    prefixes = (
        (path_prefix,) if isinstance(path_prefix, str) else tuple(path_prefix or ())
    )
    paths_set = set(path_in or ())
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(tree)
    n = len(leaves_with_paths)
    flat = np.zeros(n, dtype=np.bool_)

    for i, (path, leaf) in enumerate(leaves_with_paths):
        path_str = _path_to_string(path)

        if not _matches_leaf_type(leaf, leaf_type):
            continue

        if shape is not None and getattr(leaf, "shape", None) != shape:
            continue

        if dtype is not None and getattr(leaf, "dtype", None) != dtype:
            continue

        if ndim is not None and getattr(leaf, "ndim", None) != ndim:
            continue

        if matcher is None and where is None and not prefixes and not paths_set:
            flat[i] = True
            continue

        if (
            (matcher is not None and matcher.search(path_str) is not None)
            or (where is not None and where(path_str, leaf))
            or any(path_str.startswith(prefix) for prefix in prefixes)
            or path_str in paths_set
        ):
            flat[i] = True

    if not np.any(flat):
        warnings.warn("select() matched zero leaves", stacklevel=2)

    return Mask._from_flat(treedef, flat)


def combine_masks(*masks: Any, op: str = "or") -> Mask:
    """Combine boolean mask pytrees with a logical operator."""
    if not masks:
        raise ValueError("combine_masks requires at least one mask")

    resolved = [m if isinstance(m, Mask) else Mask(m) for m in masks]
    treedef = resolved[0]._treedef

    for m in resolved[1:]:
        if m._treedef != treedef:
            raise ValueError("Cannot combine masks from differently structured pytrees")

    if op not in ("or", "and", "xor"):
        raise ValueError("op must be one of: 'or', 'and', 'xor'")

    np_op = {"or": np.bitwise_or, "and": np.bitwise_and, "xor": np.bitwise_xor}[op]
    result = resolved[0]._flat.copy()
    for m in resolved[1:]:
        np_op(result, m._flat, out=result)

    return Mask._from_flat(treedef, result)
