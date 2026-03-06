from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax

PathPredicate = Callable[[str, Any], bool]
LeafType = type[Any] | tuple[type[Any], ...] | Callable[[Any], bool]


def _mask_tree(tree: Any) -> Any:
    return tree.tree if isinstance(tree, Mask) else tree


@dataclass(frozen=True)
class Mask:
    """Boolean mask pytree with mask algebra operators."""

    tree: Any

    def paths(self) -> list[str]:
        """Return the paths of all selected leaves."""
        return [path for path, selected in leaf_paths(self.tree) if bool(selected)]

    def matches(self) -> list[str]:
        """Alias for :meth:`paths`."""
        return self.paths()

    def count(self) -> int:
        """Return the number of selected leaves."""
        return len(self.paths())

    def __or__(self, other: Any) -> "Mask":
        return combine_masks(self, other, op="or")

    def __and__(self, other: Any) -> "Mask":
        return combine_masks(self, other, op="and")

    def __xor__(self, other: Any) -> "Mask":
        return combine_masks(self, other, op="xor")

    def __add__(self, other: Any) -> "Mask":
        return self | other

    def __sub__(self, other: Any) -> "Mask":
        other_tree = _mask_tree(other)
        return Mask(
            jax.tree_util.tree_map(
                lambda left, right: bool(left) and not bool(right),
                self.tree,
                other_tree,
            )
        )

    def __invert__(self) -> "Mask":
        return Mask(jax.tree_util.tree_map(lambda value: not bool(value), self.tree))


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
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(_mask_tree(tree))
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
    """Build a boolean mask pytree matching leaves selected by path or predicate.

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
    paths = set(path_in or ())
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(tree)
    mask_leaves = []

    for path, leaf in leaves_with_paths:
        path_str = _path_to_string(path)

        if not _matches_leaf_type(leaf, leaf_type):
            mask_leaves.append(False)
            continue

        if shape is not None and getattr(leaf, "shape", None) != shape:
            mask_leaves.append(False)
            continue

        if dtype is not None and getattr(leaf, "dtype", None) != dtype:
            mask_leaves.append(False)
            continue

        if ndim is not None and getattr(leaf, "ndim", None) != ndim:
            mask_leaves.append(False)
            continue

        if matcher is None and where is None and not prefixes and not paths:
            mask_leaves.append(True)
            continue

        matched_target = (
            matcher.search(path_str) is not None if matcher is not None else False
        )
        matched_where = where(path_str, leaf) if where is not None else False
        matched_prefix = any(path_str.startswith(prefix) for prefix in prefixes)
        matched_path_in = path_str in paths
        mask_leaves.append(
            bool(matched_target or matched_where or matched_prefix or matched_path_in)
        )

    return Mask(jax.tree_util.tree_unflatten(treedef, mask_leaves))


def combine_masks(*masks: Any, op: str = "or") -> Mask:
    """Combine boolean mask pytrees with a logical operator."""
    if not masks:
        raise ValueError("combine_masks requires at least one mask")

    mask_trees = [_mask_tree(mask) for mask in masks]

    if op == "or":

        def fn(*values: Any) -> bool:
            return any(bool(v) for v in values)

    elif op == "and":

        def fn(*values: Any) -> bool:
            return all(bool(v) for v in values)

    elif op == "xor":

        def fn(*values: Any) -> bool:
            return sum(bool(v) for v in values) % 2 == 1

    else:
        raise ValueError("op must be one of: 'or', 'and', 'xor'")

    return Mask(jax.tree_util.tree_map(fn, *mask_trees))
