import warnings

import equinox as eqx
import jax
import jax.numpy as jnp

import maskx


class Block(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray


class Model(eqx.Module):
    encoder: Block
    decoder: Block
    name: str


def make_model() -> Model:
    return Model(
        encoder=Block(weight=jnp.ones((2, 2)), bias=jnp.zeros((2,))),
        decoder=Block(weight=2 * jnp.ones((2, 2)), bias=jnp.ones((2,))),
        name="demo",
    )


def test_leaf_paths_exposes_attribute_paths():
    model = make_model()
    paths = {path for path, _ in maskx.leaf_paths(model)}

    assert "encoder/weight" in paths
    assert "encoder/bias" in paths
    assert "decoder/weight" in paths
    assert "name" in paths


def test_select_matches_regex_on_paths():
    model = make_model()
    mask = maskx.select(model, target=r"encoder/.*", leaf_type=jax.Array)
    selected = set(mask.paths())

    assert selected == {"encoder/weight", "encoder/bias"}


def test_apply_selected_updates_only_matching_leaves():
    model = make_model()
    decoder_mask = maskx.select(model, target=r"decoder/weight", leaf_type=jax.Array)
    bias_mask = maskx.select(model, target=r".*bias", leaf_type=jax.Array)
    combined = decoder_mask | bias_mask

    selected = set(combined.paths())

    assert selected == {"decoder/weight", "encoder/bias", "decoder/bias"}


def test_combine_masks_supports_and():
    model = make_model()
    decoder_mask = maskx.select(model, target=r"decoder/.*", leaf_type=jax.Array)
    weight_mask = maskx.select(model, target=r".*/weight", leaf_type=jax.Array)
    combined = decoder_mask & weight_mask

    selected = set(combined.paths())

    assert selected == {"decoder/weight"}


def test_select_can_match_non_array_leaf_types():
    model = make_model()
    mask = maskx.select(model, target=r"name", leaf_type=str)

    selected = set(mask.paths())

    assert selected == {"name"}


def test_select_can_match_by_shape_only():
    model = make_model()
    mask = maskx.select(model, shape=(2, 2))

    selected = set(mask.paths())

    assert selected == {"encoder/weight", "decoder/weight"}


def test_select_can_match_by_dtype_only():
    model = make_model()
    model = eqx.tree_at(
        lambda m: m.decoder.weight, model, model.decoder.weight.astype(jnp.bfloat16)
    )
    mask = maskx.select(model, dtype=jnp.bfloat16)

    selected = set(mask.paths())

    assert selected == {"decoder/weight"}


def test_select_combines_target_with_shape_filter():
    model = make_model()
    mask = maskx.select(model, target=r"decoder/.*", shape=(2,))

    selected = set(mask.paths())

    assert selected == {"decoder/bias"}


def test_mask_algebra_supports_difference_and_invert():
    model = make_model()
    decoder_mask = maskx.select(model, target=r"decoder/.*", leaf_type=jax.Array)
    weight_mask = maskx.select(model, target=r".*/weight", leaf_type=jax.Array)

    diff = decoder_mask - weight_mask
    inverted = ~weight_mask

    diff_selected = set(diff.paths())
    inverted_selected = set(inverted.paths())

    assert diff_selected == {"decoder/bias"}
    assert {"encoder/bias", "decoder/bias", "name"}.issubset(inverted_selected)


def test_select_supports_custom_predicate():
    model = make_model()
    mask = maskx.select(
        model,
        where=lambda path, leaf: eqx.is_array(leaf) and getattr(leaf, "ndim", 0) == 1,
    )
    selected = set(mask.paths())

    assert selected == {"encoder/bias", "decoder/bias"}


def test_mask_count_reports_number_of_matches():
    model = make_model()
    mask = maskx.select(model, target=r".*bias", leaf_type=jax.Array)

    assert mask.count() == 2


def test_select_supports_path_prefix():
    model = make_model()
    mask = maskx.select(model, path_prefix="decoder", leaf_type=jax.Array)

    assert set(mask.paths()) == {"decoder/weight", "decoder/bias"}


def test_select_supports_path_in():
    model = make_model()
    mask = maskx.select(model, path_in=["encoder/bias", "decoder/weight"])

    assert set(mask.paths()) == {"encoder/bias", "decoder/weight"}


def test_select_supports_ndim_filter():
    model = make_model()
    mask = maskx.select(model, leaf_type=jax.Array, ndim=1)

    assert set(mask.paths()) == {"encoder/bias", "decoder/bias"}


def test_summary_reports_selected_over_total():
    model = make_model()
    mask = maskx.select(model, target=r".*bias", leaf_type=jax.Array)

    assert mask.summary() == "2/5 leaves selected"


def test_repr_contains_summary():
    model = make_model()
    mask = maskx.select(model, target=r".*bias", leaf_type=jax.Array)

    assert repr(mask) == "Mask(2/5 leaves selected)"


def test_apply_transforms_selected_leaves():
    model = make_model()
    mask = maskx.select(model, target=r".*bias", leaf_type=jax.Array)

    result = mask.apply(model, lambda x: x * 0)

    assert jnp.allclose(result.encoder.bias, jnp.zeros((2,)))
    assert jnp.allclose(result.decoder.bias, jnp.zeros((2,)))
    assert jnp.allclose(result.encoder.weight, model.encoder.weight)


def test_select_warns_on_zero_matches():
    model = make_model()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mask = maskx.select(model, target=r"nonexistent_path")
        assert len(w) == 1
        assert "zero leaves" in str(w[0].message)
    assert mask.count() == 0


def test_combine_masks_rejects_different_structures():
    model_a = make_model()
    model_b = Block(weight=jnp.ones((2, 2)), bias=jnp.zeros((2,)))

    mask_a = maskx.select(model_a, leaf_type=jax.Array)
    mask_b = maskx.select(model_b, leaf_type=jax.Array)

    try:
        mask_a & mask_b
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "differently structured" in str(e)


def test_mask_is_immutable():
    model = make_model()
    mask = maskx.select(model, leaf_type=jax.Array)

    try:
        mask._flat = None
        assert False, "Expected AttributeError"
    except AttributeError:
        pass
