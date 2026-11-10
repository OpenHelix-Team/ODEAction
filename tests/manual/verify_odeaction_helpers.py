"""ODEAction utilities."""

import jax.numpy as jnp

from openpi.models.pi0 import make_field_attention_mask
from openpi.models.pi0 import make_field_positions
from openpi.models.pi0 import posemb_sincos_dense


def _assert_all(name: str, value):
    if not bool(jnp.all(value)):
        raise AssertionError(f"{name} failed")


def _assert_none(name: str, value):
    if bool(jnp.any(value)):
        raise AssertionError(f"{name} failed")


def main():
    prefix_mask = jnp.array(
        [
            [True, True, True, False],
            [True, True, True, True],
        ],
        dtype=jnp.bool_,
    )
    suffix_mask = jnp.ones((2, 3), dtype=jnp.bool_)

    attn = make_field_attention_mask(prefix_mask, suffix_mask)
    if attn.shape != (2, 7, 7):
        raise AssertionError(f"Unexpected attn shape: {attn.shape}")

    # prefix rows cannot attend suffix columns.
    _assert_none("prefix->suffix must be false", attn[:, :4, 4:])

    # suffix rows can attend valid prefix columns.
    _assert_all("suffix->valid prefix sample0", attn[0, 4:, :3])
    _assert_none("suffix->invalid prefix sample0", attn[0, 4:, 3:4])
    _assert_all("suffix->valid prefix sample1", attn[1, 4:, :4])

    # suffix rows can attend self, not other suffix tokens.
    suffix_block = attn[:, 4:, 4:]
    eye = jnp.eye(3, dtype=jnp.bool_)[None, :, :]
    _assert_all("suffix self attention must be true", suffix_block == eye)

    positions = make_field_positions(prefix_mask, suffix_mask, mode="first_suffix")
    expected_positions = jnp.array(
        [
            [0, 1, 2, 2, 3, 3, 3],  # sample0 prefix_len=3, padded prefix position remains 2; suffix all 3
            [0, 1, 2, 3, 4, 4, 4],  # sample1 prefix_len=4; suffix all 4
        ],
        dtype=positions.dtype,
    )
    if not bool(jnp.all(positions == expected_positions)):
        raise AssertionError(f"Unexpected positions:\n{positions}\nexpected:\n{expected_positions}")

    t_seq = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=jnp.float32)
    emb = posemb_sincos_dense(t_seq, embedding_dim=8, min_period=4e-3, max_period=4.0)
    if emb.shape != (2, 3, 8):
        raise AssertionError(f"Unexpected dense time embedding shape: {emb.shape}")

    print("✅ Pi05ODEAction helper checks passed")
    print("attn shape:", attn.shape)
    print("positions:\n", positions)
    print("time embedding shape:", emb.shape)


if __name__ == "__main__":
    main()
