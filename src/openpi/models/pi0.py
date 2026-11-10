import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
import openpi.models.vlm_lora as _vlm_lora
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def posemb_sincos_dense(pos: jnp.ndarray, embedding_dim: int, min_period: float, max_period: float) -> jnp.ndarray:
    """Embed evolution time independently for every action query."""
    pos = jnp.asarray(pos)
    flat = jnp.reshape(pos, (-1,))
    emb = posemb_sincos(flat, embedding_dim, min_period=min_period, max_period=max_period)
    return jnp.reshape(emb, pos.shape + (embedding_dim,))


def make_field_attention_mask(prefix_mask: jnp.ndarray, suffix_mask: jnp.ndarray) -> jnp.ndarray:
    """Let each field query attend to the visual-language prefix and itself."""
    prefix_mask = jnp.asarray(prefix_mask, dtype=jnp.bool_)
    suffix_mask = jnp.asarray(suffix_mask, dtype=jnp.bool_)
    batch_size, prefix_len = prefix_mask.shape
    suffix_len = suffix_mask.shape[1]

    prefix_to_prefix = jnp.logical_and(prefix_mask[:, :, None], prefix_mask[:, None, :])
    prefix_to_suffix = jnp.zeros((batch_size, prefix_len, suffix_len), dtype=jnp.bool_)

    suffix_to_prefix = jnp.logical_and(suffix_mask[:, :, None], prefix_mask[:, None, :])
    suffix_eye = jnp.eye(suffix_len, dtype=jnp.bool_)[None, :, :]
    suffix_valid = jnp.logical_and(suffix_mask[:, :, None], suffix_mask[:, None, :])
    suffix_to_suffix = jnp.logical_and(suffix_eye, suffix_valid)

    top = jnp.concatenate([prefix_to_prefix, prefix_to_suffix], axis=-1)
    bottom = jnp.concatenate([suffix_to_prefix, suffix_to_suffix], axis=-1)
    return jnp.concatenate([top, bottom], axis=1)


def make_field_positions(
    prefix_mask: jnp.ndarray, suffix_mask: jnp.ndarray, *, mode: str = "first_suffix"
) -> jnp.ndarray:
    """Use the first suffix position for every independent field query."""
    if mode != "first_suffix":
        raise ValueError(f"Unsupported odeaction_position_mode: {mode!r}. Expected 'first_suffix'.")
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    prefix_len = jnp.sum(prefix_mask, axis=-1)
    suffix_positions = jnp.broadcast_to(prefix_len[:, None], suffix_mask.shape)
    return jnp.concatenate([prefix_positions, suffix_positions], axis=1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05

        self.use_odeaction = config.use_odeaction
        self.odeaction_correction_gain = config.odeaction_correction_gain
        self.odeaction_neighborhood_scale = config.odeaction_neighborhood_scale
        self.odeaction_use_correction = config.odeaction_use_correction
        self.odeaction_samples_per_segment = config.odeaction_samples_per_segment
        self.odeaction_loss_valid_dims = config.odeaction_loss_valid_dims
        self.ode_solver = "euler"
        self.integration_steps = 1

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        if config.vlm_lora.enabled:
            _vlm_lora.validate_model_config(
                config.vlm_lora, str(config.paligemma_variant), str(config.action_expert_variant)
            )
        if config.vlm_lora.enabled and config.vlm_lora.any_language_enabled():
            paligemma_config = dataclasses.replace(
                paligemma_config,
                lora_configs={
                    **paligemma_config.lora_configs,
                    **_vlm_lora.make_language_lora_configs(config.vlm_lora),
                },
            )
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
                vision_lora=(config.vlm_lora if config.vlm_lora.any_vision_enabled() else None),
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def _require_initial_action(self, observation):
        """Return the action-space initial condition used for training and rollout."""
        initial_action = observation.initial_action
        if initial_action is None:
            raise ValueError("ODEAction requires initial_action normalized with action statistics.")
        if initial_action.shape[-1] != self.action_dim:
            raise ValueError(f"initial_action width must be {self.action_dim}, got {initial_action.shape[-1]}.")
        return initial_action

    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, query_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(query_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))

        num_action_tokens = action_expert_tokens.shape[1]
        ar_mask += [True] + ([False] * (num_action_tokens - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.use_odeaction:
            return self._compute_loss_odeaction(rng, observation, actions, train=train)

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _neighborhood_scale_at(self, tau: jnp.ndarray) -> jnp.ndarray:
        """Fixed neighborhood scale throughout the evolution interval."""
        return jnp.full_like(tau, self.odeaction_neighborhood_scale)

    def _sample_evolution_times(self, rng: at.KeyArrayLike, batch_size: int, horizon: int):
        """Sample evolution times across the demonstrated interpolation segments."""
        m = self.odeaction_samples_per_segment
        if m <= 0:
            raise ValueError(f"odeaction_samples_per_segment must be positive, got {m}")

        seg = jnp.arange(horizon, dtype=jnp.int32).reshape(1, horizon, 1)
        seg = jnp.broadcast_to(seg, (batch_size, horizon, m))
        sub = jnp.arange(m, dtype=jnp.float32).reshape(1, 1, m)
        eta = jax.random.uniform(rng, (batch_size, horizon, m), dtype=jnp.float32)

        eta = eta * 0.998 + 0.001
        frac = (sub + eta) / float(m)
        tau_seq = (seg.astype(jnp.float32) + frac) / float(horizon)
        return (
            jnp.reshape(tau_seq, (batch_size, horizon * m)),
            jnp.reshape(seg, (batch_size, horizon * m)),
            jnp.reshape(frac, (batch_size, horizon * m)),
        )

    def _embed_field_queries(self, query_actions: jnp.ndarray, tau_seq: jnp.ndarray):
        """Embed query actions and their evolution times for the action expert."""
        if not self.pi05:
            raise ValueError("ODEAction requires pi05=True.")
        action_tokens = self.action_in_proj(query_actions)
        time_emb = posemb_sincos_dense(tau_seq, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        suffix_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        return action_tokens, suffix_mask, time_emb

    def _compute_loss_odeaction(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        """Directly supervise F_theta with progression and correction targets."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_size = actions.shape[0]
        horizon = actions.shape[1]

        initial_action = self._require_initial_action(observation)
        full_traj = jnp.concatenate([initial_action[:, None, :], actions], axis=1)  # (B, H+1, D)

        tau_seq, seg_idx, frac = self._sample_evolution_times(time_rng, batch_size, horizon)
        num_dense_tokens = tau_seq.shape[1]

        batch_idx = jnp.arange(batch_size)[:, None]
        a_left = full_traj[batch_idx, seg_idx]
        a_right = full_traj[batch_idx, seg_idx + 1]
        gamma_tau = a_left + frac[..., None] * (a_right - a_left)
        gamma_dot_tau = (a_right - a_left) * float(horizon)

        if self.odeaction_use_correction:
            k = self.odeaction_correction_gain
            sigma_tau = self._neighborhood_scale_at(tau_seq)
            noise = jax.random.normal(noise_rng, gamma_tau.shape) * sigma_tau[..., None]
            query_action = gamma_tau + noise
            target_field = gamma_dot_tau - k * (query_action - gamma_tau)
        else:
            query_action = gamma_tau
            target_field = gamma_dot_tau

        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, adarms_cond = self._embed_field_queries(query_action, tau_seq)

        attn_mask = make_field_attention_mask(prefix_mask, suffix_mask)
        positions = make_field_positions(prefix_mask, suffix_mask, mode="first_suffix")

        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        predicted_field = self.action_out_proj(suffix_out)  # (B, M, D)

        if self.odeaction_loss_valid_dims is not None:
            valid_dims = self.odeaction_loss_valid_dims
            predicted_field = predicted_field[..., :valid_dims]
            target_field = target_field[..., :valid_dims]

        loss = jnp.mean(jnp.square(predicted_field - target_field), axis=-1)  # (B, M)
        if loss.shape[1] != num_dense_tokens:
            raise ValueError(f"ODEAction loss shape mismatch: got {loss.shape}, expected second dim {num_dense_tokens}")
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        if self.use_odeaction:
            return self._sample_actions_odeaction(rng, observation)

        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _sample_actions_odeaction(self, rng, observation):
        """Integrate the learned field from the current action using runtime ODE options."""
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        H = self.action_horizon  # 50
        S = self.integration_steps
        total_steps = H * S
        sub_dt = 1.0 / total_steps

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        initial_action = self._require_initial_action(observation)
        a = initial_action[:, None, :]  # (B, 1, D)

        def predict_transition(a_current, tau_value):
            tau_batch = jnp.broadcast_to(tau_value, (batch_size,))
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, a_current, tau_batch
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            cross_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([cross_attn_mask, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return self.action_out_proj(suffix_out[:, -1:])

        if self.ode_solver == "rk4":

            def step_fn(carry, step_idx):
                a_cur = carry  # (B, 1, D)
                tau = step_idx.astype(jnp.float32) * sub_dt
                k1 = predict_transition(a_cur, tau)
                k2 = predict_transition(a_cur + (sub_dt / 2) * k1, tau + sub_dt / 2)
                k3 = predict_transition(a_cur + (sub_dt / 2) * k2, tau + sub_dt / 2)
                k4 = predict_transition(a_cur + sub_dt * k3, tau + sub_dt)
                a_next = a_cur + (sub_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                return a_next, a_next[:, 0, :]  # carry, collected (B, D)
        else:

            def step_fn(carry, step_idx):
                a_cur = carry  # (B, 1, D)
                tau = step_idx.astype(jnp.float32) * sub_dt
                v = predict_transition(a_cur, tau)
                a_next = a_cur + sub_dt * v  # (B, 1, D)
                return a_next, a_next[:, 0, :]  # carry, collected (B, D)

        _, all_actions = jax.lax.scan(step_fn, a, jnp.arange(total_steps))

        actions = all_actions[S - 1 :: S]  # (H, B, D)
        actions = jnp.transpose(actions, (1, 0, 2))  # (B, H, D)

        return actions
