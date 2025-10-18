from inspect import isfunction
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum

from ..utils import is_torch_version, maybe_allow_in_graph
from .attention import AdaLayerNorm, AdaLayerNormZero, Attention, FeedForward


@maybe_allow_in_graph
class BasicTransformerBlock2(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        # if cross_attention_dim is not None or double_self_attention:
        #     # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
        #     # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
        #     # the second cross attention block.
        #     self.norm2 = (
        #         AdaLayerNorm(dim, num_embeds_ada_norm)
        #         if self.use_ada_layer_norm
        #         else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        #     )
        #     self.attn2 = Attention(
        #         query_dim=dim,
        #         cross_attention_dim=cross_attention_dim if not double_self_attention else None,
        #         heads=num_attention_heads,
        #         dim_head=attention_head_dim,
        #         dropout=dropout,
        #         bias=attention_bias,
        #         upcast_attention=upcast_attention,
        #     )  # is self-attn if encoder_hidden_states is none
        # else:
        self.norm2 = None
        self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            mult=1,
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        posemb: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            posemb=posemb,  # todo in self attn, posemb shoule be [pose_in, pose_in]?
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                posemb=posemb,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(hid_slice)
                    for hid_slice in norm_hidden_states.chunk(
                        num_chunks, dim=self._chunk_dim
                    )
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


@maybe_allow_in_graph
class BasicTransformerBlock3(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
    ):
        super().__init__()

        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward2(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


def get_key_value(context, kernel_size=1):
    """
    context: embedded layout maps: (b, n, context_dim, res, res)
    kernel_size: sampled area width
    """
    key_values = []
    *_, h, w = context.shape
    padded_context = nn.functional.pad(
        context,
        (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
    )
    for i in range(kernel_size):
        for j in range(kernel_size):
            key_values.append(padded_context[..., i : i + h, j : j + h])
    key_value = torch.cat(key_values, dim=1)
    return key_value


class LayoutTransformer(nn.Module):
    """
    cross attn between image latents and layout data
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        n_blocks=1,
        kernel_size=1,
        dropout=0.0,
        context_dim=None,
        checkpoint=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.transformer_blocks = nn.ModuleList(
            [
                # BasicTransformerBlock2(
                #     dim=in_channels,
                #     num_attention_heads=n_heads,
                #     attention_head_dim=d_head,
                #     dropout=dropout,
                #     cross_attention_dim=context_dim,
                #     only_cross_attention=True,
                # )
                BasicTransformerBlock3(
                    in_channels,
                    n_heads=n_heads,
                    d_head=d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                )
                for _ in range(n_blocks)
            ]
        )
        self.gradient_checkpointing = checkpoint

    def forward(self, x, context):
        """
        x: shape (b, c, h, w)
        context: shape (bt, n, context_dim, h, w) or (b, t, n, context_dim, h, w)
        """
        if len(context.shape) == 6:
            context = rearrange(context, "b t n c h w -> (b t) n c h w")
        context = get_key_value(context, self.kernel_size)
        context = rearrange(context, "b n c h w -> (b h w) n c")

        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> (b h w) 1 c")

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                _x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    # None,  # attention_mask
                    context,
                    **ckpt_kwargs,
                )
            else:
                _x = block(x, context=context)
            x = x + _x
        x = rearrange(x, "(b h w) 1 c -> b c h w", h=h, w=w)
        return x


class LayoutTransformerObj(nn.Module):
    """
    cross attn between image latents and layout data
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        kernel_size=1,
        dropout=0.0,
        context_dim=None,
        checkpoint=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        n_block = 2
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock3(
                    in_channels,
                    n_heads=n_heads,
                    d_head=d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                )
                for _ in range(n_block)
            ]
        )
        self.gradient_checkpointing = checkpoint
        self.cam_proj = nn.Linear(default(context_dim, in_channels), in_channels)

    def forward(self, x, context):
        wall_context, obj_context, cam_emb = context
        x = self.wall_forward(x, wall_context)
        x = self.obj_forward(x, obj_context, cam_emb)
        return x

    def obj_forward(self, x, obj_context, cam_emb):
        """
        x: shape (bt, c, h, w)
        obj_context: shape (b, t, n, context_dim)
        cam_emb: shape (b, t, c)
        """
        b, t, n, context_dim = obj_context.shape
        _, in_dim, h, w = x.shape
        _x = rearrange(x, "(b t) c h w -> b t c h w", t=t)
        _x = _x + self.cam_proj(cam_emb)[..., None, None]
        _x = rearrange(_x, "b t c h w -> (b t) (h w) c")
        obj_context = rearrange(obj_context, "b t n c -> (b t) n c")
        block = self.transformer_blocks[1]
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            _x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                _x,
                # None,  # attention_mask
                obj_context,
                **ckpt_kwargs,
            )
        else:
            _x = block(_x, context=obj_context)
        _x = rearrange(_x, "bt (h w) c -> bt c h w", h=h, w=w)
        x = x + _x
        return x

    def wall_forward(self, x, context):
        """
        x: shape (b, c, h, w)
        context: shape (bt, n, context_dim, h, w) or (b, t, n, context_dim, h, w)
        """
        assert len(context.shape) == 6
        context = rearrange(context, "b t n c h w -> (b t) n c h w")
        context = get_key_value(context, self.kernel_size)
        context = rearrange(context, "b n c h w -> (b h w) n c")

        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> (b h w) 1 c")

        block = self.transformer_blocks[0]
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            _x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x,
                # None,  # attention_mask
                context,
                **ckpt_kwargs,
            )
        else:
            _x = block(x, context=context)
        x = x + _x
        x = rearrange(x, "(b h w) 1 c -> b c h w", h=h, w=w)
        return x


class LayoutTransformerRGBD(LayoutTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        n_blocks=1,
        kernel_size=1,
        dropout=0.0,
        context_dim=None,
        checkpoint=True,
        skip_depth=True,
    ):
        """
        skip_depth: dont apply layout attn with depth
        """
        super(LayoutTransformerRGBD, self).__init__(
            in_channels=in_channels,
            n_heads=n_heads,
            d_head=d_head,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
            context_dim=context_dim,
            checkpoint=checkpoint,
        )
        self.skip_depth = skip_depth

    def forward(self, x, context):
        """
        x: shape (bx2xt, c, h, w) rgb alternate with depth
        context: shape (b, t, n, context_dim, h, w)
        """

        if not self.skip_depth:
            context = repeat(context, "b t n c h w -> (b k) t n c h w", k=2)
            return super(LayoutTransformerRGBD, self).forward(x, context)
        else:
            t = context.shape[1]
            rgb, depth = rearrange(x, "(b k t) c h w -> k (b t) c h w", k=2, t=t)
            rgb = super(LayoutTransformerRGBD, self).forward(rgb, context)
            return rearrange([rgb, depth], "k (b t) c h w -> (b k t) c h w", t=t, k=2)


class FeedForward2(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        assert mask is None, "mask not implemented yet"

        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        # # if exists(mask):
        # #     mask = rearrange(mask, "b ... -> b (...)")
        # #     max_neg_value = -torch.finfo(sim.dtype).max
        # #     mask = repeat(mask, "b j -> (b h) () j", h=h)
        # #     sim.masked_fill_(~mask, max_neg_value)

        # # attention, what we cannot get enough of
        # attn = sim.softmax(dim=-1)

        # out = einsum("b i j, b j d -> b i d", attn, v)
        # replace with torch 2.0.1
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def exists(val):
    return val is not None
