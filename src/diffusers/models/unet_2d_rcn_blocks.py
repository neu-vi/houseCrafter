"""
define rcn resblock and other bigger block here

"""

import copy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from cfg_util import instantiate_from_config
from einops import rearrange
from torch import Tensor

from ..utils import is_torch_version
from .dual_transformer_2d import DualTransformer2DModel
from .resnet import Downsample2D, ResnetBlock2D, Upsample2D
from .transformer_2d import Transformer2DModel


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class Embedder:
    """
    https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf_helpers.py#L15
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """
        inputs: B,C=3,*
        output: B,C_out,...
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], 1)


def get_embedder(multires, i=0):
    """
    https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf_helpers.py#L15
    """
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class RayEmbedding(nn.Module):
    """
    embed rcamera pose in Plucker coordinates
    todo: look at light field network code or nerf code
    """

    def __init__(self, d_bands: int, m_bands: int):
        super().__init__()
        self.d_embedder, d_dim = get_embedder(d_bands)
        self.m_embedder, m_dim = get_embedder(m_bands)
        self._out_channels = d_dim + m_dim

    def forward(self, ray_coords: Tensor) -> Tensor:
        """
        ray_coords: B,6,H,W the six channels are m,r
        """
        m_coords = ray_coords[:, :3]
        m_embed = self.m_embedder(m_coords)

        d_coords = ray_coords[:, 3:]
        d_embed = self.d_embedder(d_coords)
        return torch.cat([m_embed, d_embed], dim=1)


class PosEmbedding(nn.Module):
    def __init__(self, in_dim, bands=6):
        super().__init__()
        embed_kwargs = {
            "include_input": True,
            "input_dims": in_dim,
            "max_freq_log2": bands - 1,
            "num_freqs": bands,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        self.embedder = Embedder(**embed_kwargs)
        self._out_channels = self.embedder.out_dim

    def forward(self, x):
        """
        x shape: B,*,C
        return: B,*,C_out
        """
        x = rearrange(x, "B ... C -> B C ...")
        x = self.embedder.embed(x)
        x = rearrange(x, "B C ... -> B ... C")
        return x


class RayCondNorm(nn.Module):
    def __init__(
        self, ray_dim: int, input_dim: int, downsample: int = 1, eps: float = 1e-5
    ) -> None:
        super().__init__()

        self.ln = lambda x: (x - x.mean(dim=(-1, -2, -3), keepdim=True)) / (
            x.std(dim=(-1, -2, -3), keepdim=True) + eps
        )

        self.scale = nn.Conv2d(
            ray_dim, input_dim, kernel_size=downsample, stride=downsample
        )
        self.bias = nn.Conv2d(
            ray_dim, input_dim, kernel_size=downsample, stride=downsample
        )

        # init weight
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.bias.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.bias.bias)

    def forward(self, x, rays):
        """
        expect x of shape B,C,H,W
        """
        assert len(x.shape) == 4
        x = self.ln(x)
        scale = self.scale(rays)
        bias = self.bias(rays)

        x = x * (1.0 + scale) + bias
        return x


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert (
            merge_strategy in self.strategies
        ), f"merge_strategy needs to be in {self.strategies}"

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif (
            self.merge_strategy == "learned"
            or self.merge_strategy == "learned_with_images"
        ):
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            assert image_only_indicator is not None, "need image_only_indicator ..."
            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1"),
            )
            alpha = rearrange(alpha, self.rearrange_pattern)
        else:
            raise NotImplementedError
        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator)
        x = (
            alpha.to(x_spatial.dtype) * x_spatial
            + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        )
        return x


class RCNResnetBlock2D(ResnetBlock2D):
    def __init__(self, *arg, ray_dim: int, layout_cfg: Dict, mix_factor=0.95, **kwargs):
        super().__init__(*arg, **kwargs)
        self.rcn = RayCondNorm(input_dim=self.in_channels, ray_dim=ray_dim)
        self.use_layout = layout_cfg is not None
        if self.use_layout:
            layout_cfg = copy.deepcopy(layout_cfg)
            layout_cfg["params"]["in_channels"] = self.in_channels
            if layout_cfg["params"]["d_head"] == -1:
                layout_cfg["params"]["d_head"] = (
                    self.in_channels // layout_cfg["params"]["n_heads"]
                )
            self.layout = instantiate_from_config(layout_cfg)
            alpha = torch.log(torch.tensor(mix_factor / (1.0 - mix_factor)))
            self.layout_mixer = AlphaBlender(alpha=alpha, merge_strategy="learned")

    def forward(self, input_tensor, temb, rays, layouts=None):
        input_tensor = self.rcn(input_tensor, rays)
        if self.use_layout:
            x_layout = self.layout(input_tensor, layouts)
            input_tensor = self.layout_mixer(input_tensor, x_layout)
        input_tensor = super().forward(input_tensor, temb)
        return input_tensor


class RCNCrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        ray_dim=78,
        layout_cfg=None,
        pos_dim=-1,
        rgbd_cross_attention=False,
        depth_block=False,
    ):
        super().__init__()

        self.use_depth_block = depth_block
        if depth_block:
            self.add_depth_block(
                in_channels * 2,
                in_channels,
                temb_channels,
                resnet_eps,
                resnet_groups,
                dropout,
                resnet_time_scale_shift,
                resnet_act_fn,
                output_scale_factor,
                resnet_pre_norm,
            )

        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                RCNResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    ray_dim=ray_dim,
                    layout_cfg=layout_cfg if i == 0 else None,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        pos_dim=pos_dim,
                        rgbd_cross_attention=rgbd_cross_attention,
                    )
                )
            else:
                if pos_dim > 0:
                    raise NotImplementedError("pos_dim not supported for dual attn")
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def add_depth_block(
        self,
        in_channels,
        out_channels,
        temb_channels,
        resnet_eps,
        resnet_groups,
        dropout,
        resnet_time_scale_shift,
        resnet_act_fn,
        output_scale_factor,
        resnet_pre_norm,
    ):

        self.depth_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=resnet_eps,
            groups=resnet_groups,
            dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm,
        )
        self.zero_conv = zero_module(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        posemb: Optional[torch.Tensor] = None,
        rays=None,
        layouts=None,
        in_pos3d=None,
        t_out=None,
    ):
        if self.use_depth_block:
            hidden_states = self.forward_depth(hidden_states, t_out, temb)

        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    rays,
                    layouts,
                    **ckpt_kwargs,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),  # transformer_2d
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    posemb,
                    in_pos3d,
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, rays, layouts)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                    posemb=posemb,
                    in_pos3d=in_pos3d,
                )[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

    def forward_depth(
        self,
        hidden_states,
        t_out,
        temb=None,
    ):
        """
        hidden_states: b2t, c, h, w
        """
        assert t_out is not None
        rgb, depth = rearrange(
            hidden_states, "(b k t) c h w -> k (b t) c h w", k=2, t=t_out
        )
        if temb is not None:
            temb_depth = rearrange(temb, "(b k t) c -> k (b t) c", k=2, t=t_out)[1]
        else:
            temb_depth = None

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            enc_depth = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.depth_block),
                torch.cat([rgb, depth], dim=1),
                temb_depth,
                **ckpt_kwargs,
            )
        else:
            enc_depth = self.depth_block(torch.cat([rgb, depth], dim=1), temb_depth)

        depth = depth + self.zero_conv(enc_depth)

        hidden_states = rearrange(
            [rgb, depth], "k (b t) c h w -> (b k t) c h w", t=t_out, k=2
        )
        return hidden_states


class RCNDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        ray_dim=78,
        layout_cfg=None,
        depth_block=False,
    ):
        super().__init__()

        self.use_depth_block = depth_block
        if depth_block:
            self.add_depth_block(
                in_channels * 2,
                in_channels,
                temb_channels,
                resnet_eps,
                resnet_groups,
                dropout,
                resnet_time_scale_shift,
                resnet_act_fn,
                output_scale_factor,
                resnet_pre_norm,
            )

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                RCNResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    ray_dim=ray_dim,
                    layout_cfg=layout_cfg if i == 0 else None,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def add_depth_block(
        self,
        in_channels,
        out_channels,
        temb_channels,
        resnet_eps,
        resnet_groups,
        dropout,
        resnet_time_scale_shift,
        resnet_act_fn,
        output_scale_factor,
        resnet_pre_norm,
    ):

        self.depth_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=resnet_eps,
            groups=resnet_groups,
            dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm,
        )
        self.zero_conv = zero_module(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, hidden_states, temb=None, rays=None, layouts=None, t_out=None):
        if self.use_depth_block:
            hidden_states = self.forward_depth(hidden_states, t_out, temb)

        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        rays,
                        layouts,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        rays,
                        layouts,
                    )
            else:
                hidden_states = resnet(hidden_states, temb, rays, layouts)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

    def forward_depth(
        self,
        hidden_states,
        t_out,
        temb=None,
    ):
        """
        hidden_states: b2t, c, h, w
        """
        assert t_out is not None
        rgb, depth = rearrange(
            hidden_states, "(b k t) c h w -> k (b t) c h w", k=2, t=t_out
        )
        if temb is not None:
            temb_depth = rearrange(temb, "(b k t) c -> k (b t) c", k=2, t=t_out)[1]
        else:
            temb_depth = None

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            enc_depth = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.depth_block),
                torch.cat([rgb, depth], dim=1),
                temb_depth,
                **ckpt_kwargs,
            )
        else:
            enc_depth = self.depth_block(torch.cat([rgb, depth], dim=1), temb_depth)

        depth = depth + self.zero_conv(enc_depth)

        hidden_states = rearrange(
            [rgb, depth], "k (b t) c h w -> (b k t) c h w", t=t_out, k=2
        )
        return hidden_states


class RCNUNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        ray_dim=78,
        layout_cfg=None,
        pos_dim=-1,
        rgbd_cross_attention=False,
        depth_block=False,
    ):
        super().__init__()

        self.use_depth_block = depth_block
        if depth_block:
            self.add_depth_block(
                in_channels * 2,
                in_channels,
                temb_channels,
                resnet_eps,
                resnet_groups,
                dropout,
                resnet_time_scale_shift,
                resnet_act_fn,
                output_scale_factor,
                resnet_pre_norm,
            )

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        # there is always at least one resnet
        resnets = [
            RCNResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                ray_dim=ray_dim,
                layout_cfg=layout_cfg,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        pos_dim=pos_dim,
                        rgbd_cross_attention=rgbd_cross_attention,
                    )
                )
            else:
                if pos_dim > 0:
                    raise NotImplementedError("pos_dim not supported for dual attn")
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                RCNResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    ray_dim=ray_dim,
                    layout_cfg=layout_cfg,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def add_depth_block(
        self,
        in_channels,
        out_channels,
        temb_channels,
        resnet_eps,
        resnet_groups,
        dropout,
        resnet_time_scale_shift,
        resnet_act_fn,
        output_scale_factor,
        resnet_pre_norm,
    ):

        self.depth_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=resnet_eps,
            groups=resnet_groups,
            dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm,
        )
        self.zero_conv = zero_module(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        posemb: Optional[torch.Tensor] = None,
        rays=None,
        layouts=None,
        in_pos3d=None,
        t_out=None,
    ) -> torch.FloatTensor:
        if self.use_depth_block:
            hidden_states = self.forward_depth(hidden_states, t_out, temb)

        hidden_states = self.resnets[0](hidden_states, temb, rays, layouts)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
                posemb=posemb,
                in_pos3d=in_pos3d,
            )[0]
            hidden_states = resnet(hidden_states, temb, rays, layouts)

        return hidden_states

    def forward_depth(
        self,
        hidden_states,
        t_out,
        temb=None,
    ):
        """
        hidden_states: b2t, c, h, w
        """
        assert t_out is not None
        rgb, depth = rearrange(
            hidden_states, "(b k t) c h w -> k (b t) c h w", k=2, t=t_out
        )
        if temb is not None:
            temb_depth = rearrange(temb, "(b k t) c -> k (b t) c", k=2, t=t_out)[1]
        else:
            temb_depth = None

        enc_depth = self.depth_block(torch.cat([rgb, depth], dim=1), temb_depth)
        depth = depth + self.zero_conv(enc_depth)

        hidden_states = rearrange(
            [rgb, depth], "k (b t) c h w -> (b k t) c h w", t=t_out, k=2
        )
        return hidden_states


class RCNUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        ray_dim=78,
        layout_cfg=None,
        depth_block=False,
    ):
        super().__init__()

        self.use_depth_block = depth_block
        if depth_block:
            self.add_depth_block(
                prev_output_channel * 2,
                prev_output_channel,
                temb_channels,
                resnet_eps,
                resnet_groups,
                dropout,
                resnet_time_scale_shift,
                resnet_act_fn,
                output_scale_factor,
                resnet_pre_norm,
            )

        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                RCNResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    ray_dim=ray_dim,
                    layout_cfg=layout_cfg if i == 0 else None,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def add_depth_block(
        self,
        in_channels,
        out_channels,
        temb_channels,
        resnet_eps,
        resnet_groups,
        dropout,
        resnet_time_scale_shift,
        resnet_act_fn,
        output_scale_factor,
        resnet_pre_norm,
    ):

        self.depth_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=resnet_eps,
            groups=resnet_groups,
            dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm,
        )
        self.zero_conv = zero_module(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        upsample_size=None,
        rays=None,
        layouts=None,
        t_out=None,
    ):
        if self.use_depth_block:
            hidden_states = self.forward_depth(hidden_states, t_out, temb)

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        rays,
                        layouts,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        rays,
                        layouts,
                    )
            else:
                hidden_states = resnet(hidden_states, temb, rays, layouts)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

    def forward_depth(
        self,
        hidden_states,
        t_out,
        temb=None,
    ):
        """
        hidden_states: b2t, c, h, w
        """
        assert t_out is not None
        rgb, depth = rearrange(
            hidden_states, "(b k t) c h w -> k (b t) c h w", k=2, t=t_out
        )
        if temb is not None:
            temb_depth = rearrange(temb, "(b k t) c -> k (b t) c", k=2, t=t_out)[1]
        else:
            temb_depth = None

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            enc_depth = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.depth_block),
                torch.cat([rgb, depth], dim=1),
                temb_depth,
                **ckpt_kwargs,
            )
        else:
            enc_depth = self.depth_block(torch.cat([rgb, depth], dim=1), temb_depth)

        depth = depth + self.zero_conv(enc_depth)

        hidden_states = rearrange(
            [rgb, depth], "k (b t) c h w -> (b k t) c h w", t=t_out, k=2
        )
        return hidden_states


class RCNCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        ray_dim=78,
        layout_cfg=None,
        pos_dim=-1,
        rgbd_cross_attention=False,
        depth_block=False,
    ):
        super().__init__()

        self.use_depth_block = depth_block
        if depth_block:
            self.add_depth_block(
                prev_output_channel * 2,
                prev_output_channel,
                temb_channels,
                resnet_eps,
                resnet_groups,
                dropout,
                resnet_time_scale_shift,
                resnet_act_fn,
                output_scale_factor,
                resnet_pre_norm,
            )

        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                RCNResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    ray_dim=ray_dim,
                    layout_cfg=layout_cfg if i == 0 else None,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        pos_dim=pos_dim,
                        rgbd_cross_attention=rgbd_cross_attention,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def add_depth_block(
        self,
        in_channels,
        out_channels,
        temb_channels,
        resnet_eps,
        resnet_groups,
        dropout,
        resnet_time_scale_shift,
        resnet_act_fn,
        output_scale_factor,
        resnet_pre_norm,
    ):

        self.depth_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=resnet_eps,
            groups=resnet_groups,
            dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm,
        )
        self.zero_conv = zero_module(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        posemb: Optional = None,
        rays=None,
        layouts=None,
        in_pos3d=None,
        t_out=None,
    ):
        if self.use_depth_block:
            hidden_states = self.forward_depth(hidden_states, t_out, temb)

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    rays,
                    layouts,
                    **ckpt_kwargs,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    posemb,
                    in_pos3d,
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, rays, layouts)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                    posemb=posemb,
                    in_pos3d=in_pos3d,
                )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

    def forward_depth(
        self,
        hidden_states,
        t_out,
        temb=None,
    ):
        """
        hidden_states: b2t, c, h, w
        """
        assert t_out is not None
        rgb, depth = rearrange(
            hidden_states, "(b k t) c h w -> k (b t) c h w", k=2, t=t_out
        )
        if temb is not None:
            temb_depth = rearrange(temb, "(b k t) c -> k (b t) c", k=2, t=t_out)[1]
        else:
            temb_depth = None

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            enc_depth = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.depth_block),
                torch.cat([rgb, depth], dim=1),
                temb_depth,
                **ckpt_kwargs,
            )
        else:
            enc_depth = self.depth_block(torch.cat([rgb, depth], dim=1), temb_depth)

        depth = depth + self.zero_conv(enc_depth)

        hidden_states = rearrange(
            [rgb, depth], "k (b t) c h w -> (b k t) c h w", t=t_out, k=2
        )
        return hidden_states
