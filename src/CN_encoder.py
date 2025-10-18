from transformers import ConvNextV2Model
import torch
from typing import Optional
import einops
from diffusers.models.unet_2d_rcn_blocks import RayCondNorm, RayEmbedding
from torchvision import transforms

class CN_encoder(ConvNextV2Model):
    def __init__(self, config, im_size=224, use_ray=False, d_bands=6, m_bands=6):
        super().__init__(config)
        self.use_ray = use_ray
        if use_ray:
            feature_dim = self.encoder.stages[-1].layers[-1].dwconv.in_channels
            ray_dim = 6 + 6 * d_bands + 6 * m_bands
            self.rcn = RayCondNorm(input_dim=feature_dim, ray_dim=ray_dim)
            self.ray_embed = RayEmbedding(d_bands=d_bands, m_bands=m_bands)
            
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (im_size, im_size), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        rays: Optional[torch.FloatTensor] = None,
        depth_mask: Optional[torch.FloatTensor] = None,
    ):
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        pixel_values = self.transform(pixel_values)
        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        if self.use_ray:
            ray_embed = self.ray_embed(rays)
            last_hidden_state = self.rcn(last_hidden_state, ray_embed)
        
        h, w = last_hidden_state.shape[-2:]
        image_embeddings = einops.rearrange(last_hidden_state, "b c h w -> b (h w) c")
        image_embeddings = self.layernorm(image_embeddings)

        if depth_mask is not None:
            # print("masking")
            # print('last_hidden_state', last_hidden_state.shape)
            # print('depth_mask', depth_mask.shape)
            depth_mask = einops.rearrange(depth_mask, "b t h w -> (b t) 1 h w")
            depth_mask = torch.nn.functional.interpolate(
                depth_mask, size=(h, w), mode="nearest"
            )
            depth_mask = einops.rearrange(depth_mask, "b 1 h w -> b (h w) 1")
            image_embeddings = image_embeddings * depth_mask
            # last_hidden_state = last_hidden_state * depth_mask

        return image_embeddings
