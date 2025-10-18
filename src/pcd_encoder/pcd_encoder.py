import torch
from typing import List
from torch import Tensor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from .model import PointTransformerV3


class PointEncoder(PointTransformerV3, ModelMixin, ConfigMixin):
    def __init__(self, ckpt_path="", out_channels=512, **kwargs):
        super().__init__(**kwargs)

        last_channel = kwargs["enc_channels"][-1]
        if out_channels != last_channel:
            self.proj = torch.nn.Linear(last_channel, out_channels)
        else:
            self.proj = torch.nn.Indentity()

        # load pretrained weight
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            state_dict = {
                k.replace("module.", "").replace("backbone.", ""): v
                for k, v in state_dict["state_dict"].items()
                if not "dec" in k
            }
            ckpt_keys = set(list(state_dict))
            model_dict = self.state_dict()
            model_keys = set(list(model_dict))
            missing_keys = model_keys - ckpt_keys
            unexpected_keys = ckpt_keys - model_keys
            print(f"load checkpoint from {ckpt_path}")
            print(f"missing keys: {missing_keys}")
            print(f"unexpected keys: {unexpected_keys}")
            for k in model_keys.intersection(ckpt_keys):
                if not model_dict[k].size() == state_dict[k].size():
                    print(f"size mismatch: {k} model: {model_dict[k].size()} ckpt: {state_dict[k].size()}")
                    state_dict.pop(k)
            self.load_state_dict(state_dict, strict=False)            

    def forward(self, data_dict, do_cfg=False, **kwargs):
        """
        run model forward
        return feat and 3d coords in world coordinate
        """
        # min_coord = data_dict.pop("min_coord")
        # grid_size = data_dict.pop("grid_size")

        point_output = super().forward(data_dict)

        feat = point_output["feat"]
        feat = self.proj(feat)
        coord = point_output["coord"]

        # split feat and coord then padding
        offset = point_output["offset"]
        length = offset.clone()
        length[1:] = offset[1:] - offset[:-1]

        feat = torch.split(feat, length.tolist(), dim=0)
        coord = torch.split(coord, length.tolist(), dim=0)
        feat = pad_wgrad(feat)
        coord = pad_wgrad(coord)
        if do_cfg:
            coord = torch.cat([torch.zeros_like(coord), coord])
            feat = torch.cat([torch.zeros_like(feat), feat])
        return {"feat": feat, "coord": coord}


def pad_wgrad(tensors: List[Tensor]) -> Tensor:
    """
    padding tensor to the same length then stack to batch
    padding value is 0.0

    args:
        tensors: B x [T, ...] torch tensors
    return tensor B,maxT,...

    """
    lens = [x.size(0) for x in tensors]
    max_len = max(lens)
    batch_size = len(tensors)
    dims = list(tensors[0].size()[1:])

    device = tensors[0].device
    dtype = tensors[0].dtype

    output = []
    for i in range(batch_size):
        if lens[i] < max_len:
            tmp = torch.cat(
                [
                    tensors[i],
                    torch.zeros([max_len - lens[i], *dims], dtype=dtype, device=device),
                ],
                dim=0,
            )
        else:
            tmp = tensors[i]
        output.append(tmp)
    output = torch.stack(output, 0)
    return output
