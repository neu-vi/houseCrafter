import math
import zlib
from typing import List, Tuple, Union

import cv2
import lmdb
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from einops import rearrange, repeat
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NormWeightedCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
)
from pytorch3d.structures import Pointclouds
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import random
import einops

FRONT3D_BASE_MATRIX = torch.tensor(
    [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
    dtype=torch.float32,
)


def crop_img(img: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray]:
    K = K.copy()
    H, W, _ = img.shape
    if H > W:
        margin_l = (H - W) // 2
        margin_r = H - margin_l - W
        img = img[margin_l:-margin_r, :]
        K[1, 2] -= margin_l
    elif H < W:
        margin_l = (W - H) // 2
        margin_r = W - margin_l - H
        img = img[:, margin_l:-margin_r]
        K[0, 2] -= margin_l

    return img, K


def resize_img(
    img: np.ndarray, size: int, K: np.ndarray = None, interpolation=cv2.INTER_LINEAR
) -> Tuple[np.ndarray]:
    H, W, *_ = img.shape
    assert H == W

    img = cv2.resize(img, (size, size), interpolation=interpolation)
    if K is None:
        return img

    K = K.copy()
    K *= size / H
    K[2, 2] = 1.0
    return img, K


def get_ray_direction(R: Tensor, K, h: int, w: int, stride: int = 1) -> Tensor:
    """
    get ray direction in world coordinate
    R rotation matrix camera2world
    K intrinsic matrix camera2uv
    h,w image height and width

    return
        ray direction of shape h//stride,w//stride,3
        camera center

    camera coordinate: Y is down, Z is camera direction, X is right
    #camera coordinate: Z is up, Y is camera direction, X is right
    """
    assert h % stride == 0 and w % stride == 0
    K = K / stride
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    xs = torch.arange(w // stride, device=R.device) + 0.5  # shape W,
    xs = (xs - cx) / fx

    ys = torch.arange(h // stride, device=R.device) + 0.5
    ys = (ys - cy) / fy

    ys, xs = torch.meshgrid(ys, xs, indexing="ij")  # shape H,W
    zs = torch.ones_like(ys)
    ray_direction = torch.stack([xs, ys, zs], dim=-1)  # in camera coordinate
    ray_direction = ray_direction.to(R.dtype)
    ray_direction = torch.matmul(
        R.view(1, 1, 3, 3), ray_direction.unsqueeze(-1)
    ).squeeze(
        -1
    )  # in world coordinate
    return ray_direction


def get_plucker_coordinate(
    ray_direction: Tensor, ray_origin: Tensor, normalize_d=True
) -> Tensor:
    """
    ray direction: h,w,3
    ray_origin: 3 or h,w,3

    return coordinate h,w,6
    """
    if normalize_d:
        ray_direction = ray_direction / torch.linalg.norm(
            ray_direction, dim=-1, keepdim=True
        )
    if len(ray_origin.size()) == 1:
        h, w, _ = ray_direction.size()
        ray_origin = repeat(ray_origin, "c -> h w c", h=h, w=w)
    m = torch.linalg.cross(ray_origin, ray_direction)  # moment
    return torch.cat([m, ray_direction], dim=-1)


def get_world_pcd(P: Tensor, K: Tensor, depth: Tensor, stride: int = 1) -> Tensor:
    """
    get point cloud in world coordinate from depth, intrinsic matrix and camera pose
    P extrinsic matrix camera2world
    K intrinsic matrix camera2uv
    depth n,h,w

    return
        point cloud of shape n, h//stride, w//stride, 3

    camera coordinate: Y is down, Z is camera direction, X is right
    #camera coordinate: Z is up, Y is camera direction, X is right
    """
    K = K.to(P.dtype)
    depth = depth.to(P.dtype)
    _, h, w = depth.shape
    assert h % stride == 0 and w % stride == 0
    K = K / stride
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    xs = torch.arange(w // stride, device=P.device) + 0.5  # shape W,
    xs = (xs - cx) / fx

    ys = torch.arange(h // stride, device=P.device) + 0.5
    ys = (ys - cy) / fy

    ys, xs = torch.meshgrid(ys, xs, indexing="ij")  # shape H,W
    zs = torch.ones_like(ys)
    depth = depth[..., stride // 2 :: stride, stride // 2 :: stride]
    cam_pcd = torch.stack([xs, ys, zs], dim=-1) * depth.unsqueeze(-1)  # shape n,H,W,3
    world_pcd = P[:3, :3].view(1, 1, 1, 3, 3) @ cam_pcd.unsqueeze(-1) + P[:3, 3].view(
        1, 1, 1, 3, 1
    )
    return world_pcd.squeeze(-1)


def collate_fn(batch):
    out = {}
    meta = {}

    keys = list(batch[0].keys())
    for k in keys:
        if k == "meta":
            # Merge meta fields from all samples
            meta_fields = batch[0]["meta"].keys()
            for m in meta_fields:
                # Assume meta fields are either scalar, string, or indexable
                meta[m] = [sample["meta"][m] for sample in batch]
            continue
        data = [x[k] for x in batch]
        if k in ["layout_cls", "layout_pos", "back_layout_cls", "back_layout_pos", "layout_obj_cls", "layout_obj_pos"]:
            out[k] = padding(data)
        elif isinstance(data[0], Tensor):
            out[k] = torch.stack(data, dim=0)
        elif isinstance(data[0], (int, float)):
            out[k] = torch.tensor(data)
        elif k in ["frame_ids", "scene_id", "start_node", "object_id", "item_id", "layout_obj_camheight", "layout_obj_camelevation"]:
            out[k] = data
        else:
            raise ValueError(f"Unrecognized data type for {k}")
    out["meta"] = meta

    return out


def padding(data: List[Tensor], pad_values=0):
    """
    data: list each element is of shape t,n,...
    assume padding dim is 1
    """
    data = [rearrange(x, "t n ... -> n t ...") for x in data]
    data = pad_sequence(data, batch_first=True, padding_value=pad_values)
    data = rearrange(data, "b n t ... -> b t n ...")
    return data


def get_connected_subgraphs(G: nx.Graph, n_nodes: int) -> List[List]:
    """
    get list of all connected subgraphs with given size of a graph
    args:
        G: nx.Graph
        n_nodes: size of subgraph
    return list of connected subgraphs, each subgraph is a list of nodes

    NOTE: for a graph of 1700 nodes, it takes 20s to get all connected subgraphs of size 6 (1m subgraphs)
    """

    results = []
    excluded = set()
    nodes = list(G.nodes)
    for node in nodes:
        excluded.add(node)
        recursive_local_expand(
            G, {node}, set(G.neighbors(node)) - excluded, excluded, results, n_nodes
        )
    return results


def recursive_local_expand(
    G,
    node_set,
    possible,
    excluded,
    results,
    max_size,
    deterministic=True,
    return_one=False,
    curation_list=None,
):
    """
    expand the subgraph until reach the max size
    using excluded to avoid permutation of the same subgraph
    """
    if return_one and len(results) > 0:
        return
    if len(node_set) == max_size:
        results.append(node_set)
        return
    # candidates are the neighbors of the current nodeset
    candidates = sorted(list(possible - excluded))
    # exclude the candidate if the curation_list marks it as True(is_wall_only)
    if not deterministic:
        permutation = np.random.permutation(len(candidates))
        candidates = [candidates[i] for i in permutation]
    
    # if curation_list is not None:
    #     reserved_candidate = sorted(candidates)[0]
    #     candidates = [c for c in candidates if not curation_list[c]]
    #     if len(candidates) == 0:
    #         candidates = [reserved_candidate]
    #         print("No valid candidates, using reserved candidate")

    # Randomly select one candidate (or pick the first one in deterministic mode)
    if deterministic:
        chosen_node = sorted(candidates)[0]
    else:
        chosen_node = random.choice(candidates)

    # Expand with the selected node
    new_node_set = node_set | {chosen_node}
    new_possible = (possible | set(G.neighbors(chosen_node))) - excluded
    recursive_local_expand(
        G, new_node_set, new_possible, excluded | {chosen_node}, results, max_size, deterministic, return_one
    )


class DepthLinearTransform:
    """
    linearly transform range [min_depth, max_depth] to [-1,1] and vice versa
    """

    def __init__(self, min_depth: float, max_depth: float, clip=False, masked_depth=None):
        """
        min/max depth: in meters
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.range = max_depth - min_depth
        self.clip = clip
        # self.masked_depth = masked_depth

    def __call__(self, depth: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        """
        normalize depth to [-1,1]
        """
        out = 2 * (depth - self.min_depth) / self.range - 1
        if self.clip:
            if isinstance(out, Tensor):
                out = torch.clamp(out, -1, 1)
            else:
                out = np.clip(out, -1, 1)
        return out

    def denormalize(
        self, depth: Union[Tensor, np.ndarray]
    ) -> Union[Tensor, np.ndarray]:
        return (depth + 1) * self.range / 2 + self.min_depth


class DepthAffineTransform:
    """
    linearly transform range [min_depth, max_depth] to [-1,1]
    here min and max is per image
    adapted from https://github.com/prs-eth/Marigold/blob/ecf8a9eb46e9c4b211c335efd643cbbf516d1d26/src/util/depth_transform.py#L49
    """

    def __init__(self, clip=False, min_max_quantile=0.02, min_range=0.1):
        self.clip = clip
        self.min_quantile = min_max_quantile
        self.max_quantile = 1 - min_max_quantile
        self.min_range = min_range

    def __call__(
        self,
        depth: Union[Tensor, np.ndarray],
        valid_mask: Union[Tensor, np.ndarray] = None,
    ) -> Union[Tensor, np.ndarray]:
        is_numpy = isinstance(depth, np.ndarray)
        if is_numpy:
            depth = torch.tensor(depth, dtype=torch.float32)
            if valid_mask is not None:
                valid_mask = torch.tensor(valid_mask, dtype=torch.bool)

        if valid_mask is None:
            valid_mask = torch.ones_like(depth).bool()
        valid_mask = valid_mask & (depth > 0)

        if not valid_mask.any():
            valid_mask = torch.ones_like(depth).bool()

        _min, _max = torch.quantile(
            depth[valid_mask],
            torch.tensor([self.min_quantile, self.max_quantile], device=depth.device),
        )
        _max = max(_max, _min + self.min_range)
        depth = (depth - _min) / (_max - _min) * 2 - 1
        if self.clip:
            depth = torch.clamp(depth, -1, 1)
        if is_numpy:
            depth = depth.numpy()
        return depth


def render_target_view(
    input_image,
    input_depth,
    pose_in,
    pose_out,
    FOV,
    is_3dfront=False,
    kernel_size=1,
):
    """
    args:
        FOV: degree
        input_images: n,h,w,3 uint8
        input_depth: n,h,w float32
        pose_in: n,4,4 w2c
        pose_out: m,4,4 w2c
        is_3dfront: if True apply FRONT3D_BASE_MATRIX to change camera coordinate

    return
        output_image: m,h,w,3 uint8
        output_depth: m,h,w float32

    """
    # from cam coord in this code to blender cam coord
    # this code x right, y down, z forward
    # blender x right, y up, z backward
    FRONT3D_BASE_MATRIX = torch.tensor(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ],
        dtype=torch.float32,
        device=input_image.device,
    )
    N, H, W, c = input_image.shape
    assert H == W
    focal = H / 2 / np.tan(np.radians(FOV / 2))
    K = torch.tensor(
        [
            [focal, 0.0, H / 2],
            [0.0, focal, W / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float,
        device=input_image.device,
    )
    # convert to c2w
    pose_in = torch.linalg.inv(pose_in)
    pose_out = torch.linalg.inv(pose_out)
    if is_3dfront:
        pose_in[:, :3, :3] = pose_in[:, :3, :3] @ FRONT3D_BASE_MATRIX
        pose_out[:, :3, :3] = pose_out[:, :3, :3] @ FRONT3D_BASE_MATRIX
    pcd, rgb = make_pcd_batch(input_image, input_depth, pose_in, K)

    output_imgs, output_depths = [], []
    for P in pose_out:
        img, depth = render_pcd(pcd, rgb, P, H, W, K, kernel_size=kernel_size)
        output_imgs.append(img)
        output_depths.append(depth)
    return torch.stack(output_imgs), torch.stack(output_depths)


def make_pcd_batch(rgb, depth, P, K, min_depth=0.1, dilate_invalid=0, max_depth=29.9):
    """
    args:
        rgb: n,H,W,3
        depth: n,H,W
        P: n,4,4
        K: 3,3
    return pcd, color N,3
    NOTE: K and image size must match rgb and depth
    assume all images share the same K
    """
    image_size = depth.shape[-1]
    I = torch.eye(3, device=P.device)
    grid = get_ray_direction(I, K, image_size, image_size)
    pcd = grid * depth[..., None]
    invalid_mask = (depth <= min_depth) | (depth >= max_depth)
    if dilate_invalid:
        invalid_mask = (
            F.max_pool2d(
                invalid_mask[:, None, ...].float(),
                kernel_size=dilate_invalid * 2 + 1,
                stride=1,
                padding=dilate_invalid,
            ).squeeze(1)
            > 0.5
        )
    invalid_mask = invalid_mask.reshape(-1)
    pcd = rearrange(pcd, "n h w c -> n (h w) c")
    pcd = pcd @ rearrange(P[:, :3, :3], "n a b-> n b a") + P[:, None, :3, 3]
    rgb = rearrange(rgb, "n h w c -> (n h w) c")
    pcd = rearrange(pcd, "n hw c -> (n hw) c")
    return pcd[~invalid_mask], rgb[~invalid_mask]


class RGBDEncoding:
    """
    this class encodes rgbd info of each pixel into a single int64 number
    which can be decoded back to rgbd info.
    This is for efficient use of sparse tensor coalescing.
    the tradeoff is that the depth need to be quantized. and the maximum depth is 2**40 mm
    """

    DEPTH_UNIT = 0.001  # quantize depth to 1mm

    @classmethod
    def encode(cls, rgb, depth):
        """
        rgb : torch.uint8 tensor of shape (*,3)
        depth: torch.float tensor of shape (*)
        """
        depth = depth / cls.DEPTH_UNIT
        depth = depth.round().to(torch.int64)
        rgb = rgb.to(torch.int64)
        return (depth << 24) | (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]

    @classmethod
    def decode(cls, encoded):
        encoded = encoded.to(torch.int64)
        depth = (encoded >> 24).to(torch.float32) * cls.DEPTH_UNIT
        rgb = torch.stack(
            [
                (encoded >> 16) & 0xFF,
                (encoded >> 8) & 0xFF,
                encoded & 0xFF,
            ],
            dim=-1,
        ).to(torch.uint8)
        return rgb, depth


def render_pcd(pcd, color, P, H, W, K, kernel_size=1):
    """
    args:
        pcd: N,3 float32
        color: N,3, uint8 or flaot 32 in range [0,255]
        P: 4,4 c2w, cam coord is Y down, Z forward, X right
        K: 3,3
    return: rgb H,W,3 uint8, depth H,W float32
    NOTE: H W must match K

    """
    R = P[:3, :3]  # @ torch.tensor(FRONT3D_BASE_MATRIX, device=P.device)

    # transform pcd to camera coordinate
    pcd = (pcd - P[:3, 3]) @ R
    depth = pcd[:, 2]
    x = (pcd[:, 0] / depth * K[0, 0] + K[0, 2]).to(torch.int64)
    y = (pcd[:, 1] / depth * K[1, 1] + K[1, 2]).to(torch.int64)
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (depth > 0)
    x = x[mask]
    y = y[mask]
    depth = depth[mask]
    color = color[mask]
    out_of_bound_mask = None
    if kernel_size > 1:
        depth = repeat(depth, "n -> (k n)", k=kernel_size**2)
        color = repeat(color, "n c -> (k n) c", k=kernel_size**2)
        ys, xs = [], []
        for i in range(-(kernel_size // 2), kernel_size // 2 + 1):
            for j in range(-(kernel_size // 2), kernel_size // 2 + 1):
                ys.append(y + i)
                xs.append(x + j)
        y = torch.cat(ys, dim=0)
        x = torch.cat(xs, dim=0)
        out_of_bound_mask = (x < 0) | (x >= W) | (y < 0) | (y >= H)
    encoded = RGBDEncoding.encode(color, depth)  # N

    flattened_index = y * W + x
    if out_of_bound_mask is not None:
        flattened_index[out_of_bound_mask] = H * W
    flattened_img = torch_scatter.scatter(
        encoded, flattened_index, reduce="min", dim_size=H * W + 1
    )
    rgb, depth = RGBDEncoding.decode(flattened_img[:-1])

    return rgb.view(H, W, 3), depth.view(H, W)


def get_absolute_depth(
    imgs,
    relative_depths,
    frames_meta,
    layout_db_path,
    depth_model,
    bg_thr=5000,
    valid_r_depth_thr=0.01,
    img_size_layout_db=512,
):
    """
    get absolute depth from relative depth
    either using layout or depth model

    args:
        imgs: torch float n c h w in [0,255]
        relative_depths: torch float32 (n h w) in [0,1]
        frame_meta: [frame_id]
        depth_model: func(img)->depth
    return absolute_depths: torch.float32 n h w in meters
    """
    layout_db = lmdb.open(
        layout_db_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    out = []
    methods = []
    H_img = imgs[0].shape[-2]
    assert img_size_layout_db % H_img == 0
    stride = img_size_layout_db // H_img

    for img, r_depth, frame_id in zip(imgs, relative_depths, frames_meta):

        # stride 2 since the layout shape is 512x512 and the img shape is 256x256
        bg_depth, bg_mask = load_background_depth(
            layout_db,
            frame_id,
            device=img.device,
            stride=stride,
            H=img_size_layout_db,
            W=img_size_layout_db,
        )
        bg_size = bg_mask.sum()
        valid_mask_size = (bg_depth > 0).sum()
        valid_r_depth = r_depth > valid_r_depth_thr
        # use bg depth directly
        method = "layout"
        if bg_size == valid_mask_size:
            a_depth = bg_depth

        # use bg depth to find offset and scale for relative depth
        elif bg_mask.sum() > bg_thr and (bg_mask & valid_r_depth).sum() > 0.0:
            mask = bg_mask & valid_r_depth
            a_depth = scale_shift_linear(bg_depth, r_depth, mask)
            a_depth[~valid_r_depth] = 0.0
        else:
            # use depth model
            est_depth = depth_model(img)
            try:
                a_depth = scale_shift_linear(est_depth, r_depth, valid_r_depth)
                a_depth[~valid_r_depth] = 0.0
            except torch._C_.LinAlgError:
                a_depth = torch.zeros_like(r_depth)
            method = "depth_model"
        out.append(a_depth)
        methods.append(method)
    return torch.stack(out), methods

def get_estimated_depth(
    imgs,
    frames_meta,
    layout_db_path,
    depth_model,
    bg_thr=5000,
    valid_r_depth_thr=0.01,
    img_size_layout_db=512,
):
    """
    get absolute depth from relative depth
    either using layout or depth model

    args:
        imgs: torch float n c h w in [0,255]
        relative_depths: torch float32 (n h w) in [0,1]
        frame_meta: [frame_id]
        depth_model: func(img)->depth
    return absolute_depths: torch.float32 n h w in meters
    """
    layout_db = lmdb.open(
        layout_db_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    out = []
    methods = []
    H_img = imgs[0].shape[-2]
    assert img_size_layout_db % H_img == 0
    stride = img_size_layout_db // H_img

    for img, frame_id in zip(imgs, frames_meta):

        # stride 2 since the layout shape is 512x512 and the img shape is 256x256
        bg_depth, bg_mask = load_background_depth(
            layout_db,
            frame_id,
            device=img.device,
            stride=stride,
            H=img_size_layout_db,
            W=img_size_layout_db,
        )
        bg_size = bg_mask.sum()
        r_depth = depth_model(img)
        valid_mask_size = (bg_depth > 0).sum()
        valid_r_depth = r_depth > valid_r_depth_thr
        # use bg depth directly
        if bg_size == valid_mask_size:
            a_depth = bg_depth
            method = "layout"
        # use bg depth to find offset and scale for relative depth
        elif bg_mask.sum() > bg_thr and (bg_mask & valid_r_depth).sum() > 0.0:
            mask = bg_mask & valid_r_depth
            try:
                a_depth = scale_shift_linear(bg_depth, r_depth, mask)
                a_depth[~valid_r_depth] = 0.0
            except torch._C_.LinAlgError:
                a_depth = r_depth
            method = "depth_model"
        else:
            a_depth = r_depth

            method = "depth_model"
        out.append(a_depth)
        methods.append(method)
    return torch.stack(out), methods

def load_background_depth(db, frame_id, stride, device, H=512, W=512):
    """
    return
        background depth torch.float32 HxW
        background mask torch.bool HxW (True for background)
    """
    txn = db.begin(write=False)
    key = frame_id.encode("ascii")
    data = txn.get(key)
    data = zlib.decompress(data)
    data = np.frombuffer(data, dtype=np.uint16).reshape(-1, 2, H, W)

    layout_cls = torch.tensor(
        data[
            :,
            0,
            ...,
            stride // 2 :: stride,
            stride // 2 :: stride,
        ].astype(np.int32),
        dtype=torch.long,
        device=device,
    )
    depth = (
        torch.tensor(
            data[
                :,
                1,
                ...,
                stride // 2 :: stride,
                stride // 2 :: stride,
            ].astype(np.float32),
            device=device,
        )
        / 1000.0
    )
    bg_mask = layout_cls[0] > 0
    fg_mask = torch.any(layout_cls[1:] > 0, dim=0)
    bg_only = bg_mask & ~fg_mask
    return depth[0], bg_only


def scale_shift_linear(reference_depth, target_depth, mask):
    """
    Optimize a scale and shift parameter in the least squares sense,
    such that ref_depth and target_depth match.
    Formally, solves the following objective:

    min     f = || (d * a + b) - d_hat ||
    a, b

    where d is target_depth, d_hat is ref depth

    args:
        reference_depth: torch.Tensor (H, W)
        target_depth:  torch.Tensor (H, W)
        mask: torch.Tensor (H, W)

    return: d * a + b
    """
    assert mask.sum() > 0

    reference_depth_ = reference_depth[mask].unsqueeze(-1)
    target_depth_ = target_depth[mask].unsqueeze(-1)

    X = torch.cat([target_depth_, torch.ones_like(target_depth_)], dim=1)
    XTX_inv = (X.T @ X).inverse()
    XTY = X.T @ reference_depth_
    AB = XTX_inv @ XTY

    fixed_depth = target_depth * AB[0] + AB[1]
    return fixed_depth


def render_target_view_torch3d(
    input_image,
    input_depth,
    pose_in,
    pose_out,
    FOV,
    is_3dfront=False,
    batch_size=1,
    torch_renderer=None,
    radius=3.0,
    check_empty_pcd=False,
):
    """
    args:
        FOV: degree
        input_images: n,h,w,3 float32
        input_depth: n,h,w float32
        pose_in: n,4,4 w2c
        pose_out: m,4,4 w2c
        is_3dfront: if True apply FRONT3D_BASE_MATRIX to change camera coordinate

    return
        output_image: m,h,w,3 float32
        output_depth: m,h,w float32

    """
    # from cam coord in this code to blender cam coord
    # this code x right, y down, z forward
    # blender x right, y up, z backward
    FRONT3D_BASE_MATRIX = torch.tensor(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ],
        dtype=torch.float32,
        device=input_image.device,
    )
    N, H, W, c = input_image.shape
    assert H == W
    focal = H / 2 / np.tan(np.radians(FOV / 2))
    K = torch.tensor(
        [
            [focal, 0.0, H / 2],
            [0.0, focal, W / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float,
        device=input_image.device,
    )
    # convert to c2w
    pose_in = torch.linalg.inv(pose_in)
    pose_out = torch.linalg.inv(pose_out)
    if is_3dfront:
        pose_in[:, :3, :3] = pose_in[:, :3, :3] @ FRONT3D_BASE_MATRIX
        pose_out[:, :3, :3] = pose_out[:, :3, :3] @ FRONT3D_BASE_MATRIX
    pcd, rgb = make_pcd_batch(input_image, input_depth, pose_in, K)
    if check_empty_pcd and not pcd.numel():
        raise ValueError("got empty pointcloud")
    if torch_renderer is None:
        torch_renderer = Torch3DRenderer(H, input_image.device, radius=radius, fov=FOV)
    output_imgs, output_depths = [], []
    n_batch = math.ceil(len(pose_out) / batch_size)
    for i in range(n_batch):
        Ps = pose_out[i * batch_size : (i + 1) * batch_size]
        img, depth = torch_renderer(pcd, rgb, Ps)
        output_imgs.append(img)
        output_depths.append(depth)
    return torch.cat(output_imgs), torch.cat(output_depths)


class Torch3DRenderer:
    # base matrix to convert from camera coord in torch3d to camera coord in the codebase
    # cam coord in this codebase is x right y down z forward
    BASE_MATRIX = torch.tensor(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    def __init__(
        self,
        image_size,
        device,
        znear=0.05,
        zfar=30.0,
        fov=90,
        points_per_pixel=1,
        radius=2.0,
        bin_size=None,
    ):
        """radius is in pixel unit"""
        self.device = device
        self.znear = znear
        self.zfar = zfar
        self.fov = fov
        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=radius / image_size,
            points_per_pixel=points_per_pixel,
            bin_size=bin_size,
        )
        cameras = FoVPerspectiveCameras(device=device)
        self.rasterizer = PointsRasterizer(
            cameras=cameras, raster_settings=raster_settings
        )
        # self.compositor = AlphaCompositor()
        self.compositor = NormWeightedCompositor()

    @torch.no_grad()
    def __call__(self, pcd, color, P):
        """
        args:
            pcd: N,3 float32
            color: N,3, uint8
            P: n,4,4 c2w, cam coord is x right y down z forward
                in torch3d, cam coord is x left y up z forward
        """
        bs = len(P)
        pcd_torch = Pointclouds(points=[pcd] * bs, features=[color] * bs)
        cameras = self._make_cameras(P)
        fragments = self.rasterizer(pcd_torch, cameras=cameras)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            pcd_torch.features_packed().permute(1, 0),
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)
        depth = fragments.zbuf[..., 0]  # take depth from the first point
        return images, depth

    def _make_cameras(self, P):
        """
        P: n,4,4 c2w
        """
        P = P.clone()
        P[:, :3, :3] = P[:, :3, :3] @ self.BASE_MATRIX.to(P.device)
        P = torch.linalg.inv(P)
        R = P[:, :3, :3].transpose(1, 2)
        T = P[:, :3, 3]
        return FoVPerspectiveCameras(
            znear=self.znear,
            zfar=self.zfar,
            fov=self.fov,
            R=R,
            T=T,
            device=self.device,
        )


def make_pipeline_input(
    batch,
    device,
    weight_dtype,
    use_ray,
    output_depth,
    generator,
    guidance_scale=3.0,
    depth_transform=None,
    torch_renderer=None,
    ddim_inversion=False,
    null_layout=False,
    gaussian_clip=0.0,
):
    """
    batch from dataloader

    return kwargs for pipeline
    """
    T_in = batch["image_input"].size(1)
    input_image = batch["image_input"].to(dtype=weight_dtype).to(device)
    pose_in = batch["pose_in"].to(dtype=weight_dtype).to(device)  # BxTx4
    pose_out = batch["pose_out"].to(dtype=weight_dtype).to(device)  # BxTx4
    pose_in_inv = batch["pose_in_inv"].to(dtype=weight_dtype).to(device)  # BxTx4
    pose_out_inv = batch["pose_out_inv"].to(dtype=weight_dtype).to(device)  # BxTx4

    # make args for ddim inversion
    if ddim_inversion:
        try:
            # NOTE assume batch size is 1
            assert batch["depth_input"].size(0) == 1
            input_depth = batch["depth_input"].to(device)
            warp_img, warp_depth = render_target_view_torch3d(
                rearrange((input_image + 1) * (255 / 2), "b t c h w -> (b t) h w c").to(
                    torch.float32
                ),
                rearrange(input_depth, "b t h w -> (b t) h w").to(torch.float32),
                rearrange(pose_in.to(torch.float32), "b t c d -> (b t) c d"),
                rearrange(pose_out.to(torch.float32), "b t c d -> (b t) c d"),
                FOV=90.0,
                is_3dfront=True,
                torch_renderer=torch_renderer,
                batch_size=1,
                check_empty_pcd=True,
            )
            warp_img = rearrange(warp_img, "t h w c -> 1 t c h w")
            warp_img = (warp_img.to(dtype=weight_dtype) / 255.0 - 0.5) * 2
            warp_depth = [depth_transform(depth) for depth in warp_depth]
            warp_depth = rearrange(warp_depth, "t h w -> 1 t h w").to(
                dtype=weight_dtype
            )
        except ValueError as e:
            if "empty pointcloud" in e.args[0]:
                warp_img = None
                warp_depth = None
                ddim_inversion = False
            else:
                raise e
    else:
        warp_img = None
        warp_depth = None

    input_image = rearrange(input_image, "b t c h w -> (b t) c h w")  # T_in

    kwargs = {}
    # make args for ray condition
    if use_ray:
        target_rays = [
            v.to(dtype=weight_dtype, device=device)
            for k, v in batch.items()
            if "target_ray" in k
        ]
        kwargs["target_rays"] = target_rays

        cond_rays = [
            v.to(dtype=weight_dtype, device=device)
            for k, v in batch.items()
            if "cond_ray" in k
        ]
        cond_rays = [rearrange(r, "b t c h w -> (b t) c h w") for r in cond_rays]

        if len(cond_rays):
            # NOTE: only support 1 cond_ray
            kwargs["cond_rays"] = cond_rays[0]

    # make args for layout
    if "layout_cls" in batch:
        layout_dict = {}
        for k, v in batch.items():
            if "layout" in k:
                if v.dtype != torch.long:
                    v = v.to(dtype=weight_dtype, device=device)
                else:
                    v = v.to(device=device)
                layout_dict[k] = v
        if null_layout:
            for k in list(layout_dict):
                layout_dict[k] = torch.zeros_like(layout_dict[k])
        kwargs["layouts"] = layout_dict

    # make args for 3d position from input depth
    in_pos3d = batch.get("in_pos3d", None)
    if in_pos3d is not None:
        in_pos3d = in_pos3d.to(dtype=weight_dtype, device=device)
    depth_mask_cond = batch.get("depth_mask_cond", None)
    if depth_mask_cond is not None:
        depth_mask_cond = depth_mask_cond.to(dtype=torch.float32, device=device)
        b,t,h,w,c = in_pos3d.shape
        # interpolate depth_mask_cond to in_pos3d size
        # in_pos3d: torch.Size([8, t, 7, 7, 3]) mask: torch.Size([8, t, 224, 224])
        depth_mask_cond = einops.rearrange(depth_mask_cond, "b t h w -> (b t) 1 h w")
        resized_mask = F.interpolate(depth_mask_cond, size=(h, w), mode='nearest')  # Shape: (8, 1, 7, 7)
        resized_mask = einops.rearrange(resized_mask, "(b t) 1 h w -> b t h w 1", t=t)
        in_pos3d = in_pos3d * resized_mask  



    h, w = input_image.shape[2:]
    return dict(
        input_imgs=input_image,
        prompt_imgs=input_image,
        poses=[[pose_out, pose_out_inv], [pose_in, pose_in_inv]],
        height=h,
        width=w,
        T_in=T_in,
        T_out=pose_out.shape[1],
        guidance_scale=guidance_scale,
        num_inference_steps=50,
        generator=generator,
        output_type="numpy",
        output_depth=output_depth,
        in_pos3d=in_pos3d,
        warp_img=warp_img,
        warp_depth=warp_depth,
        ddim_inversion=ddim_inversion,
        gaussian_clip=gaussian_clip,
        depth_mask_cond = depth_mask_cond,
        **kwargs,
    )


def make_pipeline_input_cross_rgbd(
    batch,
    device,
    weight_dtype,
    use_ray,
    output_depth,
    generator,
    guidance_scale=3.0,
    depth_transform=None,
    torch_renderer=None,
    ddim_inversion=False,
    null_layout=False,
    gaussian_clip=0.0,
):
    """
    batch from dataloader

    return kwargs for pipeline
    """
    T_in = batch["image_input"].size(1)
    input_image = batch["image_input"].to(dtype=weight_dtype).to(device)
    pose_in = batch["pose_in"].to(dtype=weight_dtype).to(device)  # BxTx4
    pose_out = batch["pose_out"].to(dtype=weight_dtype).to(device)  # BxTx4
    pose_in_inv = batch["pose_in_inv"].to(dtype=weight_dtype).to(device)  # BxTx4
    pose_out_inv = batch["pose_out_inv"].to(dtype=weight_dtype).to(device)  # BxTx4

    # make args for ddim inversion
    if ddim_inversion:
        raise NotImplementedError("ddim inversion not supported for cross rgbd")
        try:
            # NOTE assume batch size is 1
            assert batch["depth_input"].size(0) == 1
            input_depth = batch["depth_input"].to(device)
            warp_img, warp_depth = render_target_view_torch3d(
                rearrange((input_image + 1) * (255 / 2), "b t c h w -> (b t) h w c").to(
                    torch.float32
                ),
                rearrange(input_depth, "b t h w -> (b t) h w").to(torch.float32),
                rearrange(pose_in.to(torch.float32), "b t c d -> (b t) c d"),
                rearrange(pose_out.to(torch.float32), "b t c d -> (b t) c d"),
                FOV=90.0,
                is_3dfront=True,
                torch_renderer=torch_renderer,
                batch_size=1,
                check_empty_pcd=True,
            )
            warp_img = rearrange(warp_img, "t h w c -> 1 t c h w")
            warp_img = (warp_img.to(dtype=weight_dtype) / 255.0 - 0.5) * 2
            warp_depth = [depth_transform(depth) for depth in warp_depth]
            warp_depth = rearrange(warp_depth, "t h w -> 1 t h w").to(
                dtype=weight_dtype
            )
        except ValueError as e:
            if "empty pointcloud" in e.args[0]:
                warp_img = None
                warp_depth = None
                ddim_inversion = False
            else:
                raise e
    else:
        warp_img = None
        warp_depth = None

    input_image = rearrange(input_image, "b t c h w -> (b t) c h w")  # T_in

    kwargs = {}
    # make args for ray condition
    if use_ray:
        target_rays = [
            v.to(dtype=weight_dtype, device=device)
            for k, v in batch.items()
            if "target_ray" in k
        ]
        target_rays = [
            repeat(x, "b t c h w -> (b k) t c h w", k=2) for x in target_rays
        ]
        kwargs["target_rays"] = target_rays

        cond_rays = [
            v.to(dtype=weight_dtype, device=device)
            for k, v in batch.items()
            if "cond_ray" in k
        ]
        cond_rays = [rearrange(r, "b t c h w -> (b t) c h w") for r in cond_rays]

        if len(cond_rays):
            # NOTE: only support 1 cond_ray
            kwargs["cond_rays"] = cond_rays[0]

    # make args for layout
    if "layout_cls" in batch:
        layout_pos = batch["layout_pos"].to(dtype=weight_dtype, device=device)
        kwargs["layouts"] = {
            "layout_pos": layout_pos,
            "layout_cls": batch["layout_cls"].to(device=device),
        }
        if null_layout:
            kwargs["layouts"]["layout_cls"] = torch.zeros_like(
                kwargs["layouts"]["layout_cls"]
            )
            kwargs["layouts"]["layout_pos"] = torch.zeros_like(
                kwargs["layouts"]["layout_pos"]
            )
    if "back_layout_cls" in batch:
        kwargs["layouts"]["back_layout_cls"] = batch["back_layout_cls"].to(
            device=device
        )
        kwargs["layouts"]["back_layout_pos"] = batch["back_layout_pos"].to(
            dtype=weight_dtype, device=device
        )
        if null_layout:
            kwargs["layouts"]["back_layout_cls"] = torch.zeros_like(
                kwargs["layouts"]["back_layout_cls"]
            )
            kwargs["layouts"]["back_layout_pos"] = torch.zeros_like(
                kwargs["layouts"]["back_layout_pos"]
            )

    # make args for 3d position from input depth
    in_pos3d = batch.get("in_pos3d", None)
    if in_pos3d is not None:
        in_pos3d = in_pos3d.to(dtype=weight_dtype, device=device)
        in_pos3d = repeat(in_pos3d, "b t h w c -> (b k) t h w c", k=2)

    pose_out, pose_out_inv, pose_in, pose_in_inv = map(
        lambda x: repeat(x, "b t c d -> (b k) t c d", k=2),
        [pose_out, pose_out_inv, pose_in, pose_in_inv],
    )
    h, w = input_image.shape[2:]

    return dict(
        input_imgs=input_image,
        prompt_imgs=input_image,
        poses=[[pose_out, pose_out_inv], [pose_in, pose_in_inv]],
        height=h,
        width=w,
        T_in=T_in,
        T_out=pose_out.shape[1],
        guidance_scale=guidance_scale,
        num_inference_steps=50,
        generator=generator,
        output_type="numpy",
        output_depth=output_depth,
        in_pos3d=in_pos3d,
        warp_img=warp_img,
        warp_depth=warp_depth,
        ddim_inversion=ddim_inversion,
        gaussian_clip=gaussian_clip,
        rgbd_cross_attention=True,
        **kwargs,
    )


class DB:
    def __init__(self, path, read_only=False):
        self.read_only = read_only
        if read_only:
            self._db = lmdb.open(
                path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        else:
            self._db = lmdb.open(path, map_size=int(1e12))

    def get_depth(self, key):
        with self._db.begin(write=False) as txn:
            depth_key = key + "_depth"
            depth = txn.get(depth_key.encode("ascii"))
            if depth is None:
                return None
            depth = zlib.decompress(depth)
            depth = np.frombuffer(depth, dtype=np.uint16)
            size = int(np.sqrt(len(depth)))
            depth = depth.reshape(size, size)
            depth = depth.astype(np.float32) * 0.001
        return depth

    def get_calib_depth(self, key):
        with self._db.begin(write=False) as txn:
            depth_key = key + "_calibdepth"
            depth = txn.get(depth_key.encode("ascii"))
            if depth is None:
                return None
            depth = zlib.decompress(depth)
            depth = np.frombuffer(depth, dtype=np.uint16)
            size = int(np.sqrt(len(depth)))
            depth = depth.reshape(size, size)
            depth = depth.astype(np.float32) * 0.001
        return depth

    def get_rgb(self, key):
        with self._db.begin(write=False) as txn:
            rgb_key = key + "_rgb"
            rgb = txn.get(rgb_key.encode("ascii"))
            if rgb is None:
                return None
            rgb = np.frombuffer(rgb, dtype=np.uint8)
            rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        return rgb

    def get_pose(self, key):
        with self._db.begin(write=False) as txn:
            pose_key = key + "_pose"
            pose = txn.get(pose_key.encode("ascii"))
            if pose is None:
                return None
            pose = np.frombuffer(zlib.decompress(pose), dtype=np.float32)
            pose = pose.reshape(4, 4).copy()
            pose[3, 3] = 1
        return pose

    def get_complete_keys(self):
        """
        return key that have rgb
        """
        with self._db.begin(write=False) as txn:
            exist_keys = list(txn.cursor().iternext(values=False))
            exist_keys = [key.decode() for key in exist_keys]
            exist_keys = [key for key in exist_keys if key.endswith("_rgb")]
            exist_keys = [key.replace("_rgb", "") for key in exist_keys]
            exist_keys = sorted(exist_keys)
        return exist_keys

    def write_rgb(self, key, rgb, lossless=False):
        """
        key:str
        rgb: h,w,c uint8
        """
        if self.read_only:
            raise ValueError("read only db")
        assert rgb.dtype == np.uint8
        out_txn = self._db.begin(write=True)
        rgb_key = f"{key}_rgb".encode("ascii")
        if lossless:
            rgb = cv2.imencode(".png", rgb)[1]
        else:
            rgb = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        out_txn.put(rgb_key, rgb)
        out_txn.commit()

    def write_pose(self, key, pose):
        """
        pose 4,4 float32 c2w
        """
        if self.read_only:
            raise ValueError("read only db")
        out_txn = self._db.begin(write=True)
        pose_key = f"{key}_pose".encode("ascii")
        pose = zlib.compress(pose.astype(np.float32).tobytes())
        out_txn.put(pose_key, pose)
        out_txn.commit()

    def write_depth(self, key, depth):
        """
        depth: h,w uint16
        """
        if self.read_only:
            raise ValueError("read only db")
        assert depth.dtype == np.uint16
        out_txn = self._db.begin(write=True)
        depth_key = f"{key}_depth".encode("ascii")
        depth = zlib.compress(depth.tobytes())
        out_txn.put(depth_key, depth)
        out_txn.commit()
