import torch
import numpy as np
import cv2
import lmdb
import zlib
import open3d as o3d
from data_modules.data_utils import make_pcd_batch
from tqdm import tqdm
import json
FRONT3D_BASE_MATRIX = torch.tensor(
    [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
    dtype=torch.float32,
)

def load_data(db_path, skips = []):
    db = lmdb.open(db_path, readonly=True)
    imgs, depths, poses = [], [], []
    with db.begin(write=False) as txn:
        views = list(txn.cursor().iternext(values=False))
        views = [key.decode() for key in views]
        views = [key.split("_")[0] for key in views if "rgb" in key]
        views = sorted(list(set(views)))
        views = [view for view in views if view not in skips]
        for view in views:
            rgb_key = f"{view}_rgb".encode("ascii")
            img = txn.get(rgb_key)
            img = np.frombuffer(img, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if img.shape[0] != 256:
                continue
            imgs.append(torch.tensor(img, dtype=torch.float32))

            pose_key = f"{view}_pose".encode("ascii")
            c2w = txn.get(pose_key)
            c2w = (
                np.frombuffer(zlib.decompress(c2w), dtype=np.float32)
                .reshape(4, 4)
                .copy()
            )
            c2w[3, 3] = 1.0
            poses.append(torch.tensor(c2w, dtype=torch.float32))

            depth_key = f"{view}_depth".encode("ascii")
            depth = zlib.decompress(txn.get(depth_key))
            # depth = np.frombuffer(depth, dtype=np.uint16).reshape(
            #     *self.depth_shape_db
            # )
            depth = np.frombuffer(depth, dtype=np.uint16)
            size = int(np.sqrt(len(depth)))
            depth = depth.reshape(size, size)
            # convert from mm to m in this case depth_scale = 1000
            depth = depth.astype(np.float32) / 1000.0
            depths.append(torch.tensor(depth, dtype=torch.float32))
    return imgs, depths, poses, views


def get_scene_pcd(imgs, depths, poses, K):
    pcds, colors = [], []
    for img, depth, pose in tqdm(zip(imgs, depths, poses)):
        pose[:3,:3] = pose[:3, :3] @ FRONT3D_BASE_MATRIX
        pcd, color = make_pcd_batch(
            img[None, ...], depth[None, ...], pose[None, ...], K, dilate_invalid=2
        )
        pcds.append(pcd.numpy())
        colors.append(color.numpy() / 255.0)
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(np.concatenate(pcds, axis=0))
    o3d_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors, axis=0))
    return o3d_pcd

def clean_pcd(pcd, voxel_size=0.05):
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


if __name__ == "__main__":
    # root_dir = "/mnt/Data/hieu/escher/sample_vis/iodepth_affine600_00001quantile_4hop_ddim_inversion_wholescene"
    root_dir = "/media/hieu/T7/3dfront_gen/sample_vis/iodepth_affine600_00001quantile_4hop_ddim_inversion_bedroom"
    # scene_id = "0337f31c-ff58-48a8-8a03-1db9f02e6471"
    scene_id = "031fba32-7c48-4c1e-8342-aab66d6e531f"
    db_path = f"{root_dir}/db/{scene_id}"
    # pcd_path = f"/home/hieu/Downloads/wholescene_{scene_id}.ply"
    pcd_path = f"/home/hieu/Downloads/bedroom.ply"
    meta_path = f"{root_dir}/sequence_{scene_id}.json"
    
    meta = json.load(open(meta_path, "r"))
    sequence = list(meta.values())[0]
    target_frames = sum([x["frame_ids"]["target"] for x in sequence], [])
    depth_methods = sum([x["depth_method"] for x in sequence], [])
    skip_frames = [view for view, method in zip(target_frames, depth_methods) if method != "layout"]
    image_size = 256
    FOV = 90
    center = image_size / 2
    focal = image_size / 2 / np.tan(np.radians(FOV / 2))
    K = torch.tensor(
        [
            [focal, 0.0, center],
            [0.0, focal, center],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float,
    )
    imgs, depths, poses, views = load_data(db_path, skip_frames)
    o3d_pcd = get_scene_pcd(imgs, depths, poses, K)
    print(len(o3d_pcd.points))
    o3d_pcd = clean_pcd(o3d_pcd, 0.1)
    print(len(o3d_pcd.points))
    o3d.io.write_point_cloud(pcd_path, o3d_pcd)
