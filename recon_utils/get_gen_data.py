import os
import shutil
import json
import lmdb
import cv2
import numpy as np
import zlib
import math
import argparse

'''
execute this script from chen.yiwe accout so it is easier to download
'''

class gen_data_loader():
    def __init__(self, 
                base_dir = '/work/vig/hieu/gen300/insepction0212',
                dst_path = '/scratch/chen.yiwe/generated_data_v0'):
        self.base_dir = base_dir
        self.dst_path = dst_path
        os.makedirs(self.dst_path, exist_ok=True)

    def transfer_gen_scene(self, scene_name):
        scene_gen_path = os.path.join(self.base_dir, scene_name)
        scene_dst_path = os.path.join(self.dst_path, scene_name)
        if os.path.exists(scene_dst_path):
            return True
        os.makedirs(scene_dst_path, exist_ok=True)
        # copy the directory, if exist, overwrite
        if os.path.exists(scene_dst_path):
            shutil.rmtree(scene_dst_path)
        shutil.copytree(scene_gen_path, scene_dst_path, dirs_exist_ok=True)
        print(f'finished transferring {scene_name}')
        return False
    
    def decode_scene(self, scene_name):
        scene_dst_path = os.path.join(self.dst_path, scene_name)
        if not os.path.exists(os.path.join(scene_dst_path, 'colors')):
            os.makedirs(os.path.join(scene_dst_path, 'colors'))
        if not os.path.exists(os.path.join(scene_dst_path, 'depth')):
            os.makedirs(os.path.join(scene_dst_path, 'depth'))
        if not os.path.exists(os.path.join(scene_dst_path, 'cam_Ts')):
            os.makedirs(os.path.join(scene_dst_path, 'cam_Ts'))
        db_file_path = os.path.join(scene_dst_path, 'db', scene_name)
        env = lmdb.open(
                os.path.join(db_file_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        with env.begin() as txn:
            exist_keys = list(txn.cursor().iternext(values=False))
            exist_keys = [key.decode() for key in exist_keys]
            print('exp:', exist_keys[0:50])
            exist_keys = [key for key in exist_keys if "rgb" in key]
            print(f'found {len(exist_keys)} frames in {scene_name}')
            
            i = 0
            for key in exist_keys:
                frame_index = key.split('_')[0]
                color_raw = txn.get(key.encode("ascii"))
                color_raw = np.frombuffer(color_raw, dtype=np.uint8)
                color_raw = cv2.imdecode(color_raw, cv2.IMREAD_COLOR)
                color_raw = color_raw[:, :, ::-1]
                cv2.imwrite(os.path.join(scene_dst_path, 'colors', f'{frame_index}.png'), color_raw)
                # print('depth key:', frame_index + '_depth')
                depth_key = frame_index + '_depth'
                depth_raw = txn.get(depth_key.encode("ascii"))
                depth_raw = zlib.decompress(depth_raw)
                depth_raw = np.frombuffer(depth_raw, dtype=np.uint16)
                size = math.sqrt(depth_raw.shape[0])
                depth_raw = depth_raw.reshape((int(size), int(size)))
                np.save(os.path.join(scene_dst_path, 'depth', key + ".npy"), depth_raw)
                #also save the depth in grayscale
                cv2.imwrite(os.path.join(scene_dst_path, 'depth', f'{frame_index}.png'), depth_raw)
                
                cam_T_key = frame_index + '_pose'
                cam_T_raw = txn.get(cam_T_key.encode("ascii"))
                cam_T_raw = zlib.decompress(cam_T_raw)
                cam_T_raw = np.frombuffer(cam_T_raw, dtype=np.float32)
                cam_T_raw = cam_T_raw.reshape(4, 4)
                np.save(os.path.join(scene_dst_path, 'cam_Ts', f'{frame_index}.npy'), cam_T_raw)
                i += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some scenes.')
    parser.add_argument('--base_dir', 
                        type=str, 
                        default='/work/vig/hieu/gen300/insepction0212', 
                        help='Base directory of the scenes')
    parser.add_argument('--dst_path', 
                        type=str, 
                        default='/scratch/chen.yiwe/generated_data_v0',
                        help='Destination path for the generated data')
    parser.add_argument('--get_all_scenes',
                        type=bool,
                        default=False,
                        help='Get all scenes in the base_dir')
    args = parser.parse_args()

    # base_dir = '/work/vig/hieu/gen300/gen_result_inspection_gt_reference_exp'
    # dst_path = '/scratch/chen.yiwe/generated_data_gt_condition_256'   
    gen_loader = gen_data_loader(base_dir=args.base_dir, dst_path=args.dst_path)
    # scenes = os.listdir('/work/vig/hieu/gen300/iodepth_layoutobj_3m_10k_final_vae_ft_noddim_inspection')
    if args.get_all_scenes:
        scenes = os.listdir(args.base_dir)
    else:
        scenes = [
                # '6634054e-6ff5-43d2-958f-a80bb7eee357', 
                # '6b56575f-a746-4062-8296-b566d5de7b60', 
                '64cce374-230b-4fe2-8240-69f81c8cfb33',
                # '651c37ce-c0cd-47c6-843f-3aa192235e39',
                ]
    # # transfer the gt data
    for scene in scenes:
        transferred = gen_loader.transfer_gen_scene(scene)
        if not transferred:
            gen_loader.decode_scene(scene)
