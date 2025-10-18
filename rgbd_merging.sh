cd recon_utils
python get_gen_data.py --base_dir ../gen_rgbd --dst_path ../generated_data_v0 --get_all_scenes True
python fuse_gen_data.py --gen_dir ../generated_data_v0