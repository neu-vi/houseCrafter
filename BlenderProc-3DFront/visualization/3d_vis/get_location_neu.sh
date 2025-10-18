python get_locations.py \
  --locations_folder /work/vig/hieu/3dfront_data/locations_100_scene_7 \
  --front_folder /work/vig/Datasets/3D-Front/3D-FRONT \
  --future_folder /work/vig/Datasets/3D-Front/3D-FUTURE-model \
  --future_bbox_folder /work/vig/hieu/3dfront_data/model_bbox \
  --error_folder /work/vig/hieu/3dfront_data/locations_error_test \
  --dist 0.7 \
  --end 100


python get_poses.py \
  --poses_folder /work/vig/hieu/3dfront_data/poses_100 \
  --front_folder /work/vig/Datasets/3D-Front/3D-FRONT \
  --future_folder /work/vig/Datasets/3D-Front/3D-FUTURE-model \
  --future_bbox_folder /work/vig/hieu/3dfront_data/model_bbox \
  --error_folder /work/vig/hieu/3dfront_data/poses_error_test \
  --dist 0.2 \
  --end 100