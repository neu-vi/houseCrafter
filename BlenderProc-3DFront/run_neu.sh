# blenderproc run \
# examples/datasets/front_3d_with_improved_mat/render_dataset_improved_mat.py \
# examples/datasets/front_3d_with_improved_mat/3D-FRONT \
# examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model \
# examples/datasets/front_3d_with_improved_mat/3D-FRONT-texture \
# 6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json \
# resources/cctextures/ \
# examples/datasets/front_3d_with_improved_mat/renderings

# python visualization/front3d/vis_front3d.py --json_file 6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json



blenderproc run \
examples/datasets/front_3d_with_improved_mat/render.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/BlenderProc-3DFront/resources/cctextures/ \
/work/vig/hieu/3dfront_data/images_100scenes_pano \
--locations_folder /work/vig/hieu/3dfront_data/locations_100_scene\
--error_folder /media/hieu/T7/3dfront_render/images_100scenes_pano_error \
--end 100 \
--offset 0 \
--step 8 \
# python visualization/front3d/vis_front3d.py --json_file 6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json