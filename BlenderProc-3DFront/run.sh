# blenderproc run \
# examples/datasets/front_3d_with_improved_mat/render_dataset_improved_mat.py \
# examples/datasets/front_3d_with_improved_mat/3D-FRONT \
# examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model \
# examples/datasets/front_3d_with_improved_mat/3D-FRONT-texture \
# 6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json \
# resources/cctextures/ \
# examples/datasets/front_3d_with_improved_mat/renderings

# python visualization/front3d/vis_front3d.py --json_file 6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json


# render views
# blenderproc run \
# examples/datasets/front_3d_with_improved_mat/render.py \
# examples/datasets/front_3d_with_improved_mat/3D-FRONT \
# examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model \
# examples/datasets/front_3d_with_improved_mat/3D-FRONT-texture \
# resources/cctextures/ \
# /media/hieu/T7/3D-FRONT-renderings \
# --locations_folder /media/hieu/T7/3d-front/locations
# python visualization/front3d/vis_front3d.py --json_file 6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json



blenderproc run \
examples/datasets/front_3d_with_improved_mat/render_layout.py \
examples/datasets/front_3d_with_improved_mat/3D-FRONT \
/media/hieu/T7/3D-FRONT-laytout_renderings \
--locations_folder /media/hieu/T7/3d-front/locations \
--layout_folder /media/hieu/T7/3d-front/layouts
# python visualization/front3d/vis_front3d.py --json_file 6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json