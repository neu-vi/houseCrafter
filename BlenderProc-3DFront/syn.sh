#!/bin/bash
set -x

TEMP_FILE=.tmp
PROJECT_NAME=$(basename "`pwd`")
echo "Run training for project $PROJECT_NAME"

/bin/cat <<EOM >$TEMP_FILE
*__pycache__*

cache/*
logs/*
pretrained/*
notebooks/*
*runs/*
apex/*
*.png
*outputs/*
history_/*
*egg*
*.pth
push_code.sh
*.ipynb
.git/*
tmp
# *.csv
data/visible*
data/topdown*
detectron2*
Mask2Former*
X-Decoder/*
bevpool/*
*.pt
.vscode*
*.npy
*.pkl
data/*
*TbHJrupSAjP*
*wandb*
# *.sh
*.ply
*logs_*
*.safetensors
cctextures/*
eschernet_layout_16k_ckpt/*
*.tar.xz
*.hdf5
images_renderings/*
debugging/*
EOM

if [ "$1" == "neu_trans" ]; then
    echo "Push code to shared dir"
    REMOTE_HOME="/work/vig/hieu/BlenderProc-3DFront"

elif [ "$1" == "vig_hieu" ]; then
    echo "Push code to shared dir"
    REMOTE_HOME="/mnt/Data/hieu/BlenderProc-3DFront"
elif [ "$1" == "xps" ]; then
    echo "Push code to shared dir"
    REMOTE_HOME="/home/hieu/vln_map"
elif [ "$1" == "gcp" ]; then
    echo "Push code to shared dir"
    REMOTE_HOME="/home/hieu/hm3dautovln"

elif [ "$1" == "gcp2" ]; then
    echo "Push code to shared dir"
    REMOTE_HOME="/home/hieu/hm3dautovln"

elif [ "$1" == "gcp3" ]; then
    echo "Push code to shared dir"
    REMOTE_HOME="/home/hieu/hm3dautovln"

else
    echo "Unknown server"
    exit
fi
# config jump server
# if [ "$1" == "medical" ] || [ "$1" == "dgx1" ]; then
#     JUMP=""
# else
#     echo "dgx2 server, need to use dgx1 as proxy"
#     JUMP="-J dgx1"
# if [ "$1" == "dgx1" ]; then
#     JUMP="-J dev@localhost:69"
# elif [ "$1" == "dgx2" ]; then
#     JUMP="-J dev@localhost:69,dev@10.100.53.77:8008"
# fi
# push code to server
rsync -vr -P --exclude-from $TEMP_FILE "$PWD/" $1:$REMOTE_HOME/
# rsync -vr -P $1:"$REMOTE_HOME/test/test_bev_net.ipynb" $PWD/test/
# rsync -vr -P $1:"$REMOTE_HOME/test/test_dataloading.ipynb" $PWD/test/
# rsync -vr -P $1:"$REMOTE_HOME/test/test_compute_semseg.ipynb" $PWD/test/
# rsync -vr -P $1:"$REMOTE_HOME/test/viz_prediction.ipynb" $PWD/test/
# rsync -vr -P --exclude-from $TEMP_FILE $1:$REMOTE_HOME/ "$PWD/"
# pull model weights and log files from server
# remove temp. file
rm $TEMP_FILE



# rsync -avr -P --exclude=*output/* /home/hieu/lib/det2/projects/spine medical:/home/dev/hieunt/det2/projects/