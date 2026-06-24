cd ../../../

python  mono/tools/train.py \
        --config mono/configs/RAFTDecoder/vit.raft5.small.ddos.py \
        --use-tensorboard True \
        --load-from /home/vector/Tan/xunfeidan/Metric3D/weight/metric_depth_vit_small_800k.pth \
        --experiment_name ddos3
