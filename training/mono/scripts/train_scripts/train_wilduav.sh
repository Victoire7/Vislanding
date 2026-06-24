cd ../../../

#python  mono/tools/train.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav2.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/weight/metric_depth_vit_small_800k.pth
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250113_190822/ckpt/step00001900.pth

#python  mono/tools/train_safe.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_safe_seg1_lw4-1_reverse.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250114_195813/ckpt/step00020000.pth

# python  mono/tools/train_safe.py \
#         --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_safe_seg1_lw3-1_dice3_reverse.py \
#         --use-tensorboard True \
#         --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250114_195813/ckpt/step00020000.pth

#python  mono/tools/train_safe.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_safe_seg1_lw2-1_reverse.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250114_195813/ckpt/step00020000.pth

#python  mono/tools/train_safe.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_safe_seg1_lw1-1_gru0.1.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250114_195813/ckpt/step00020000.pth
#
#python  mono/tools/train_safe.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_safe_seg1_lw1-1_dice2.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250114_195813/ckpt/step00020000.pth
#
#python  mono/tools/train_safe.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_safe_seg1_lw1-1_dice3.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250114_195813/ckpt/step00020000.pth
#
#python  mono/tools/train_safe.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_safe_seg1_lw1-1.5.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250114_195813/ckpt/step00020000.pth

#python  mono/tools/train_safe.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_safe_seg1_lw1-3.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250114_195813/ckpt/step00020000.pth

#python  mono/tools/train_safe.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_safe_seg1_conf.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250114_195813/ckpt/step00020000.pth

#python  mono/tools/train_safe.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_safe_seg1_0.5.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250114_195813/ckpt/step00020000.pth
#
#python  mono/tools/train_safe.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_safe_seg1_conf_0.5.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250114_195813/ckpt/step00020000.pth

#python  mono/tools/train_safe.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav_onlysafe.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav_safe_bak/20250118_000312/ckpt/step00004000.pth


# python  mono/tools/train_safe_semi_new.py \
#         --config mono/configs/2026_new/vit.raft5.small.sd_safe_base_rawmetric.py \
#         --use-tensorboard False \
#         --load-from /home/vector/Tan/xunfeidan/Metric3D/weight/metric_depth_vit_small_800k.pth

# python  mono/tools/train_safe_semi_new.py \
#         --config mono/configs/2026_new/vit.raft5.small.wilduav_safe_lw2-1_reverse_bs8_multi_nosemi_nohidden_freezedepth_rawmetric.py \
#         --use-tensorboard False \
#         --load-from /home/vector/Tan/xunfeidan/Metric3D/weight/metric_depth_vit_small_800k.pth




# python  mono/tools/train_safe_semi_new.py \
#         --config mono/configs/2026_new/vit.raft5.small.sd_safe_base-depth-normal.py \
#         --use-tensorboard False \
#         --load-from /home/vector/Tan/xunfeidan/Metric3D/training/202504/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth

# python  mono/tools/train_safe_semi_new.py \
#         --config mono/configs/2026_new/vit.raft5.small.sd_safe_base-depth.py \
#         --use-tensorboard False \
#         --load-from /home/vector/Tan/xunfeidan/Metric3D/training/202504/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth

# python  mono/tools/train_safe_semi_new.py \
#         --config mono/configs/2026_new/vit.raft5.small.sd_safe_base-normal.py \
#         --use-tensorboard False \
#         --load-from /home/vector/Tan/xunfeidan/Metric3D/training/202504/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth

python  mono/tools/train_safe_semi_new.py \
        --config mono/configs/2026_new/vit.raft5.small.wilduav_safe_lw2-1_reverse_bs8_multi_nosemi_nohidden_freezedepth-normal.py \
        --use-tensorboard False \
        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/202504/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth

# python  mono/tools/train_safe_semi_new.py \
#         --config mono/configs/2026_new/vit.raft5.small.wilduav_safe_lw2-1_reverse_bs8_multi_nosemi_nohidden_freezedepth-normal-depth.py \
#         --use-tensorboard False \
#         --load-from /home/vector/Tan/xunfeidan/Metric3D/training/202504/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth

# python mono/tools/train_safe_semi_new.py \
#         --config mono/configs/2026_new/vit.raft5.small.wilduav_safe_lw2-1_reverse_bs8_multi_nosemi_nohidden_freezedepth-depth.py \
#         --use-tensorboard False \
#         --load-from /home/vector/Tan/xunfeidan/Metric3D/training/202504/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth