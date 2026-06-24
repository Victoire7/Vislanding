cd ../../../

#python  mono/tools/train.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav2.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/weight/metric_depth_vit_small_800k.pth
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250113_190822/ckpt/step00001900.pth

#python  mono/tools/train_safe_semi.py \
#        --config mono/configs/semi/vit.raft5.small.wilduav_semi_new.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/weight/metric_depth_vit_small_800k.pth

#python  mono/tools/train_safe_semi.py \
#        --config mono/configs/semi/vit.raft5.small.wilduav_safe_seg1_lw3-1_dice2_reverse_semi_bs8_args_test.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav_semi/20250207_011955/best_depth_ckpt/student_step00017600_0.0934.pth
#

#python  mono/tools/train_safe_semi_new.py \
#        --config mono/configs/ablation/vit.raft5.small.wilduav_safe_lw3-1_reverse_bs8_multi_newsemi2_0.7-nodice_all_freezedepth_nokl.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/student_step00019800_0.0987.pth
#
#python  mono/tools/train_safe_semi_new.py \
#        --config mono/configs/ablation/vit.raft5.small.wilduav_safe_lw3-1_reverse_bs8_multi_newsemi2_0.7-nodice_noall_freezedepth_nokl.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/student_step00019800_0.0987.pth
#
#python  mono/tools/train_safe_semi_new.py \
#        --config mono/configs/ablation/vit.raft5.small.wilduav_safe_lw3-1_reverse_bs8_multi_newsemi2_0.7-nodice_nohidden_freezedepth_kl3.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/student_step00019800_0.0987.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/ablation/vit.raft5.small.wilduav_safe_lw1.5-1_reverse_bs8_multi_semi_nohidden_freezedepth_ema0.3.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/student_step00019800_0.0987.pth

#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/ablation/vit.raft5.small.wilduav_safe_lw1.5-1_reverse_bs8_multi_nosemi_nohidden_freezedepth.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/ablation/vit.raft5.small.wilduav_safe_lw1.5-1_reverse_bs8_multi_nosemi_nohidden_freezedepth_punish2.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/ablation/vit.raft5.small.wilduav_safe_lw1.5-1_reverse_bs8_multi_nosemi_noall_freezedepth.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/ablation/vit.raft5.small.wilduav_safe_lw1.5-1_reverse_bs8_multi_nosemi_nohidden_freezedepth2.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/ablation/vit.raft5.small.wilduav_safe_lw1.5-1_reverse_bs8_multi_nosemi_nohidden_freezedepth_punish2.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/ablation/vit.raft5.small.wilduav_safe_lw1-1_reverse_bs8_multi_nosemi_nohidden_freezedepth_newsize_refine.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
#
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/ablation/vit.raft5.small.wilduav_safe_lw2-1_reverse_bs8_multi_nosemi_nohidden_freezedepth_newsize_refine.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/0226/vit.raft5.small.wilduav_safe_lw2-1_bs8_multi_nosemi_nohidden_freezedepth_ASPP.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/0226/vit.raft5.small.wilduav_safe_lw2-1_bs8_multi_nosemi_nohidden_freezedepth_focal.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
#
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/0226/vit.raft5.small.wilduav_safe_lw2-1_bs8_multi_nosemi_nohidden_freezedepth_focal_dice.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
python  mono/tools/train_safe_semi_new.py \
        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/0226/vit.raft5.small.wilduav_safe_lw2-1_bs8_multi_nosemi_nohidden_freezedepth_baseline.py \
        --use-tensorboard True \
        --load-from /home/vector/Tan/xunfeidan/Metric3D/weight/metric_depth_vit_small_800k.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/0226/vit.raft5.small.wilduav_safe_lw1-1_reverse_bs8_multi_nosemi_nohidden_freezedepth_focal_samplewise_finalonly.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/0226/vit.raft5.small.wilduav_safe_lw1-1_reverse_bs8_multi_nosemi_nohidden_freezedepth_focal_samplewise.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config /home/vector/Tan/xunfeidan/Metric3D/training/mono/configs/0226/vit.raft5.small.wilduav_safe_lw1-1_reverse_bs8_multi_nosemi_withhidden_freezedepth_focal_samplewise.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
##
#python  mono/tools/train_safe_semi_new.py \
#        --config mono/configs/ablation/vit.raft5.small.wilduav_safe_lw1-1_reverse_bs8_multi_nosemi_0.9-nodice_nohidden_freezedepth_nokl_biglrema_emafreeze.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/student_step00019800_0.0987.pth
#
#python  mono/tools/train_safe_semi_new.py \
#        --config mono/configs/ablation/vit.raft5.small.wilduav_safe_lw1-1_reverse_bs8_multi_newsemi2_0.9-nodice_nohidden_freezedepth_nokl_biglrema_emafreeze.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/student_step00019800_0.0987.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config mono/configs/ablation/vit.raft5.small.wilduav_safe_lw3-1_reverse_bs8_args_test_ema1000_multi_newsemi2_0.9_test1-nodice_noall.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs_exp/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/student_step00019800_0.0987.pth
#python  mono/tools/train_safe_semi_new.py \
#        --config mono/configs/semi/vit.raft5.small.wilduav_newsemi_new.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/weight/metric_depth_vit_small_800k.pth
#python  mono/tools/train_safe_semi.py \
#        --config mono/configs/semi/vit.raft5.small.wilduav_safe_lw3-1_dice2_reverse_bs8_args_test_ema1000_multi_semi2_0.9_test1_loadteacher.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/teacher_step00019800_0.0986.pth
##
#
#python  mono/tools/train_safe_semi.py \
#        --config mono/configs/semi/vit.raft5.small.wilduav_safe_lw3-1_dice2_reverse_bs8_args_test_ema1000_multi_semi3_0.9_test1_semienonly.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav_semi_new/20250212_214200/best_depth_ckpt/student_step00019800_0.0987.pth
#***************************************************************************************************************************************************************************
#python  mono/tools/train_safe_semi.py \
#        --config mono/configs/semi/vit.raft5.small.wilduav_safe_lw3-1_dice2_reverse_bs8_args_test_ema1000_multi_semi2_0.7_test2.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav_safe_lw3-1_dice2_reverse_bs8_args_test_ema1000_multi_semi2_0.9_test1/20250212_035904/step2000.pth

#python  mono/tools/train_safe_semi.py \
#        --config mono/configs/semi/vit.raft5.small.wilduav_safe_lw3-1_dice2_reverse_bs8_args_test_ema1000_multi_semi2_0.9_test1.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav_semi/20250207_011955/best_depth_ckpt/student_step00017600_0.0934.pth

#python  mono/tools/train_safe_semi.py \
#        --config mono/configs/semi/vit.raft5.small.wilduav_safe_lw3-1_dice2_reverse_bs8_args_test_ema2000_multi.py \
#        --use-tensorboard True \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav_semi/20250207_011955/best_depth_ckpt/student_step00017600_0.0934.pth

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