cd ../../../

CUDA_VISIBLE_DEVICES=1 python  mono/tools/test.py \
        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav2.py \
        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav2/20250113_221319/best_ckpt/step00007400_0.0798.pth

#CUDA_VISIBLE_DEVICES=0 python  mono/tools/test.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav.py \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav+midair_new/20250110_000239/ckpt/step00001550.pth

#CUDA_VISIBLE_DEVICES=0 python  mono/tools/test.py \
#        --config mono/configs/RAFTDecoder/vit.raft5.small.wilduav+midair_sig.py \
#        --load-from /home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav+midair_sig/20250109_162711/ckpt/step00002250.pth
#CUDA_VISIBLE_DEVICES=0 python  mono/tools/test.py \
#        mono/configs/RAFTDecoder/vit.raft5.small.wilduav.py \
#        --load-from training/work_dirs/vit.raft5.small.wilduav/20241231_113740/ckpt/step00002900.pth
#
#CUDA_VISIBLE_DEVICES=0 python  mono/tools/test.py \
#        mono/configs/RAFTDecoder/vit.raft5.small.wilduav.py \
#        --load-from training/work_dirs/vit.raft5.small.wilduav/20241231_113740/ckpt/step00003000.pth
#
#CUDA_VISIBLE_DEVICES=0 python  mono/tools/test.py \
#        mono/configs/RAFTDecoder/vit.raft5.small.wilduav.py \
#        --load-from training/work_dirs/vit.raft5.small.wilduav/20241231_113740/ckpt/step00003100.pth
#
#CUDA_VISIBLE_DEVICES=0 python  mono/tools/test.py \
#        mono/configs/RAFTDecoder/vit.raft5.small.wilduav.py \
#        --load-from training/work_dirs/vit.raft5.small.wilduav/20241231_113740/ckpt/step00003200.pth