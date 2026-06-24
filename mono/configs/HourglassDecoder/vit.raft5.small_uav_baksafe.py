_base_=[
       '../_base_/models/encoder_decoder/dino_vit_small_reg.dpt_raft.py',
       '../_base_/datasets/_data_base_.py',
       '../_base_/default_runtime.py',
       ]

import numpy as np
model=dict(
    decode_head=dict(
        type='RAFTDepthNormalBakSafeDPT5',
        iters=4,
        n_downsample=2,
        detach=False,
    )
)


# max_value = 300
# configs of the canonical space
data_basic = dict(
    canonical_space=dict(
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(1, 400),
    clip_depth_range=(1, 400),
    crop_size=(616, 1064),  # %28 = 0
)


batchsize_per_gpu = 1
thread_per_gpu = 1
with_safe = True
single_step_safe = False
semi_entropy_only = False
freeze_depth = False
flow_ablation = [True, True, False]
with_refine = False
