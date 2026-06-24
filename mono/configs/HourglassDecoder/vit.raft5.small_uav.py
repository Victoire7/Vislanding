_base_=[
       '../_base_/models/encoder_decoder/dino_vit_small_reg.dpt_raft.py',
       '../_base_/datasets/_data_base_.py',
       '../_base_/default_runtime.py',
       ]

import numpy as np
model=dict(
    decode_head=dict(
        type='RAFTDepthNormalDPT5',
        iters=4,
        n_downsample=2,
        detach=False,
    )
)


max_value = 300
# configs of the canonical space
data_basic=dict(
    canonical_space = dict(
        # img_size=(540, 960),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(10, 300),
#     crop_size=(544, 1216),
#     crop_size = (544, 992),
    clip_depth_range=(10, 300),
    crop_size = (616, 1064),  # %28 = 0
)


batchsize_per_gpu = 1
thread_per_gpu = 1
