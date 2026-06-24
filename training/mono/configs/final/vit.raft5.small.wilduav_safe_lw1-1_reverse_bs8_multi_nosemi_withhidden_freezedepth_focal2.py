_base_ = ['../_base_/losses/all_losses.py',
          '../_base_/models/encoder_decoder/dino_vit_small_reg.dpt_raft.py',
          '../_base_/datasets/safe_wilduav.py',
          '../_base_/datasets/new_unlabel_safe_wilduav.py'
          ]

import numpy as np

model = dict(
    decode_head=dict(
        type='RAFTDepthNormalSafeDPT5',
        iters=4,
        n_downsample=2,
        detach=False,
    ),
)

# loss method
losses = dict(
    decoder_losses=[
        # dict(type='VNLoss', sample_ratio=0.2, loss_weight=0.1),
        # dict(type='GRUSequenceLoss', loss_weight=0.01, loss_gamma=0.9, stereo_sup=0.0),
        # dict(type='DeNoConsistencyLoss', loss_weight=0.01, loss_fn='CEL', scale=2),
        dict(type='SemiSafeEntropyLoss', loss_weight=1, use_conf=False, class_weight=[1.0, 1.0], final_only=False,
             loss_raw_size=False),
        # dict(type='SafeStandardDiceLoss', loss_weight=1),
        dict(type='SafeFocalLoss', loss_weight=2)
    ],
)

data_array = [
    [
        dict(Safe_WildUAV='SafeWildUAV_dataset'),
        dict(NewUnlabel_Safe_WildUAV='NewUnlabelSafeWildUAV_dataset')
    ],
]

# configs of the canonical space
data_basic = dict(
    canonical_space=dict(
        # img_size=(540, 960),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(1, 400),
    #     crop_size=(544, 1216),
    #     crop_size = (544, 992),
    clip_depth_range=(1, 400),
    crop_size=(616, 1064),  # %28 = 0
)

# online evaluation
# evaluation = dict(online_eval=True, interval=1000, metrics=['abs_rel', 'delta1', 'rmse'], multi_dataset_eval=True)
# log_interval = 100

interval = 200
log_interval = 10
val_interval = 200
semi_start_iter = 6000
ema_start_iter = 1000
ema_momentum = 0.2
safe_pseudo_thresh = 0.9
depth_pseudo_thresh = [0.8, 1.1]
freeze_when_ema = False
with_safe = True
single_step_safe = False
freeze_depth = True
semi_entropy_only = False
with_refine = False
use_kl = False
semi_weight = 2.0
flow_ablation = [True, True, True]  # depth, normal, fuse_feature
evaluation = dict(
    online_eval=True,
    interval=val_interval,
    metrics=['abs_rel', 'delta1', 'rmse', 'normal_mean', 'normal_rmse', 'normal_a1'],
    multi_dataset_eval=True,
    exclude=['DIML_indoor', 'GL3D', 'Tourism', 'MegaDepth'],
    seg_eval_cfg=dict(
        metrics=['mIoU', 'mDice', 'mFscore'],
        class_names=['safe', 'unsafe'],
        ignore_index=2,
        thresh=0.7,
    )
)

# save checkpoint during training, with '*_AMP' is employing the automatic mix precision training
checkpoint_config = dict(by_epoch=False, interval=interval, max_keep_ckpts=20)
runner = dict(type='IterBasedRunner_AMP', max_iters=5010)

# optimizer
optimizer = dict(
    type='AdamW',
    encoder=dict(lr=5e-7, betas=(0.9, 0.999), weight_decay=0, eps=1e-10),
    decoder=dict(lr=3e-5, betas=(0.9, 0.999), weight_decay=0, eps=1e-10),
    # encoder=dict(lr=1e-6, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-6),
    # decoder=dict(lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
    strict_match=True
)
# optimizer = dict(
#     type='AdamW',
# #     encoder=dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
#     encoder=dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-6),
#     decoder=dict(lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
# )
# schedule
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=20,
                 warmup_ratio=1e-6,
                 power=0.9, min_lr=1e-5, by_epoch=False)
# lr_config = dict(policy='poly',
#                  warmup='linear',
#                  warmup_iters=500,
#                  warmup_ratio=1e-6,
#                  power=0.9, min_lr=1e-6, by_epoch=False)

acc_batch = 2
batchsize_per_gpu = 16
thread_per_gpu = 4
dist_params = dict(backend='nccl', port=29500)
test_metrics = ['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3', 'rmse_log', 'log10', 'normal_mean',
                'normal_rmse', 'normal_median', 'normal_a3', 'normal_a4', 'normal_a5']
SafeWildUAV_dataset = dict(
    data=dict(
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                      dict(type='LabelScaleCononical'),
                      # dict(type='ResizeCanonical', ratio_range=(0.9, 1.4)),
                      dict(type='RandomResize',
                           prob=0.8,
                           ratio_range=(0.5, 0.99),
                           is_lidar=False),
                      dict(type='RandomCrop',
                           crop_size=(0, 0),  # crop_size will be overwriteen by data_basic configs
                           crop_type='rand',
                           ignore_label=-1,
                           padding=[0, 0, 0]),
                      dict(type='RandomEdgeMask',
                           mask_maxsize=50,
                           prob=0.2,
                           rgb_invalid=[0, 0, 0],
                           label_invalid=-1, ),
                      dict(type='RandomHorizontalFlip',
                           prob=0.4),
                      dict(type='PhotoMetricDistortion',
                           to_gray_prob=0.1,
                           distortion_prob=0.1, ),
                      dict(type='Weather',
                           prob=0.05),
                      dict(type='RandomBlur',
                           prob=0.05),
                      dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                      dict(type='ToTensor'),
                      dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                      ],
            # sample_size = 10,
        ),
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                      dict(type='LabelScaleCononical'),
                      dict(type='ResizeKeepRatio',
                           resize_size=(616, 1064),
                           # (544, 992), #(768, 1088), #(768, 1120), # (768, 1216), #(768, 1024), # (768, 1216),  #(768, 1312), #
                           ignore_label=-1,
                           padding=[0, 0, 0]),
                      dict(type='ToTensor'),
                      dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                      ],
            sample_size=1200,
        ),
        test=dict(
            pipeline=[dict(type='BGR2RGB'),
                      dict(type='LabelScaleCononical'),
                      dict(type='ResizeKeepRatio',
                           resize_size=(616, 1064),
                           # (544, 992), #(768, 1088), #(768, 1120), # (768, 1216), #(768, 1024), # (768, 1216),  #(768, 1312), #
                           ignore_label=-1,
                           padding=[0, 0, 0]),
                      dict(type='ToTensor'),
                      dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                      ],
            sample_ratio=1.0,
            sample_size=-1,
        )
    ))
NewUnlabelSafeWildUAV_dataset = dict(
    data=dict(
        train=dict(
            strong_pipeline=[dict(type='BGR2RGB'),
                             dict(type='RandomResize',
                                  prob=0.8,
                                  ratio_range=(0.5, 0.99),
                                  is_lidar=False),
                             dict(type='RandomCrop',
                                  crop_size=(0, 0),  # crop_size will be overwriteen by data_basic configs
                                  crop_type='rand',
                                  ignore_label=-1,
                                  padding=[0, 0, 0]),
                             dict(type='RandomEdgeMask',
                                  mask_maxsize=50,
                                  prob=0.2,
                                  rgb_invalid=[0, 0, 0],
                                  label_invalid=-1, ),
                             dict(type='RandomHorizontalFlip',
                                  prob=0.4),
                             dict(type='PhotoMetricDistortion',
                                  to_gray_prob=0.1,
                                  distortion_prob=0.1, ),
                             dict(type='Weather',
                                  prob=0.05),
                             dict(type='RandomBlur',
                                  prob=0.05),
                             dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                             dict(type='ToTensor'),
                             dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                             ],
            weak_pipeline=[dict(type='BGR2RGB'),
                           dict(type='RandomResize',
                                prob=0.8,
                                ratio_range=(0.5, 0.99),
                                is_lidar=False),
                           dict(type='RandomCrop',
                                crop_size=(0, 0),  # crop_size will be overwriteen by data_basic configs
                                crop_type='rand',
                                ignore_label=-1,
                                padding=[0, 0, 0]),
                           dict(type='RandomEdgeMask',
                                mask_maxsize=50,
                                prob=0.2,
                                rgb_invalid=[0, 0, 0],
                                label_invalid=-1, ),
                           dict(type='RandomHorizontalFlip',
                                prob=0.4),
                           dict(type='ToTensor'),
                           dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                           ],
            # sample_size = 10,
            sample_ratio=1.0,
            sample_size=-1,
        )
    ))
