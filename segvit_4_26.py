classes = [
    'background', 'tool1_manipulator', 'tool1_shaft', 'tool2_manipulator',
    'tool2_shaft', 'tool3_manipulator', 'tool3_shaft', 'tool4_manipulator',
    'tool4_shaft', 'uterus'
]
checkpoint = 'pretrained/vit_base_patch16_224.pth'
img_size = 640
in_channels = 768
out_indices = [5, 7, 11]
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/vit_base_patch16_224.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(640, 640),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        out_indices=[5, 7, 11],
        final_norm=False,
        norm_cfg=dict(type='LN', eps=1e-06, requires_grad=True),
        with_cls_token=False,
        interpolate_mode='bicubic'),
    decode_head=dict(
        type='ATMHead',
        img_size=640,
        in_channels=768,
        channels=768,
        num_classes=10,
        num_layers=3,
        num_heads=12,
        use_stages=3,
        embed_dims=384,
        ignore_index=255,
        loss_decode=dict(
            type='ATMLoss', num_classes=10, dec_layers=3, loss_weight=1.0)),
    test_cfg=dict(mode='whole'))
data_root = 'data/surgery/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        img_scale=(640, 640),
        ratio_range=(0.75, 1.25),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640), cat_max_ratio=0.9),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomRotate', prob=0.2, degree=10),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(640, 640), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        img_ratios=[0.75, 1.0, 1.25],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CustomDataset',
        data_root='data/surgery/',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='Resize',
                img_scale=(640, 640),
                ratio_range=(0.75, 1.25),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(640, 640), cat_max_ratio=0.9),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='RandomRotate', prob=0.2, degree=10),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(640, 640), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        classes=[
            'background', 'tool1_manipulator', 'tool1_shaft',
            'tool2_manipulator', 'tool2_shaft', 'tool3_manipulator',
            'tool3_shaft', 'tool4_manipulator', 'tool4_shaft', 'uterus'
        ]),
    val=dict(
        type='CustomDataset',
        data_root='data/surgery/',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                img_ratios=[0.75, 1.0, 1.25],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=[
            'background', 'tool1_manipulator', 'tool1_shaft',
            'tool2_manipulator', 'tool2_shaft', 'tool3_manipulator',
            'tool3_shaft', 'tool4_manipulator', 'tool4_shaft', 'uterus'
        ]),
    test=dict(
        type='CustomDataset',
        data_root='data/surgery/',
        img_dir='images/testing',
        ann_dir='annotations/testing',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                img_ratios=[0.75, 1.0, 1.25],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=[
            'background', 'tool1_manipulator', 'tool1_shaft',
            'tool2_manipulator', 'tool2_shaft', 'tool3_manipulator',
            'tool3_shaft', 'tool4_manipulator', 'tool4_shaft', 'uterus'
        ]))
class_weight = [0.5, 8.0, 5.0, 1.0, 1.0, 8.0, 1.5, 10.0, 2.0, 1.5]
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_embed=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            ln=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
cudnn_benchmark = True
find_unused_parameters = True
seed = 42
deterministic = False
gpu_ids = [0]
work_dir = './work_dirs/segvit_vit-base_surgery_continue_4_27'
log_level = 'INFO'
workflow = [('train', 1)]
device = 'cuda'
resume_from = None
auto_resume = False
load_from = r'F:\download\SegVit-master\work_dirs\segvit_vit-base_surgery_continue_4_26\iter_4000.pth'