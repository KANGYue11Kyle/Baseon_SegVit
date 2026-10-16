checkpoint = './pretrained/vit_base_p16_384_20220308-96dfe169.pth'
backbone_norm_cfg = dict(type='LN', eps=1e-06, requires_grad=True)
img_size = 512
in_channels = 768
out_indices = [5, 7, 11]
model = dict(
    type='EncoderDecoder',
    pretrained='./pretrained/vit_base_p16_384_20220308-96dfe169.pth',
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
        interpolate_mode='bicubic',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segvit/vit_base_patch16_224.pth'
        )),
    decode_head=dict(
        type='ATMHead',
        img_size=512,
        in_channels=768,
        channels=768,
        num_classes=10,
        num_layers=3,
        num_heads=12,
        use_stages=3,
        embed_dims=384,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(
                type='ATMLoss', num_classes=10, dec_layers=4, loss_weight=1.0)
        ]),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
find_unused_parameters = True
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    momentum=0.9,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_embed=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
classes = [
    'background', 'tool1_manipulator', 'tool1_shaft', 'tool2_manipulator',
    'tool2_shaft', 'tool3_manipulator', 'tool3_shaft', 'tool4_manipulator',
    'tool4_shaft', 'uterus'
]
data_root = 'data/surgery/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(640, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
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
        flip=False,
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset',
        data_root='data/surgery/',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(640, 640), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(640, 640)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
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
                flip=False,
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
                flip=False,
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
seed = 0
deterministic = False
gpu_ids = [0]
work_dir = './work_dirs/segvit_vit-base_640x640_80k_surgery'
auto_resume = False
