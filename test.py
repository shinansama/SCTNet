import os
import os.path as osp
import sys
import time

current_directory = os.getcwd()
sys.path.append(current_directory)

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (load_checkpoint, wrap_fp16_model)

from mmseg.apis import single_gpu_test, train_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_dp, get_device, setup_multi_processes


# config_path = r"configs\sctnet\ADE20K\sctnet-b_8x4_160k_ade.py"
# checkpoint = r"configs\sctnet\ADE20K\pretrain\SCTNet-B-ADE20K.pth"

# config_path = r"configs\sctnet\COCO-Stuff-10K\sctnet_b_4x4_160k.py"
# checkpoint = r"configs\sctnet\COCO-Stuff-10K\pretrain\SCTNet-B_COCO-Stuff-10K.pth"

config_path = r"configs\sctnet\pets\sctnet-b_8x4_160k_pets.py"
checkpoint = r"configs\sctnet\pets\pretrain\SCTNet-B-ADE20K.pth"


cfg = mmcv.Config.fromfile(config_path)
# set multi-process settings
setup_multi_processes(cfg)

# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

cfg.gpu_ids = [0]

# mkdir work_dir
work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_path))[0])
mmcv.mkdir_or_exist(osp.abspath(work_dir))
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
json_file = osp.join(work_dir, f'eval_single_scale_{timestamp}.json')
output_img_dir = osp.join(work_dir, f'eval_single_scale_{timestamp}')

# build the dataloader
# TODO: support multiple images per gpu (only minor changes are needed)
dataset = build_dataset(cfg.data.test)

# The default loader config
loader_cfg = dict(
    # cfg.gpus will be ignored if distributed
    num_gpus=len(cfg.gpu_ids),
    dist=False,
    shuffle=False)
# The overall dataloader settings
loader_cfg.update({
    k: v
    for k, v in cfg.data.items() if k not in [
        'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
        'test_dataloader'
    ]
})
test_loader_cfg = {
    **loader_cfg,
    'samples_per_gpu': 1,
    'shuffle': False,  # Not shuffle by default
    **cfg.data.get('test_dataloader', {})
}
# build the dataloader
data_loader = build_dataloader(dataset, **test_loader_cfg)


# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    print('"CLASSES" not found in meta, use dataset.CLASSES instead')
    model.CLASSES = dataset.CLASSES
if 'PALETTE' in checkpoint.get('meta', {}):
    model.PALETTE = checkpoint['meta']['PALETTE']
else:
    print('"PALETTE" not found in meta, use dataset.PALETTE instead')
    model.PALETTE = dataset.PALETTE

# clean gpu memory when starting a new evaluation.
torch.cuda.empty_cache()

cfg.device = get_device()

model = revert_sync_batchnorm(model)
model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

eval_kwargs = {}

if __name__ == '__main__':
    results = train_segmentor(
        model,
        data_loader,
        cfg,
        distributed=False,
        validate=True,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        meta=None)
    # results = single_gpu_test(
    #     model,
    #     data_loader,
    #     show=False,
    #     out_dir=output_img_dir,
    #     efficient_test=False,
    #     opacity=0.5,
    #     pre_eval=False,
    #     format_only=False,
    #     format_args=eval_kwargs)
    #
    # eval_kwargs.update(metric="mIoU")
    # metric = dataset.evaluate(results, **eval_kwargs)
    # metric_dict = dict(config=config_path, metric=metric)
    # mmcv.dump(metric_dict, json_file, indent=4)
