import argparse
import importlib
import mmcv
from mmdet.utils import collect_env
from mmdet.apis import set_random_seed
from mmdet.models import build_detector
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmcv.parallel import MMDataParallel

from projects.mmdet3d_plugin.datasets import custom_build_dataset
from mmdet.core import EvalHook


# 参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--work-dir")
parser.add_argument("--resume-from")
parser.add_argument("--no-validate")
parser.add_argument("--gpu-ids")
parser.add_argument("--seed", type = int , default = 0, help = "random seed")
parser.add_argument("--deterministic", action = "store_true")
args = parser.parse_args()

cfg = Config.fromfile(args.config)

# 导入plugin包
plugin_dir = cfg.plugin_dir
_module_dir = os.path.dirname(plugin_dir)
_module_dir = _module_dir.split("/")
_module_path = _module_dir[0]
for m in _module_dir[1:]:
    _module_path = _module_path + "." + m

import sys
sys.path.append('/home/bo.yang5/other/Sparse4D-full')
plg_lib = importlib.import_module()

cfg.work_dir = osp.join("./work_dir", osp.splitext(osp.basename(args.config))[0])
cfg.gpu_ids = args.gpu_ids

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

set_random_seed(args.seed, deterministic = args.deterministic)

#============================== train ==============================
#数据集
datasets = [build_dataset(cfg.data.train)]
data_loaders = [
    build_dataloader(
        ds,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        dist = distributed,
        seed = cfg.seed,
        nonshuffler_sampler = dict(type = "DistributedSampler"),
        runner_type = runner_type,
    )
    for ds in datasets
]

#构建模型，模型参数初始化
model = build_detector(cfg.model, train_cfg = cfg.get("train_cfg"), test_cfg = cfg.get("test_cfg"))
model.init_weights()
model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids = cfg.gpu_ids)

#优化器
optimizer = build_optimizer(model, cfg.optimizer)

#runner
runner = build_runner(
    cfg.runner,
    default_args = dict(
        model = model,
        optimizer = optimizer,
        work_dir = cfg.work_dir,
        logger = logger,
        meta = meta,
    ),
)

runner.register_training_hooks(
    cfg.lr_config,
    optimizer_config,
    cfg.checkpoint_config,
    cfg.log_config,
    cfg.get("momentum_config", None),
)

#============================== val ==============================
#数据集
val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode = True))
val_dataloader = build_dataloader(
    val_dataset,
    samples_per_gpu = val_samples_per_gpu,
    workers_per_gpu = cfg.data.workers_per_gpu,
    dist = distributed,
    shuffle = False,
    nonshuffler_sampler = dict(type="DistributedSampler"),
)

#runner
eval_cfg = cfg.get("evaluation", {})
eval_hook = EvalHook
runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
