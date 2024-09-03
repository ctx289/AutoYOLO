import os
import argparse
from tools.dictaction import DictAction
from tools.dictaction import ModelLoader, FileModelLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--model', help='path to model file, i.e. yolov8n.pt, Represents model methods and pre-trained models')
    parser.add_argument('--data',
                        help='path to data file, i.e. ./ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop.yaml')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--project', help='project name')
    parser.add_argument('--name', help='experiment name, results saved to \'project/name\' directory')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument('--cuda_visible_devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    if args.work_dir and args.project:
        raise ValueError(
            '--work-dir and --project cannot be both '
            'specified')
    return args


def train(args, use_wrapper=True, model_loader: ModelLoader = None):
    """
    当指定了model_loader, args.model 和 args.config 中 model 字段表示模型方法, 如 "yolov8n.pt", 预训练模型从model_loader中获取;
    当未指定model_loader, args.model 和 args.config 中 model 字段即表示模型方法, 又表示预训练模型路径;
    """

    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    # Normal Import
    from ultralytics.cfg import check_dict_alignment
    from ultralyticsp.custom import yaml_load, set_random_seed
    from ultralyticsp.custom.callbacks import logger_train_start, logger_train_epoch_start, logger_train_end
    from ultralyticsp.models.yolo.detect import COCOYOLODetectionTrainer
    from ultralyticsp.models.rtdetr import COCORTDETRTrainer
    from ultralyticsp.models import CustomYOLO, CustomRTDETR

    # for single gpu training; use torch.distributed.run;
    # import torch
    # torch.distributed.init_process_group(backend='nccl')

    cfg = yaml_load(args.config)

    if args.model is not None:
        cfg['model'] = args.model

    # load binary stream
    if model_loader is None:
        model_loader = FileModelLoader(cfg['model'])
    model_Bytes = model_loader.load()

    if args.data is not None:
        cfg['data'] = args.data
    if args.work_dir is not None:
        cfg['project'] = os.path.dirname(args.work_dir)
        cfg['name'] = os.path.basename(args.work_dir)
    else:
        if args.project is not None:
            cfg['project'] = args.project
        if args.name is not None:
            cfg['name'] = args.name
    if args.no_validate:
        cfg['val'] = False

    # cfg options
    if args.cfg_options is not None:
        cfg_options = args.cfg_options
        check_dict_alignment(cfg, cfg_options)
        cfg = {**cfg, **cfg_options}  # merge cfg and overrides dicts (prefer overrides)

    if use_wrapper:
        # init
        if 'rtdetr' in os.path.basename(cfg['model']):
            model = CustomRTDETR(model_Bytes)
            # NOTE: F.grid_sample which is in rt-detr does not support deterministic=True
            # NOTE: amp training causes nan outputs and end with error while doing bipartite graph matching
        elif 'yolo' in os.path.basename(cfg['model']):
            model = CustomYOLO(model_Bytes)
        else:
            raise NotImplementedError
        model_Bytes.close()
        # Display model information (optional)
        model.info()
        # callbacks
        model.add_callback('on_train_start', logger_train_start)
        model.add_callback('on_train_epoch_start', logger_train_epoch_start)
        model.add_callback('on_train_end', logger_train_end)
        # train
        model.train(**cfg)
    else:
        """
        # === deprecated ===
        # 不推荐使用
        # Pretrained models in binary streaming or encrypted formats are not supported
        """
        # init
        if 'rtdetr' in os.path.basename(cfg['model']):
            # NOTE: F.grid_sample which is in rt-detr does not support deterministic=True
            # NOTE: amp training causes nan outputs and end with error while doing bipartite graph matching
            trainer = COCORTDETRTrainer(overrides=cfg)
        elif 'yolo' in os.path.basename(cfg['model']):
            trainer = COCOYOLODetectionTrainer(overrides=cfg)
        else:
            raise NotImplementedError
        # callbacks
        trainer.add_callback('on_train_start', logger_train_start)
        trainer.add_callback('on_train_epoch_start', logger_train_epoch_start)
        trainer.add_callback('on_train_end', logger_train_end)
        # train
        trainer.train()


if __name__ == '__main__':
    args = parse_args()
    train(args)
