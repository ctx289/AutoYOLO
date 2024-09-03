from copy import copy
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralyticsp.data import build_coco_yolo_dataset
from ultralyticsp.models.yolo.detect.val import COCOYOLODetectionValidator

# modify _setup_train for check_amp
import cv2
import math
import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from ultralytics.utils import (LOGGER, RANK, __version__, callbacks)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_imgsz
from ultralytics.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, one_cycle)
from ultralytics.utils import (ROOT, colorstr, ONLINE)

# for multi-gpu training
from ultralyticsp.engine.trainer import CustomBaseTrainer


class COCOYOLODetectionTrainer(CustomBaseTrainer, DetectionTrainer):
    
    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        
        # NOTE. modified by ryanwfu 2023/09/19 : rect=mode == 'val' -> rect=False
        return build_coco_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs) # no rect
    
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return COCOYOLODetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    
    # NOTE. for multi-gpu train in windows use gloo backend
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        from ultralyticsp.utils.torch_utils import custom_torch_distributed_zero_first
        from ultralytics.data import build_dataloader
        from torch import distributed as dist
        
        assert mode in ['train', 'val']
        with custom_torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        
        # NOTE. (ryanwfu) 对dataloader, train data有workers个, val data 有  workers * 2 个，因为val data 的 batchsize 扩大了两倍
        workers = self.args.workers if mode == 'train' else self.args.workers * 2

        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    # In order to solve the problem that the pre-trained model will be re-downloaded 
    # to the working directory when checking amp, the _setup_train function needs to be modified. 
    # The code of the function _setup_train is basically copied from (ultralytics==8.0.156)
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py#L210   
    def _setup_train(self, world_size):
        """
        Builds dataloaders and optimizer on correct rank process.
        """

        # Model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = self.args.freeze if isinstance(
            self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']  # always freeze these layers
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                            'See ultralytics.engine.trainer for customization of frozen layers.')
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            # NOTE. modify by ryanwfu 2023/09/01
            self.amp = torch.tensor(self.check_amp(self.model, self.args.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            # NOTE. modified by ryanwfu 2023/09/22
            dist.broadcast(self.amp.to(torch.uint8), src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)

        # Batch size
        if self.batch_size == -1:
            if RANK == -1:  # single-GPU only, estimate best batch size
                self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)
            else:
                SyntaxError('batch=-1 to use AutoBatch is only available in Single-GPU training. '
                            'Please pass a valid batch size value for Multi-GPU DDP training, i.e. batch=16')

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        
        # NOTE. (ryanwfu) Attention here! During the training process, testset is used for validation. And the batch size is doubled
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay,
                                              iterations=iterations)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')
    
    def check_amp(self, model, pretrained_model_path):
        """
        This function checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLOv8 model.
        If the checks fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP
        results, so AMP will be disabled during training.

        Args:
            model (nn.Module): A YOLOv8 model instance.

        Example:
            ```python
            from ultralytics import YOLO
            from ultralytics.utils.checks import check_amp

            model = YOLO('yolov8n.pt').model.cuda()
            check_amp(model)
            ```

        Returns:
            (bool): Returns True if the AMP functionality works correctly with YOLOv8 model, else False.
        """
        device = next(model.parameters()).device  # get model device
        if device.type in ('cpu', 'mps'):
            return False  # AMP only used on CUDA devices

        def amp_allclose(m, im):
            """All close FP32 vs AMP results."""
            a = m(im, device=device, verbose=False)[0].boxes.data  # FP32 inference
            with torch.cuda.amp.autocast(True):
                b = m(im, device=device, verbose=False)[0].boxes.data  # AMP inference
            del m
            return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # close to 0.5 absolute tolerance

        f = ROOT / 'assets/bus.jpg'  # image to check
        
        # NOTE. modified by ryanwfu. 2023/09/28, compatible with Chinese paths in windows
        im = cv2.imdecode(np.fromfile(str(f).encode('utf-8'), dtype=np.uint8), cv2.IMREAD_COLOR) if f.exists() else np.ones((640, 640, 3))
        # im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if ONLINE else np.ones((640, 640, 3))

        prefix = colorstr('AMP: ')
        LOGGER.info(f'{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8n...')
        warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."
        try:
            # NOTE. modify by ryanwfu 2023/09/12
            import os
            from ultralytics import YOLO
            check_amp_model_path = pretrained_model_path
            if os.path.exists(pretrained_model_path) and 'yolo' in os.path.basename(pretrained_model_path)\
                and pretrained_model_path.endswith('.pt'):
                assert amp_allclose(YOLO(check_amp_model_path), im)
                LOGGER.info(f'{prefix}checks from pt file, checks passed ✅')
            # NOTE. modify by ryanwfu 2023/09/20
            else:
                assert amp_allclose(YOLO("yolov8n.yaml"), im)
                LOGGER.info(f'{prefix}checks from yaml file, checks passed ✅')
        except ConnectionError:
            LOGGER.warning(f'{prefix}checks skipped ⚠️, offline and unable to download YOLOv8n. {warning_msg}')
        except (AttributeError, ModuleNotFoundError):
            LOGGER.warning(
                f'{prefix}checks skipped ⚠️. Unable to load YOLOv8n due to possible Ultralytics package modifications. {warning_msg}'
            )
        except AssertionError:
            LOGGER.warning(f'{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to '
                        f'NaN losses or zero-mAP results, so AMP will be disabled during training.')
            return False
        return True
    
    # def _setup_ddp(self, world_size):
    #     """Initializes and sets the DistributedDataParallel parameters for training."""
    #     import os
    #     from datetime import timedelta
    #     torch.cuda.set_device(RANK)
    #     self.device = torch.device('cuda', RANK)
    #     BACKEND = 'nccl' if not dist.is_nccl_available() else 'gloo'
    #     LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}, BACKEND {BACKEND}')
    #     os.environ['NCCL_BLOCKING_WAIT'] = '1'  # set to enforce timeout
    #     dist.init_process_group(
    #         BACKEND,
    #         timeout=timedelta(seconds=10800),  # 3 hours
    #         rank=RANK,
    #         world_size=world_size)