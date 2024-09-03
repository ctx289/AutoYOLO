# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model

# for decode encrypt file
import _io
import torch
from pathlib import Path
from typing import Union
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, callbacks 
from ultralytics.utils.downloads import GITHUB_ASSET_STEMS
from ultralytics.nn.tasks import guess_model_task, nn
from ultralytics.nn.modules import Detect, Segment

# for custom inference
import sys
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.cfg import get_cfg
from ultralytics.utils import ROOT, is_git_dir, LOGGER

# for no rect, It doesn't matter (since 2023/09/19) whether rect True or False, cause this bug has been fixed.
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.checks import check_imgsz


class CustomModel(Model):
    """
    Custom model for decode encrypt file
    """

    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        """
        Initializes the YOLO model.

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
            task (Any, optional): Task type for the YOLO model. Defaults to None.
        """
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        self.task = task  # task type
        if isinstance(model, _io.BytesIO) or isinstance(model, _io.BufferedReader):
            self._load_BytesIO(model, task)
        else:
            model = str(model).strip()  # strip spaces

            # Check if Ultralytics HUB model from https://hub.ultralytics.com
            if self.is_hub_model(model):
                from ultralytics.hub.session import HUBTrainingSession
                self.session = HUBTrainingSession(model)
                model = self.session.model_file

            # Load or create new YOLO model
            suffix = Path(model).suffix
            if not suffix and Path(model).stem in GITHUB_ASSET_STEMS:
                model, suffix = Path(model).with_suffix('.pt'), '.pt'  # add suffix, i.e. yolov8n -> yolov8n.pt
            if suffix in ('.yaml', '.yml'):
                self._new(model, task)
            else:
                self._load(model, task)  
    
    def _load_BytesIO(self, weights: _io.BytesIO, task=None):
        assert isinstance(weights, _io.BytesIO) or isinstance(weights, _io.BufferedReader)
        self.model, self.ckpt = self.attempt_load_one_weight_BytesIO(weights)
        self.task = self.model.args['task']
        self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
        self.ckpt_path = self.model.pt_path
        self.overrides['model'] = weights
        self.overrides['task'] = self.task
    
    def attempt_load_one_weight_BytesIO(self, weight, device=None, inplace=True, fuse=False):
        """Loads a single model weights."""
        assert isinstance(weight, _io.BytesIO) or isinstance(weight, _io.BufferedReader)
        ckpt, weight = torch.load(weight, map_location='cpu'), None
        args = {**DEFAULT_CFG_DICT, **(ckpt.get('train_args', {}))}  # combine model and default args, preferring model args
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
        model.pt_path = weight  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.])

        model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()  # model in eval mode

        # Module updates
        for m in model.modules():
            t = type(m)
            if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
                m.inplace = inplace
            elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        # Return model and ckpt
        return model, ckpt
    
    @smart_inference_mode()
    def predict(self, source=None, stream=False, predictor=None, custom_inference=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            predictor (BasePredictor): Customized predictor.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))
        # Check prompts for SAM/FastSAM
        prompts = kwargs.pop('prompts', None)
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        if not is_cli:
            overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        if not self.predictor:
            self.task = overrides.get('task') or self.task
            predictor = predictor or self.smart_load('predictor')
            self.predictor = predictor(overrides=overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
            if 'project' in overrides or 'name' in overrides:
                self.predictor.save_dir = self.predictor.get_save_dir()

        # NOTE. commented by ryanwfu 2023/09/25
        # # Set prompts for SAM/FastSAM
        # if len and hasattr(self.predictor, 'set_prompts'):
        #     self.predictor.set_prompts(prompts)

        # NOTE. modified by ryanwfu 2023/09/18
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream, custom_inference=custom_inference)
    
    @smart_inference_mode()
    def val(self, data=None, validator=None, **kwargs):
        """
        Validate a model on a given dataset.

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            validator (BaseValidator): Customized validator.
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()

        # NOTE. modified by ryanwfu 2023/09/19
        overrides['rect'] = False  

        overrides.update(kwargs)
        overrides['mode'] = 'val'
        if overrides.get('imgsz') is None:
            overrides['imgsz'] = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        validator = validator or self.smart_load('validator')
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = validator(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics

        return validator.metrics