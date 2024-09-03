from ultralytics.engine.predictor import BasePredictor

import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import smart_inference_mode

# select_specified_device
import platform
from ultralytics.utils import LOGGER, RANK, __version__
from ultralytics.utils.checks import check_version
from ultralytics.utils.torch_utils import get_cpu_info
from ultralyticsp.utils import ops as custom_ops
TORCH_2_0 = check_version(torch.__version__, '2.0.0')

# stream_inference
import cv2
from pathlib import Path
from ultralytics.utils import LOGGER, colorstr


class CustomBasePredictor(BasePredictor):

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(model or self.args.model,
                                 
                                 # NOTE. modified by ryanwfu 2023/09/12
                                 device=self.select_specified_device(self.args.device, verbose=verbose),

                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()
    
    def select_specified_device(self, device='', batch=0, newline=False, verbose=True):
        s = f'Ultralytics YOLOv{__version__} 🚀 Python-{platform.python_version()} torch-{torch.__version__} '
        device = str(device).lower()
        for remove in 'cuda:', 'none', '(', ')', '[', ']', "'", ' ':
            device = device.replace(remove, '')  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
        cpu = device == 'cpu'
        mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
        if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
            devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
            n = len(devices)  # device count

            # NOTE. modified by ryanwfu 2023/09/12
            if n > 1:
                raise ValueError("Only support single specified gpu")
            
            if n > 1 and batch > 0 and batch % n != 0:  # check batch_size is divisible by device_count
                raise ValueError(f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                                f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}.")
            space = ' ' * (len(s) + 1)
            for i, d in enumerate(devices):
                p = torch.cuda.get_device_properties(i)
                s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
            
            # NOTE. modified by ryanwfu 2023/09/12
            arg = f'cuda:{devices[0]}'
            
        elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available() and TORCH_2_0:
            # Prefer MPS if available
            s += f'MPS ({get_cpu_info()})\n'
            arg = 'mps'
        else:  # revert to CPU
            s += f'CPU ({get_cpu_info()})\n'
            arg = 'cpu'

        if verbose and RANK == -1:
            LOGGER.info(s if newline else s.rstrip())
        return torch.device(arg)

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)

        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        # NOTE. modified by ryanwfu 2023/09/12
        self.seen, self.windows, self.batch, profilers = 0, [], None,\
              (custom_ops.Profile(device=self.device), custom_ops.Profile(device=self.device), custom_ops.Profile(device=self.device))

        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im0s, vid_cap, s = batch

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.inference(im, *args, **kwargs)

            # Postprocess
            with profilers[2]:
                self.results = self.postprocess(preds, im, im0s)
            self.run_callbacks('on_predict_postprocess_end')

            # Visualize, save, write results
            n = len(im0s)
            for i in range(n):
                self.seen += 1
                self.results[i].speed = {
                    'preprocess': profilers[0].dt * 1E3 / n,
                    'inference': profilers[1].dt * 1E3 / n,
                    'postprocess': profilers[2].dt * 1E3 / n}
                p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                p = Path(p)

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))
                if self.args.save or self.args.save_txt:
                    self.results[i].save_dir = self.save_dir.__str__()
                if self.args.show and self.plotted_img is not None:
                    self.show(p)
                if self.args.save and self.plotted_img is not None:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))

            self.run_callbacks('on_predict_batch_end')
            yield from self.results

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *im.shape[2:])}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')
    
    @smart_inference_mode()
    def custom_stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)

        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch = 0, [], None

        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im0s, vid_cap, s = batch

            # Preprocess
            im = self.preprocess(im0s)

            # Inference
            preds = self.inference(im, *args, **kwargs)

            # Postprocess
            self.results = self.custom_postprocess(preds, im, im0s)
            self.run_callbacks('on_predict_postprocess_end')

            self.run_callbacks('on_predict_batch_end')
            yield from self.results

        self.run_callbacks('on_predict_end')
    
    def __call__(self, source=None, model=None, stream=False, custom_inference=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        if not custom_inference:
            if stream:
                return self.stream_inference(source, model, *args, **kwargs)
            else:
                return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one
        else:
            if stream:
                return self.custom_stream_inference(source, model, *args, **kwargs)
            else:
                return list(self.custom_stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

