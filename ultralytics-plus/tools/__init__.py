__version__ = "2.0.0"


from .train import train
from .val import val, val_format_only
from .dictaction import DictAction, ModelLoader, FileModelLoader
from .auto_resolution import AutoResolution

__all__ = ["train", "val", "val_format_only", 'DictAction', 'ModelLoader', 'FileModelLoader', 'AutoResolution']