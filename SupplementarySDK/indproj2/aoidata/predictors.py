from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Union, Tuple, List
import numpy as np


class Predictor(metaclass=ABCMeta):
    """
    SDK interface for algorithm inferencing
    """

    SUPPORTED_DEVICES = ('cpu', 'cuda')

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """
        reserved for future use
        """
        ...

    @abstractmethod
    def initialize(self, *, model_path: str, **kwargs: Dict[str, Any]
                   ) -> None:
        """
        initialize the predictor SDK
        model_path: a direcotry or archive path to the model so the SDK can
            initialize from (configs, weights, etc.)
        kwargs: reserved for future use

        raise when there are errors
        """
        raise NotImplementedError

    @abstractmethod
    def inference(self, *args, **kwargs) -> Any:
        """
        a generic interface, types of arguments and return values will be
        specialized in derived classes
        """
        raise NotImplementedError

    def set_device(self, device_type: str, device_id: int = 0) -> None:
        """
        interface for switching running device
        """
        assert device_type in Predictor.SUPPORTED_DEVICES
        assert isinstance(device_id, int) and device_id >= 0
        self._set_device(device_type, device_id)

    @abstractmethod
    def _set_device(self, device_type: str, device_id: int = 0) -> None:
        """
        abstract method for derived classes to define how they switch devices
        """
        raise NotImplementedError


@dataclass
class DetBox:
    x: float
    y: float
    w: float
    h: float
    score: float
    tag: str
    attributes: Dict[str, Union[bool, str, float, Tuple[float, ...]]]

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def as_dict(self) -> Dict[str, Union[float, str]]:
        from dataclasses import asdict
        return asdict(self)

    @property
    def is_ng(self) -> bool:
        # defaults to all results to be NG
        return self.attributes.get('is_ng', True)

    @property
    def x1y1x2y2(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    # more properties can be added when needed in the future
    # NOTE: should be a default value for compability


class DetPredictor(Predictor):
    """
    base class for SDK for normal detection algorithms
    """

    @abstractmethod
    def inference(self, image: np.ndarray) -> List[DetBox]:
        """
        A normal detection algorithm should take a image as input
        and produce a list of boxes with score and maybe other attributes
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def all_tags(self) -> List[str]:
        """
        A normal detection algorithm should have a fixed set of tags
        (or categories) that all detection results belong to
        """
        raise NotImplementedError

