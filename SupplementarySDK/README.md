## 工作流程

1. 平台层面获得SDK目录，读取`sdk_entry.yaml`
2. 将SDK目录加入PYTHONPATH环境变量，并加载所声明的模块，并依据类名创建SDK实例
3. 调用SDK的`initialize`方法初始化SDK
4. 调用`inference`执行推理

## 实现概述

SDK需要继承`DetPredictor`并实现以下方法：
- `initialize`: SDK初始化
- `all_tags`: 返回类别列表，顺序即为底层模型输出中对应类别的序号
- `inference`: 推理接口
- `_set_device`: 切换cpu/cuda接口

```
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
```