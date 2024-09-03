""" Define the base segmentor """
from abc import abstractclassmethod


class BaseSegmentor(object):
    def __init__(self,
                 config,
                 ckpt,
                 input_name='images',
                 output_name='preds',
                 gpu_id=0,
                 verbose=False, 
                 **kwargs):
        self.config = config
        self.ckpt = ckpt
        self.input_name = input_name
        self.output_name = output_name
        self.verbose = verbose
        self.gpu_id = gpu_id
        self.model = self.init(self.config, self.ckpt, self.gpu_id, **kwargs)

    @abstractclassmethod
    def init(self, config, ckpt, device, **kwargs):
        pass

    @abstractclassmethod
    def predict(self, feed_dict):
        pass

    def __call__(self, feed_dict, **kwargs):
        """ inference with feed data

        Args:
            feed_dict (dict)
        """
        return self.predict(feed_dict)
