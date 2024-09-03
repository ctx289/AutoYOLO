""" Base Pipeline """
import logging
import time

from ..builder import PIPELINE, build_module


@PIPELINE.register_module()
class BasePipeline(object):
    def __init__(self, modules, gpu_id=0, verbose=False):
        if not isinstance(modules, list):
            raise ValueError("Pipeline only list as modules")
        self._module_cfg = modules
        self._gpu_id = gpu_id

        self._modules = dict()
        for idx in range(len(self._module_cfg)):
            start_time = time.time()
            name = "{}_{}".format(idx, self._module_cfg[idx]['type'])
            module = build_module(self._module_cfg[idx],
                                  default_args={
                                      "gpu_id": gpu_id,
                                      "verbose": verbose
                                  })
            self._modules[name] = module
            if verbose:
                logging.info(
                    "Successfully built module {} after {}s".format(
                        name, round(time.time() - start_time, 5)))
        self._sorted_module_names = sorted(self._modules.keys())

    @property
    def sorted_module_names(self):
        return self._sorted_module_names

    @property
    def modules(self):
        return self._modules

    @property
    def gpu_id(self):
        return self._gpu_id

    def with_module(self, check_type):
        """check whether cls has module

        Args:
            check_type ([type]): [description]

        Returns:
            [type]: [description]
        """
        for module_name in self._sorted_module_names:
            if isinstance(self.modules[module_name], check_type):
                return self.modules[module_name]
        return False

    def __getitem__(self, module_name):
        return self.modules[module_name]

    def __len__(self):
        return len(self.modules)

    def reset(self):
        for key in self._modules.keys():
            if hasattr(self._modules[key], "reset"):
                self._modules[key].reset()

    def __call__(self, feed_dict):
        for name in self._sorted_module_names:
            feed_dict = self._modules[name](feed_dict)
        return feed_dict

    def time_call(self, feed_dict):
        time_dict = dict()
        zero_start = time.time()
        for name in self._sorted_module_names:
            start_time = time.time()
            feed_dict = self._modules[name](feed_dict)
            time_dict[name] = time.time() - start_time
        time_dict['whole_pipeline'] = time.time() - zero_start
        return feed_dict, time_dict
