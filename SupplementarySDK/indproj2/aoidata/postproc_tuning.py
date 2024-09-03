from typing import (
    Generic, TypeVar, Sequence as Seq,
    Type, List, Dict, Union
)
from numbers import Real, Integral
from abc import ABCMeta, abstractmethod

TunableTypes = (str, Real, Integral, bool)
VT = TypeVar('VT', *TunableTypes, covariant=True)


class TunableParam(Generic[VT]):
    """
    Record of a scalar parameter that can cooperate with tuning algorithms
    """
    def __init__(self, tp: Type[VT], init_val: VT,
                 avail_vals: Seq[VT]) -> None:
        assert any(issubclass(tp, acc_tp) for acc_tp in TunableTypes)
        assert isinstance(init_val, tp)
        avails = [val for val in set(avail_vals)]
        assert all(isinstance(val, tp) for val in avails)
        if init_val not in avails:
            avails = [init_val] + list(avail_vals)
        self.__type = tp
        self.__curr = init_val
        self.__opts = avails

    @property
    def current(self) -> VT:
        return self.__curr

    @property
    def val_type(self) -> Type[VT]:
        return self.__type

    @property
    def options(self) -> List[VT]:
        return self.__opts.copy()

    def set_to(self, val: VT):
        if val not in self.__opts:
            raise ValueError(val)
        self.__curr = val


class Options(TunableParam[VT]):
    """
    Tunable parameter with a pre-defined value set
    """
    def __init__(self, tp: Type[VT], avail_vals: List[VT]):
        super().__init__(tp, avail_vals[0], avail_vals)


class Sequence(TunableParam[Integral]):
    """
    Tunable parameter, whose options are sequence of integral values
    """
    def __init__(self, start: Integral, stop: Integral,
                 step: Integral = 1, end_included: bool = False) -> None:
        assert start < stop or (end_included and start <= stop)
        assert step != 0
        vals = list(range(start, stop, step))
        if vals[-1] + step <= stop and end_included:
            vals.append(stop)
        super().__init__(Integral, start, vals)


class Range(TunableParam[Real]):
    """
    Tunable parameter, whose options are real values sampled from given interval
    """
    def __init__(self, start: Real, stop: Real, num_samples: int = 2) -> None:
        import numpy as np
        vals = np.linspace(start, stop, num_samples, endpoint=True)
        super().__init__(Real, start, [float(val) for val in vals])


class Switch(TunableParam[bool]):
    """
    Tunable parameter which can be true or false
    """
    def __init__(self, initial_state: bool) -> None:
        super().__init__(bool, initial_state, [True, False])


class Constant(TunableParam[VT]):

    def __init__(self, tp: Type[VT], val: VT) -> None:
        super().__init__(tp, val, [val])


class OneOf(Options[VT]):

    def __init__(self, *opts: TunableParam[VT]):
        assert any(all(issubclass(opt2.val_type, opt.val_type)
                       for opt2 in opts)
                   for opt in opts)
        base_tp = [opt.val_type for opt in opts if all(
            issubclass(opt2.val_type, opt.val_type) for opt2 in opts
        )]
        vals = sum((opt.options for opt in opts), [])
        super().__init__(base_tp, vals)


SDKResultType = TypeVar("SDKResultType", covariant=True)
class PostProcTuningMixin(metaclass=ABCMeta):

    @abstractmethod
    def PPT_get_all_parameter_group_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def PPT_get_parameter_group(self, key: str) -> Dict[str, TunableParam]:
        raise NotImplementedError

    @abstractmethod
    def PPT_reset_to_identity(self):
        raise NotImplementedError

    def PPT_param_dict(self) -> Dict[
        str, Dict[str, Union[str, Real, Integral, bool]]
    ]:
        return {gkey: {key: param.current for key, param in
                       self.PPT_get_parameter_group(gkey).items()}
                for gkey in self.PPT_get_all_parameter_group_names()}

    def PPT_load_param_dict(self, vals: Dict[
        str, Dict[str, Union[str, Real, Integral, bool]]
    ], strict: bool = True):
        for gkey in self.PPT_get_all_parameter_group_names():
            if gkey not in vals:
                if strict:
                    raise ValueError(f'group {gkey} not provided')
                else:
                    continue
            param_group = self.PPT_get_parameter_group(gkey)
            group_vals = vals[gkey]
            for key, param in param_group.items():
                if key not in group_vals:
                    if strict:
                        raise ValueError(f'{key} of group {gkey} not provided')
                    else:
                        continue
                param.set_to(group_vals[key])

    @abstractmethod
    def do_post_proc(self, origin: SDKResultType) -> SDKResultType:
        raise NotImplementedError


if __name__ == '__main__':
    opts = Options(str, ['x', 'y', 'z'])
    opts2 = Options(Real, [1, 2, 3])
    a = Sequence(5, 10, 2)
    a.current
    b = Range(5, 7, 10)
    b.current
    c = Switch(False)
    c.current
    const = Constant(str, 'zzz')
    d = OneOf(opts, Constant(str, 'v'))
    d.current

    tune = PostProcTuningMixin()

    # for group in tune.PPT_get_all_parameter_group_names():
    #     params = tune.PPT_get_parameter_group(group)
    #     searcher = GridSearch(params, max_trials=5000)  # or other search algo
    #     to_maximize = SomeTarget(original_results, **target_kwargs)
    #     for exp_id, vals in searcher:  # TODO: how to parallel
    #         score = to_maximize.evaluate(tune.using(group, vals))
    #         searcher.record(exp_id, score)
    #     tune.using(group, searcher.report())
