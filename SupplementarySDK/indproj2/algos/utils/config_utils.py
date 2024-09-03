"""Utils to deal with project config
"""


def recursive_change_keys(origin_dict, change_dict, change_key=None):
    """recursive_change_keys
        Recursively change keys or add keys in origin_dict by change_dict
        Notice:
        This recursion only stops at base level, when value is not list and dict

    Args:
        origin_dict (dict): origin dict to be replace
        replace_dict (dict): new dict to replace or add keys in origin_dict
        change_key (str/int): for recursion
    """
    if isinstance(change_dict, dict):
        # if key not in origin dict, just put current dict to origin dict
        if change_key is not None and isinstance(
                origin_dict, dict) and change_key not in origin_dict.keys():
            origin_dict[change_key] = change_dict
            return

        next_origin = origin_dict if change_key is None else origin_dict[
            change_key]

        for ckey, cvalue in change_dict.items():
            recursive_change_keys(next_origin, cvalue, ckey)

    elif isinstance(change_dict, list):
        next_origin = origin_dict if change_key is None else origin_dict[
            change_key]
        for ckey, cvalue in enumerate(change_dict):
            recursive_change_keys(next_origin, cvalue, ckey)

    else:
        origin_dict[change_key] = change_dict
