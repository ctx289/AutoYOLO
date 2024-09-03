import yaml
from pathlib import Path

# NOTE. modified by ryanwfu. 2023/10/10; Force the file to be in utf-8 format and ensure that the path is read normally
def custom_yaml_save(file='data.yaml', data=None):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v)

    # Dump data to file in YAML format
    with open(file, 'w', encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, encoding='utf-8', allow_unicode=True)