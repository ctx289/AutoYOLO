import os
import re
import shutil
import socket
import sys
import tempfile
from pathlib import Path

from ultralytics.utils.torch_utils import TORCH_1_9
from ultralytics.utils.dist import generate_ddp_file
from ultralytics.utils.dist import find_free_network_port

def custom_generate_ddp_command(world_size, trainer):
    """Generates and returns command for distributed training."""
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = str(Path(sys.argv[0]).resolve())
    safe_pattern = re.compile(r'^[a-zA-Z0-9_. /\\-]{1,128}$')  # allowed characters and maximum of 100 characters
    if not (safe_pattern.match(file) and Path(file).exists() and file.endswith('.py')):  # using CLI
        file = generate_ddp_file(trainer)
    dist_cmd = 'torch.distributed.run' if TORCH_1_9 else 'torch.distributed.launch'
    port = find_free_network_port()
    cmd = [sys.executable, '-m', dist_cmd, '--nproc_per_node', f'{world_size}', '--master_port', f'{port}', file] + sys.argv[1:]
    return cmd, file