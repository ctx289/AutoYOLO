#!/usr/bin/env python3
# coding:utf-8
# Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
# ==============================================================================
"""This file is the utilities for folder or path check, clear or create."""
from __future__ import absolute_import, division, print_function

import os
import shutil


def check_exist_create(folder_dir):
    """Check whether folder exists and create or not.

    Args:
        folder_dir (str): folder_dir path.
    """

    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def delete_create_folder(folder_dir):
    """Delete folder if exists and create.

    Args:
        folder_dir (str): folder_dir path.
    """

    if os.path.exists(folder_dir):
        shutil.rmtree(folder_dir)
        os.makedirs(folder_dir)
    else:
        os.makedirs(folder_dir)
