#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
호환용 실행 래퍼 (train_maskrcnn.py 호출)
"""
import sys
from train_maskrcnn import main if 'main' in dir() else None
import subprocess

if __name__ == '__main__':
    # train_maskrcnn.py를 동일 인자로 실행
    cmd = [sys.executable, 'train_maskrcnn.py'] + sys.argv[1:]
    exit_code = subprocess.call(cmd)
    sys.exit(exit_code)
