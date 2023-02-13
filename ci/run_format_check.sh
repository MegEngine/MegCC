#!/bin/bash -e

set -e
function set_pip_mirror() {
cat > /etc/pip.conf <<EOF
[global]
timeout = 60
index-url = http://mirrors.i.brainpp.cn/pypi/simple/
extra-index-url =
    http://pypi.i.brainpp.cn/brain/dev/+simple
trusted-host =
    mirrors.i.brainpp.cn
    pypi.i.brainpp.cn
EOF
}
set_pip_mirror
# cpp format
python3 -m pip install tqdm
clang-format --version || (wget http://brain-ftp.megvii-inc.com/clang-format-12-0-1-linux && chmod +x  clang-format-12-0-1-linux && ln -s clang-format-12-0-1-linux clang-format)
export PATH="$PWD:$PATH"
export CLANG_FORMAT="$(which clang-format)"
echo "using clang-format from $CLANG_FORMAT"
./script/format.py compiler runtime immigration benchmark yolox_example
