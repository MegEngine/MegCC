#!/bin/bash -e

set -x

cat > /etc/pip.conf <<EOF
[global]
timeout = 60
index-url =  http://mirrors.i.brainpp.cn/pypi/simple/
extra-index-url =
    http://pypi.i.brainpp.cn/brain/dev/+simple
trusted-host =
    mirrors.i.brainpp.cn pypi.i.brainpp.cn
    pypi.i.brainpp.cn
EOF

python3 -m pip install boto3
python3 -m pip install redis
aws --version || python3 -m pip install awscli
