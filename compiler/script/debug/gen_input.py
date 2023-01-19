import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dtype', type=str, help='dtype like uint8, float32')
parser.add_argument('--shape', type=str, help='shape like 1,1,128,128')
parser.add_argument('-o',
                    '--output',
                    type=str,
                    required=True,
                    help='output path')
args = parser.parse_args()
shape = [*(int(i) for i in args.shape.split(','))]
if args.dtype == 'uint8':
    res = np.random.randint(0, 255, size=shape, dtype=args.dtype)
else:
    assert args.dtype == 'float32'
    res = np.random.randint(0, 255, size=shape).astype(args.dtype) / 256
res.tofile(args.output)
np.save(args.output + '.npy', res, False)
