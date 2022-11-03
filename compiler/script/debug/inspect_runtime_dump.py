import numpy as np
import os
file_dir = '/home/liujunjie/gitlab/megcc/compiler/build_host/dump/'
file_name_list = [
    'tensor:51_kernel_elementwise_TRUE_DIV_f32f32f32_1_4',
    'tensor:64_kernel_elementwise_TRUE_DIV_f32f32f32_1_4',
    'tensor:76_kernel_elementwise_TRUE_DIV_f32f32f32_1_5',
    'tensor:88_kernel_elementwise_TRUE_DIV_f32f32f32_1_5',
]
def print_tensor(file_path, shape=None):
    with open(file_path, 'rb') as f:
        tensor = np.frombuffer(f.read(), dtype=np.float32)
    mean = np.average(tensor)
    max  = np.max(tensor)
    min  = np.min(tensor)
    if shape:
        tensor = tensor.reshape(shape)
        print(tensor) 
    print(os.path.basename(file_path), tensor.shape, tensor.flatten()[:10], 'mean', mean, 'max', max, 'min', min)
    
for file_name in file_name_list:
    print_tensor(file_dir+file_name)
    print_tensor(file_dir+file_name + '_input0')
    print('')