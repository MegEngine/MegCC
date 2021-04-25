|  CV算子 | Limit | Dtype | Backend | 
|---|---|---|---|
| transpose |   | ui8 | arm/barematel  |
| flip |  | ui8 | arm/barematel  |
| cvt_color | rgb2bgr,yuv2bgr,rgb2yuv | ui8 | arm/barematel  |
| resize |  linear | ui8 | arm/barematel  |
| rotate |   | ui8 | arm/barematel  |
| warp_affine | replicate_linear,constant_linear  | ui8 | arm/barematel  |
| roi_copy |   | ui8 | arm/barematel  |


|  NN算子 | Arm64 | ArmV7 | Barematel | 
|---|---|---|---|
| conv  | nchw44-dot, nchw44-f32, nchw-f32  | nchw44-f32, nchw-f32  |  nchw44-f32, nchw-f32 |
| elemwise |  RELU,EXP,ADD,H_SWISH,SIGMOID,SUB,MUL,TRUE_DIV,FUSE_ADD_RELU,FUSE_MUL_ADD3| same with Arm64 | RELU,EXP,NEGATE,ROUND,ABS,H_SWISH,LOG,ADD,SUB,MUL,MAX,MIN,LEQ,LT,FLOOR_DIV,TRUE_DIV,FUSE_ADD_RELU,FUSE_ADD_SIGMOID,FUSE_ADD_TANH,FUSE_MUL_ADD3,FUSE_MUL_ADD4 |
| elemwise_multi |  |  | qadd |
| pool | nchw44-f32,nchw44-int8 | same with Arm64 | nchw-f32-int8 |
| reduce | f32 | same with Arm64 | f32 |
| relayout | all | same with Arm64 | all |
| resize | nchw-f32-LINEAR | same with Arm64 | nchw-f32-LINEAR |
| matmul | nchw-f32 | nchw-f32 | nchw-f32 |
| matinv |  |  | nchw-f32 |
| typecvt | q8<->f32, u8->f32, q8<->q8 | same with Arm64 | near all |
| warpaffine | nhwc-nchw-f32-u8 | same with Arm64 | nhwc-nchw-f32-u8 |
| warp_perspective |  |  | nhwc-nchw-f32-u8 |
| batch_matmul |  |  | nchw-f32 |
| powc |  |  | nchw-f32 |
| indexingMultiAxis |  |  | f32-i32 |



