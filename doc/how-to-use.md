# MegCC Release Tree
```
.
├── bin       :megcc release software, used to gen kernels and binded model
├── example   :example to use megcc
├── runtime   :runtime framework, should compile with generated kernels
└── script    :script used to pack resource used by pipeline
```
```
bin
├── mgb-importer  :aux app, import mdl model and output model mlir
├── megcc-opt     :aux app, use megcc pass to optimize the mlir, it can show the detail of ir transform
├── mgb-runner    :aux app, run mdl model with mgb naive kernel, used to check megcc correctness
├── hako-to-mgb   :aux app, unpack hako model to mgb model
└── mgb-to-tinynn :core app, megcc compiler, compile mdl model to kernel and binded model

```

## Use Example In Android
- Download the precompiled megcc from [barin++ oss web](https://oss.iap.hh-b.brainpp.cn/megengine-built/megcc)
- Extract the download tar file. `tar -xvf megcc_release_*.tar.gz`
### Fast Try
> warning: **You should make sure a ndk is install in your computer，and set env NDK_ROOT to its direction** 

There are many scripts to help user compile a model and the recompile the generated kernel to SDK, user can use these scripts to have a glance.
#### Compile mobilenet model
- Execute `cd release`，and `./example/run_megcc.sh`，it will compile the mobilenet model store in the example dir，and generate `megcc_gen` that contains all resource needed by deploy. `megcc_gen` is also packed to `megcc_ppl_gen.tar.gz`  
#### Run mobilenet example 
> warning: **Make sure you have termux with rsync app that can access by ssh user@you_phone** 
 
* Execute `./ppl_build.sh` will compile the kernel and runtime to a executable file.
* Execute `python3 test_model.py --target user@you_phone ./` in `megcc_gen` directory, this command will send the build executable file to the target and execute it.

### Megcc Compile Mobilenet Detail
There are there steps to finish compile the mobilenet model with MegCC.
* First, describe models and cv kernels which pipeline need by json file.  
* Second, generate tinynn(runtime name) kernels and tinynn models by dumping the json file
* Final, integrate tinynn src file and models to you pipeline with lite API. 
#### Fill json File
Fill json file with all models and cv kernel the sdk needed. You need to specify which input shape is used for inference. If you may use two instance of input shape for one model, use `:` to separate two instance. 
For example:
```json
{
    "dump_dir":"./batch_dump/",
    "models":[
        {
            "model_name":"det_nchw44",
            "model_path":"path/to/det_u8.mdl",
            "input_shape_str":"data=(1,1,384,288):data=(1,1,288,384)",
            "enable_nchw44":true
        },
        {
            "model_name":"pf_nchw44",
            "model_path":"path/to/pf_u8.mdl",
            "input_shape_str":"data=(1,1,112,112)",
            "enable_nchw44":true
        }
    ],
    "cv":{
        "transpose":["ui8"],
        "roicopy":["ui8"],
        "rotate":["ui8"],
        "flip":["ui8"],
        "resize_linear":["ui8"],
        "warp_affine_replicate_linear":["ui8"],
        "rgb2bgr":["ui8"],
        "yuv2bgr_nv21":["ui8"],
        "rgb2yuv":["ui8"]
    }
}
```
#### Generate kernel and model with json file
Generate kernels and models which needed by sdk with prebuild mgb-to-tinynn app. There is a helper script named `ppl_gen.sh` to dump the files and pack them to a tar file named `megcc_ppl_gen.tar.gz`.   
- Execute `./script/ppl_gen.sh ./bin/mgb-to-tinynn ./path/to/you.json megcc_gen --arm64`. MegCC will compile models and dump new models with armv8 kernels, then pack the models, kernels, runtime file to `megcc_ppl_gen.tar.gz`.    
- Or execute `./script/ppl_gen.sh ./bin/mgb-to-tinynn ./path/to/you.json megcc_gen --arm64v7` to dump both kernel for armv8 and armv7. However model support both armv8 and armv7 will be larger than the one only support one arch.   

#### Integrate megcc
> warning: **You should make sure a NDK is installed in your computer，and set env NDK_ROOT to its direction** 

- Unzip `megcc_ppl_gen.tar.gz` by execute `tar -xvf megcc_ppl_gen.tar.gz` and enter the extracted dir
  - if build runtime with armv8, execute `./ppl_build.sh`.
  - if build with armv7, execute `./ppl_build.sh -m armeabi-v7a`. 
- Then the new tinynn model is in `model` directory, header and lib is in `./build/install/`. `./build/install/bin/tinynn_test_lite` is an example to run model, which compile from the source file `./example/lite_main.c`. 
- Integrate megcc with `lite` API to run tinynn model, reference to `lite_main.c`. Use CV API from `tinycv_c.h` to construct pipeline.  

#### Check megcc correctness and performance
> warning: **You should make sure a ndk is install in your computer，and set env NDK_ROOT to its direction** 

>  warning: **Make sure you have termux with rsync app that can access by ssh user@you_phone** 

`test_model.py` script will help you to check the result correctness and performance stand alone with MegEngine   
Checking correctness and performance, execute `python3 test_model.py --target user@you_phone ./ --mdl="new_model_name_in_json_file:/path/to/the/origin/model.mdl"`，the log will report whether the result is correct.  

## Advanced usage
Megcc component is build on mlir. You can use these tools to explore the detail of compiling. 
### mgb-to-tinynn
mgb-to-tinynn is core compiler, use `--help` to get more detail.   
Execute `./bin/mgb-to-tinynn ./example/mobilenet.mdl --input-shapes="data=(1,3,224,224)" ./dump_kernel --arm64 --enable_nchw44` to dump mdl model to tiny model, and save kernel to dump_kernel directory.  
Use `--arm64v7` rather than `--arm64` to dump both arm64 and armv7 kernel. It will make model bigger than the one only dumped for arm64 arch.   
Use `--enable_nchw44_dot` to enable dot kernel support.    
Use `--save-model` to pack tiny model to c file that you can embed model into runtime. It will be useful, if there is not file system in deploy environment   

### mgb-importer
mgb-importer is used to parse megengine mdl or mge model to mlir text. Reading mlir text will help you make clear the model detail.   
Execute `./bin/mgb-importer example/mobilenet.mdl mobilenet.mlir` to dump mdl model to mlir text. Explore the mlir file, module means model, ParamStorage means weights, func means compute graph, instruction in func means layer or op  

### mgb-runner
mgb-runner is used to run megengine model and write result to file. It is used to check correctness of megcc by test_model script.   
Execute `./bin/mgb-runner ./example/mobilenet.mdl ./mgb_out --input-shapes="data=(1,3,224,224)" --input-data="data=input_1_3_224_224_fp32.bin"` with input bin file input_1_3_224_224_fp32.bin, result is stored in mgb_out directory  

### hako-to-mgb
hako-to-mgb is used to unpack hako model which end with '.emod'. Version 2 unpack is default, you can use `--hako 1` to unpack version 1 model.  

### megcc-opt
megcc-opt is used to show intermediate result of mgb-to-tinynn. You can get origin mlir from mgb-importer, and trans it to final mlir by `--MGB-to-Kernel --finalizing-bufferize --memory-forwarding --static-memory-planning` args.  
