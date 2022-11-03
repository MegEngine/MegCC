## example code 

* `standard_OS` directory is for standard operating system

  * can build with megcc `runtime`  with the same env(same toolchains, or one CMakeList.txt)

* `Nonstandard_OS` directory is for non-standard operating system code examples

  * how to work for `Nonstandard_OS`:

    * check `runtime/CMakeLists.txt` option `TINYNN_BUILD_FOR_NOT_STANDARD_OS`,  will use clang --target arch-xxx to build arch-xxx only static library
      * `TINYNN_BUILD_FOR_NOT_STANDARD_OS` will auto enable `TINYNN_CALLBACK_ENABLE`, which will isolate APIS(always be system related) to customer space
    * check `runtime/scripts/check_tinynn_lib.py`, will use this scripts to check static library usability
  
  * `tee` directory is for [TEE](https://en.wikipedia.org/wiki/Trusted_execution_environment) ca/ta example, need build at [optee_example](https://optee.readthedocs.io/en/latest/building/gits/optee_examples/optee_examples.html) or other vendor tee sdk
    * build `TinyNN` with  `-o NOT_STANDARD_OS` flag
      * python3 runtime/scripts/runtime_build.py --kernel_dir YOUR_BUILD/kernel_dir/ --cross_build --cross_build_target_arch aarch64 --cross_build_target_os NOT_STANDARD_OS --remove_old_build
      * run example: python3 tee/test_optee.py --tinynn_lib_install_dir YOUR_BUILD/kernel_dir/install --optee_repo_dir YOUR_OPTEE_REPO_ROOT
  * `freeRTOS` directory is for [freeRTOS](https://www.freertos.org/) example, need build at [freeRTOS repo](https://github.com/FreeRTOS/FreeRTOS)
  
    * build `TinyNN` with  `-o NOT_STANDARD_OS` flag
      * python3 runtime/scripts/runtime_build.py --kernel_dir YOUR_BUILD/kernel_dir/ --cross_build --cross_build_target_arch cortex-m --cross_build_target_os NOT_STANDARD_OS --remove_old_build
      * run example: python3 freeRTOS/test_freertos.py --tinynn_lib_install_dir YOUR_BUILD/kernel_dir/install --free_rtos_repo_dir YOUR_FREERTOS_REPO_ROOT
      * you can refs this example to deploy megcc runtime to others [RTOS](https://en.wikipedia.org/wiki/Real-time_operating_system)
  
  * `bare_board` directory for no operating system env , test use [QEMU](https://en.wikipedia.org/wiki/QEMU)
    * now only example for arm and aarch64, for aarch64 exmaple by cmd:
      * python3 runtime/scripts/runtime_build.py --kernel_dir YOUR_BUILD/kernel_dir/ --cross_build --cross_build_target_arch aarch64 --cross_build_target_os NOT_STANDARD_OS --remove_old_build
      * run example: python3 bare_board/test_bare_board_qemu.py --tinynn_lib_install_dir YOUR_BUILD/kernel_dir/install --test_arch aarch64
    * also, the examples code can be use at standard operating system
      * just change the toolchains
      * do not build with `startup.s` and link with `link.ld`
