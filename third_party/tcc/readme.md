## tcc
MegCC 中的 JIT 是通过 [tcc](https://download.savannah.gnu.org/releases/tinycc/) 实现的，
third_party 中的库和头文件是手动从 [tcc](https://download.savannah.gnu.org/releases/tinycc/) 的源码编译的 linux/macos x86-64 版本。
如果是 MacOS, 在构建 libtcc_apple_x86.a 时，需要先打上补丁: compat_new_macos.patch
注意：libtcc.a 依赖 ld，编译 tcc 时候 ld 的版本不要太高

### JIT实现
在 KernelGen 的 Jit 中调用 libtcc 中代码编译，链接，获取符号等接口，进行 C
code
在线编译，并最终会得到一个可执行的函数指针，通过这个函数指针进行函数调用，完成
jit 运行 C code，在线编译时候依赖的头文件通过字符串的形式写在了 JitHeader.h
中，这些头文件保存在 third_party/tcc/jitheader/ 下面，在线编译时需要链接 libtcc1.a 库文件，这个文件通过：

```
   xxd -i libtcc1.a > JitLib.h
```
在线编译时将 JitLib.h 文件写到 /tmp/libtcc1.a 中，并指定链接路径为
/tmp/，即可正确完成编译。

** JIT编译的 get workspace 函数不能依赖外部的 extern
函数，不然编译会报错，需要实现 GetWorkspaceBodyAndJitExec 这个不依赖 extern
函数的接口**
