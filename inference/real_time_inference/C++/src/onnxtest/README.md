
```shell
tar zxvf ../../lib/onnxruntime-linux-x64-1.11.0.tgz
```
Set the path of onnxruntime as the HOST_PACKAGE_DIR in CMakeLists.txt and make.

You can also compile the code by:
```shell
g++ -o run run.cpp ../../lib/onnxruntime-linux-x64-1.11.0/lib/libonnxruntime.so.1.11.0  -I../../lib/onnxruntime-linux-x64-1.11.0/include/
```
