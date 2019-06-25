### Compile avro library
A quick way to install it is as follows:
```
git clone https://github.com/apache/avro.git
cd avro/lang/c
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=avrolib -DCMAKE_BUILD_TYPE=Release -DTHREADSAFE=true
make
make test
make install
```
Note, that `make install` by default will place header files and library files into default places if you did not specify the `MAKE_INSTALL_PREFIX` directory.

When building avro 1.8.2 you may run into this issue https://issues.apache.org/jira/browse/AVRO-1844 with the jansson dependency. To pick up the bug fix I build from master. That is this hash a296ebc8894ad52ae7c0946e635b93cadc709e70.

On my system (RHEL 7) I also had to manually compile & install jansson from here http://www.digip.org/jansson/releases/ (used 2.12). I've also set the pkg path variable for `pkg_check_modules` to pick up the non-standard location of the jansson install like that
```export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig```.

After a successful compilation make sure to place the headers and lib into the folders `./include` and `./lib`. 

To avoid having to checkin multiple *.so files we rename the SONAME in `libavro.so` from `libavro.so.23` to `libavro.so` using this command
```
patchelf --set-soname libavro.so libavro.so
```

When inspecting the symbol names with
```
readelf -d libavro.so
```
you should see a listing like this
```
0x0000000000000001 (NEEDED)             Shared library: [libjansson.so.4]
0x0000000000000001 (NEEDED)             Shared library: [libz.so.1]
0x0000000000000001 (NEEDED)             Shared library: [liblzma.so.5]
0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
0x000000000000000e (SONAME)             Library soname: [libavro.so]
...
```