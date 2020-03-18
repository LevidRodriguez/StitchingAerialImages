

set -ex



test -f ${PREFIX}/include/x264.h
test -f ${PREFIX}/lib/libx264.a
test -f ${PREFIX}/lib/libx264.so
test -f ${PREFIX}/lib/libx264.so.152
x264 --help
exit 0
