

set -ex



test -f ${PREFIX}/lib/libgnutls${SHLIB_EXT}
test -f ${PREFIX}/lib/libgnutlsxx${SHLIB_EXT}
exit 0
