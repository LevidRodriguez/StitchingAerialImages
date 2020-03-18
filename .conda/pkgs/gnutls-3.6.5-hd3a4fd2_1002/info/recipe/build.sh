#!/bin/bash

if [[ ${target_platform} =~ .*linux.* ]]; then
   export LDFLAGS="$LDFLAGS -Wl,-rpath-link,${PREFIX}/lib"
fi

# disable libidn for security reasons:
#   http://lists.gnupg.org/pipermail/gnutls-devel/2015-May/007582.html
# if ever want it back, package and link against libidn2 instead
#
# Also --disable-full-test-suite does not disable all tests but rather
# "disable[s] running very slow components of test suite"

export CPPFLAGS="${CPPFLAGS//-DNDEBUG/}"

libtoolize --copy --force --verbose
# libtoolize deletes things we need from build-aux, this puts them back
automake --add-missing --copy --verbose

./configure --prefix="${PREFIX}"          \
            --without-idn                 \
            --cache-file=test-output.log  \
            --disable-full-test-suite     \
            --disable-maintainer-mode     \
            --with-included-libtasn1      \
            --with-included-unistring     \
            --without-p11-kit || { cat config.log; exit 1; }

cat libtool | grep as-needed 2>&1 >/dev/null || { echo "ERROR: Not using libtool with --as-needed fixes?"; exit 1; }

make -j${CPU_COUNT} ${VERBOSE_AT}
make install
make -j${CPU_COUNT} check V=1 || { echo CONDA-FORGE TEST OUTPUT; cat test-output.log; cat tests/test-suite.log; cat tests/slow/test-suite.log; exit 1; }
