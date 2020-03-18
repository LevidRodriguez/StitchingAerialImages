:: site.cfg should not be defined here.  It is provided by blas devel packages (either mkl-devel or openblas-devel)

COPY %PREFIX%\site.cfg site.cfg

python setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1

COPY %RECIPE_DIR%\f2py.bat %PREFIX%\Scripts
