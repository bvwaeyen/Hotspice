@echo off
rem Call this batch file as 'profile <name>.py' or 'profile -m <module_name>', where <name>.py is the
rem file to be profiled, or in case of a module use <module_name> as one would usually import the module.
rem The cProfile output is saved in a file as "/profiling/cProfile_<name>_<timestamp>.txt".

IF NOT EXIST "%~dp0\profiling" md "%~dp0\profiling"
IF "%1" == "-m" (
    python -m cProfile -s tottime -m %2 > "%~dp0\profiling\cProfile_%2_%date:~6,4%%date:~3,2%%date:~0,2%%time:~0,2%%time:~3,2%%time:~6,2%.txt"
) ELSE (
    python -m cProfile -s tottime %1 > "%~dp0\profiling\cProfile_%~nx1_%date:~6,4%%date:~3,2%%date:~0,2%%time:~0,2%%time:~3,2%%time:~6,2%.txt"
)
