@echo off
rem Call this batch file as 'profile <name>.py' or 'profile -m <module_name>', where <name>.py is the
rem file to be profiled, or in case of a module use <module_name> as one would usually import the module.
rem To profile a function line-by-line, add the '@profile' decorator to the function, and call this batch
rem file as 'profile -l <name>.py'. Alternatively, you can run the command 'kernprof -l -v <name>.py'
rem (kernprof is a module installable through 'pip install line_profiler'), but this does not save the output.
rem Examples for each of the 3 cases:
rem profile hotspin/core.py
rem profile -l examples/pinwheelASI.py
rem profile -m hotspin.experiments

rem The output is saved in a file as "/profiling/<cProfile|kernprof>_<name>_<timestamp>.txt".

IF NOT EXIST "%~dp0\profiling" md "%~dp0\profiling"
IF "%1" == "-m" (
    python -m cProfile -s tottime -m %2 > "%~dp0\profiling\cProfile_%2_%date:~6,4%%date:~3,2%%date:~0,2%%time:~0,2%%time:~3,2%%time:~6,2%.txt"
) ELSE (
    IF "%1" == "-l" (
        echo "!!! Do not forget to add the @profile decorator the the function(s) of interest, otherwise the output of this profiling will be empty."
        kernprof -l -o "%~dp0\profiling\kernprof_%~nx2_%date:~6,4%%date:~3,2%%date:~0,2%%time:~0,2%%time:~3,2%%time:~6,2%.lprof" -v %2
        python -m line_profiler "%~dp0\profiling\kernprof_%~nx2_%date:~6,4%%date:~3,2%%date:~0,2%%time:~0,2%%time:~3,2%%time:~6,2%.lprof" > "%~dp0\profiling\kernprof_%~nx2_%date:~6,4%%date:~3,2%%date:~0,2%%time:~0,2%%time:~3,2%%time:~6,2%.txt"
        del "%~dp0\profiling\kernprof_%~nx2_%date:~6,4%%date:~3,2%%date:~0,2%%time:~0,2%%time:~3,2%%time:~6,2%.lprof"
    ) ELSE (
        python -m cProfile -s tottime %1 > "%~dp0\profiling\cProfile_%~nx1_%date:~6,4%%date:~3,2%%date:~0,2%%time:~0,2%%time:~3,2%%time:~6,2%.txt"
    )
)
