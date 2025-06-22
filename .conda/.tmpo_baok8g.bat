@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "C:\Users\hp\miniconda3\condabin\conda.bat" activate "d:\naufalarizq\ipb\lomba\lkti-bi\olah-data\.conda"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@d:\naufalarizq\ipb\lomba\lkti-bi\olah-data\.conda\python.exe -Wi -m compileall -q -l -i C:\Users\hp\AppData\Local\Temp\tmp1zidiqie -j 0
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL
