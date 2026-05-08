@echo off
set VENV_DIR=.venv_wit_final

echo Setting up %VENV_DIR% environment...
D:\installed\Python\Python310\python.exe -m venv %VENV_DIR%

echo Installing pip and requirements...
%VENV_DIR%\Scripts\python.exe -m pip install --upgrade pip
%VENV_DIR%\Scripts\python.exe -m pip install -r requirements.txt

echo Downgrading ipywidgets and notebook for compatibility...
%VENV_DIR%\Scripts\python.exe -m pip install "ipywidgets==7.6.5" "notebook<7" nbclassic

echo Installing witwidget extension...
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
%VENV_DIR%\Scripts\jupyter-nbextension.exe install --py --symlink --sys-prefix witwidget
%VENV_DIR%\Scripts\jupyter-nbextension.exe enable --py --sys-prefix witwidget
%VENV_DIR%\Scripts\jupyter-nbextension.exe enable --py --sys-prefix widgetsnbextension

echo Environment setup complete! Use start_jupyter.bat to run.
