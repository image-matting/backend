# Image Matting Backend

## How to install

### Poetry

We are using poetry as a dependency management tool.

1. Install poetry by following
   the [installation instructions](https://python-poetry.org/docs/#windows-powershell-install-instructions)
2. Install the [poetry plugin](https://plugins.jetbrains.com/plugin/14307-poetry) in PyCharm
3. Restart PyCharm

### Backend repository

1. git clone https://github.com/image-matting/backend.git
2. Open the cloned directory in Pycharm → File → Open
3. Create a virtual environment and activate it:
    - ```virtualenv venv```
    - Windows: ```.\venv\Scripts\activate```
      Linux: ```source ./venv/bin/activate```
4. Configure the python interpreter to be poetry:
    - Click on Python Interpreter in the bottom right corner
    - Add Interpreter → Poetry → Choose python.exe from the project's venv directory

   In the end, you should have something like this:
   ![img.png](https://user-images.githubusercontent.com/26183144/147248278-1fc05e08-17c9-4007-ac76-c196b0ea7e1e.png)
5. Install the dependencies
    - ```poetry install```
6. If the IDE still cannot find the dependencies, you can Alt + Enter on some red import statement and install it
   manually. This will hopefully make the IDE realize that it should be using the venv directory.
7. Execute pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

### Adding/Removing Dependencies

1. Adding a new dependency: ```poetry add <package name>```
    - In case of errors about missing .whl files, delete C:\Users\\\<USERNAME>\AppData\Local\pypoetry\Cache directory
3. Removing an existing dependency ```poetry remove <package_name>```
4. Further Information - [Poetry Basic Usage](https://python-poetry.org/docs/basic-usage/)
