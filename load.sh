#!/bin/bash

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d sj2129tommy/nlc2bash-21epoch
unzip  -qq /content/ServalShell/nlc2bash-21epoch.zip
