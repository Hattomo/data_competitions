# data_competitions
Play ground for data competitions like kaggle, solafune, etc

## software
For new project, use the following software versions.
- python 3.10
- pytorch latest stable version >= 2.0.1
- poety

## Set up new project

```sh
#  create new  project
poetry new [project_name]

# install package
poetry add [package_name]

# setup environment
poetry install

# remove package
poetry remove [package_name]
```

## install package for my environment
```sh
# install pytorch 1
# (New way but it download all packages for each platform)
poetry source add torch_cu118 --priority=explicit https://download.pytorch.org/whl/cu118
poetry add torch torchvision torchaudio --source torch_cu118

# install pytorch 2
# https://download.pytorch.org/whl/torch_stable.html
# python:3.10 cuda:11.8 os:linux pytorch:2.0.1
[tool.poetry.dependencies]
torch = { url = "https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl"}

# install dev packages
poetry add pytest --group dev
poetry add yapf --group dev
poetry add pylint --group dev
```
## activate/deactivate poetry environment
```sh
poetry shell

source ./.venv/bin/activate
deactivate
```

## Run python program
```sh
poetry run python [program_name.py]
```
