# 3d-reconstruction
## Environment：windows 10
## python: 3.8.5

This 3D reconstruction process is implemented with face3d(github) and dlib-19.20.0(python)

### 1.Fetch and install face3d
#### 1.1 download the code from https://github.com/YadiraF/face3d
#### 1.2 Prepare BFM data according to the instruction of face3d
#### 1.3 Follow the Readme to install and run some examples

### 2.Install dilb
#### 1.1 download the code from http://dlib.net/
#### 1.2 Use command line in the folder path to install
```python 
python setup.py install
```
#### 1.3 run python_examples to check

### 3.3D reconstruction with dlib and face3d
run the code my_project.py with command
```python
python my_project_demo.py [faces_folder_path] [result_folder_path]
```
It will automatically handle all photos in the [faces_folder_path] and make re-rendered 3d face for every detected faces, then save them in [result_folder_path].
