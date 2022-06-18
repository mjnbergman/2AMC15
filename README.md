# Continuous environment:

## Python Environment
We installed our environment in the following way: 
1. `conda create --name DIC2 tensorflow-gpu`
2. `pip install ipykernel`
3. `python -m ipykernel install --user --name=DIC2`
4. `pip install gym opencv-python shapely matplotlib keras-rl`
5. `pip install tensorflow==2.9 --user`
6. `pip install tensorflow-probability`

This ensured for us that if a CUDA GPU is available, it can be used.