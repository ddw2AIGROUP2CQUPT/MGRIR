# MGRIR
A Multi-granularity Graph-based Representation for Image Recognition



## Graph prepared

For mnist datasets:

1. Enter directory `\MGRIR\graph\mnist`, and replace the corresponding data path

2. then run `python to_h5.py`

3. If you want to filter edges, the code will in the `find_boundary.py` 

   ```python
   if weight >= 0: # you can set the threshold you want
       ...
   ```

The same is true for cifar data (skip)



## Train the model

For mnist datasets:

1. Enter directory `\MGRIR`, and replace the corresponding graph path and models

2. Then run `python main_train.py`. The training model files will be stored in the ”checkpoints“ folder, and the training logs will be recorded in the ”runs“ folder.

3. You can run the `test.py` to take the model named `--checkpoint_path` to test the accuracy of all training sets. Replace the graph path `--data_dir`



## Visualization the graph

If you want to visualization the graph, more details can be found in `find_boundary.py`

