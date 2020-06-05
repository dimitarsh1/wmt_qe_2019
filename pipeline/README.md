# Siamese text classifier using BiRNN, LSTM network.

Requires: 
- pytorch v. 0.3.1 
- python v3 

Train a Siamese Neural network for sentence-pair classifier.

usage: 
    train.py [-h] -d DATA_FOLDER [-s SOURCE_EXT] [-t TARGET_EXT]
                [-b BATCH_SIZE] [-a ATTENTION_TYPE] [-g GPUID]

optional arguments:

    -h, --help            show this help message and exit
    -d DATA_FOLDER, --data-folder DATA_FOLDER
                        the folder containing the train, test, dev sets.
    -s SOURCE_EXT, --source-ext SOURCE_EXT
                        the extension of the source files.
    -t TARGET_EXT, --target-ext TARGET_EXT
                        the extension of the target files.
    -b BATCH_SIZE, --batch-size BATCH_SIZE
                        the batch size.
    -a ATTENTION_TYPE, --attention-type ATTENTION_TYPE
                        the attention type: 'dot', 'rte', 'None'.
    -g GPUID, --gpuid GPUID
                        the ID of the GPU to use.

Example:
- Train with default parameters from data stored in the 'data' folder:
```python3 train.py -d data```

- Train on a GPU with ID 1:
```python3 train.py -d data -g 1```

- Train using attention and batch size of 128
```python3 trian.py -d data -a dot -b 128```
