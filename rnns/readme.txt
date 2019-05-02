The code is based on the neural network language model example provided by PyTorch: https://github.com/pytorch/examples/tree/master/word_language_model

The code works in conjunction with three files: train.txt, valid.txt, test.txt which by default are searched in the folder ./corpus/ .
The files in this repo are blindtexts. The sources of the actual files are the ENCOW corpus and the experimental stimuli described in Frank 2013.

The file main.py trains the language models and requires the files data.py (loading corpus) and model.py (generating the RNN object) in the same directory.
The file gated_surprisal.py loads the saved models and computes surprisal on the test data. It will create the output file gated_surprisal.txt in the folder ./output/ .

The current version of this project is developed and tested in a lamachine virtual environment: https://proycon.github.io/LaMachine/
