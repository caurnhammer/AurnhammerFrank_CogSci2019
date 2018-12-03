Code for the recurrent neural network language models discussed in "Comparing gated and simple recurrent neural network architectures as models of human sentence processing": https://psyarxiv.com/wec74/

The code is based on the neural network language model example provided by PyTorch: https://github.com/pytorch/examples/tree/master/word_language_model

The code works in conjunction with three files: train.txt, valid.txt, test.txt which by default are searched in the folder ./corpus/ .

The file gated_surprisal.py loads the saved models and computes surprisal on the test data. It will create the output file gated_surprisal.txt in the folder ./output/ .

The current version of this project is developed and tested in a lamachine virtual environment: https://proycon.github.io/LaMachine/
