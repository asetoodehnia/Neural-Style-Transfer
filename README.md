## Neural Style Transfer

### Adel Setoodehnia

### The Code
All the code required to generate the images can be found in main.py
***NOTE***: there are dependencies to the root directory structure that must be met in order to run the code successfully.

### How to run the code

Running the code is very simple:

- type `python3 main.py` followed by the corresponding flags and arguments to run specific parts of the code.
- type `python3 main.py -h` to see the following help message on how each command works:

usage: main.py [-h] [-num_epochs NUM_EPOCHS] [-alpha ALPHA] [-beta BETA]
               [-rand] [-run RUN RUN]

optional arguments:
  -h, --help            show this help message and exit
  -num_epochs NUM_EPOCHS
                        number of epochs.
  -alpha ALPHA          value of alpha (content loss weight).
  -beta BETA            value of beta (style loss weight).
  -rand                 whether or not to initialize with random noise.
  -run RUN RUN          computes style transfer between content image and and
                        style image. Expects relative file path to content and
                        style image respectively.

Enjoy! :)
