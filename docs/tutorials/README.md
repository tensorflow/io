# TensorFlow I/O Tutorials

TensorFlow I/O welcomes and highly encourages tutorial contributions.


## How To Contribute

I/O tutorials are created using [Google Colab](https://colab.research.google.com/)
and the jupyter notebooks are saved to this directory in the repository. To do
this, follow the below steps:

1. Create a new branch on your fork of TensorFlow I/O.
2. Goto [Google Colab](https://colab.research.google.com/) and start a new
notebook using addons example template:
[notebook template](https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb).
3. Edit the the links for the "View source on GitHub" and "Run in Google Colab"
URL boxes so that they match the name of your new example notebook.
4. Follow the guidelines of the template.
5. "Save a copy in Github" and select your new branch. The notebook should be
named `subpackage_submodule`.
6. After step 5, the notebook will be committed to your branch directly from colab.
However, to check for linting issues and to auto format your notebook, pull the changes to your system
and run the following from the `io` directory:
    ```console
    $ sudo python3 -m pip install setuptools
    $ sudo python3 -m pip install -U git+https://github.com/tensorflow/docs
    $ echo "Auto format the notebooks: "
    $ find docs -name '*.ipynb' | xargs python3 -m tensorflow_docs.tools.nbfmt
    $ echo "Check for failed lint: "
    $ find docs -name '*.ipynb' | xargs python3 -m tensorflow_docs.tools.nblint --arg=repo:tensorflow/io
    ```
7. Update `docs/tutorials/_toc.yaml` with the notebook details (please refer existing entries).
8. Submit the branch as a PR on the TF-I/O [Github](https://github.com/tensorflow/io)