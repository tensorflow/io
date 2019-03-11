curl https://bootstrap.pypa.io/get-pip.py | python
python -m pip install pylint

curl -o .pylint -sSL https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc

find . -name \*.py | xargs pylint --rcfile=.pylint
