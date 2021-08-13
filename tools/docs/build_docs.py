# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Modified from the tfdocs example api reference docs generation script.

This script generates API reference docs.

Build whl file in wheelhouse, note:
1) The `docker` installation is needed.
2) There are 4 whl files in wheelhouse for 2.7, 3.5, 3.6, 3.7
   The `Install pre-requisites` will selectively only install one version.

$ bash -x -e .travis/python.release.sh

Install pre-requisites.

$> python -m pip pip install -U git+https://github.com/tensorflow/docs
$> python -m pip install wheelhouse/tensorflow_io-*-cp$(python -c 'import sys; print(str(sys.version_info[0])+str(sys.version_info[1]))')*.whl

Generate Docs:

$> from the repo root run: python tools/docs/build_docs.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib

from absl import app
from absl import flags

os.environ["GENERATING_TF_DOCS"] = "True"
import tensorflow_io.python.api as tfio

from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import parser
from tensorflow_docs.api_generator import public_api
from tensorflow_docs.api_generator import utils

PROJECT_SHORT_NAME = 'tfio'
PROJECT_FULL_NAME = 'TensorFlow I/O'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'git_branch',
    default=None,
    help='The name of the corresponding branch on github.')

flags.DEFINE_string("output_dir", "/tmp/io_api",
                    "Where to output the docs")

CODE_PREFIX_TEMPLATE = "https://github.com/tensorflow/io/tree/{git_branch}/tensorflow_io"
flags.DEFINE_string(
    "code_url_prefix", None,
    "The url prefix for links to the code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "io/api_docs/python",
                    "Path prefix in the _toc.yaml")

flags.mark_flags_as_mutual_exclusive(['code_url_prefix', 'git_branch'])


def main(argv):
    if argv[1:]:
        raise ValueError('Unrecognized arguments: {}'.format(argv[1:]))

    if FLAGS.git_branch:
      code_url_prefix = CODE_PREFIX_TEMPLATE.format(git_branch=FLAGS.git_branch)
    elif FLAGS.code_url_prefix:
      code_url_prefix = FLAGS.code_url_prefix
    else:
      code_url_prefix = CODE_PREFIX_TEMPLATE.format(git_branch='master')

    doc_generator = generate_lib.DocGenerator(
        root_title=PROJECT_FULL_NAME,
        # Replace `tensorflow_docs` with your module, here.
        py_modules=[(PROJECT_SHORT_NAME, tfio)],
        base_dir=pathlib.Path(tfio.__file__).parents[2],
        code_url_prefix=code_url_prefix,
        # This callback cleans up a lot of aliases caused by internal imports.
        callbacks=[public_api.explicit_package_contents_filter],
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path)

    doc_generator.build(FLAGS.output_dir)

    print('Output docs to: ', FLAGS.output_dir)


if __name__ == '__main__':
    app.run(main)
