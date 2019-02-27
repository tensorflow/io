# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import tempfile
import fnmatch

content = """
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
\"\"\"Setup for pip package.\"\"\"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

REQUIRED_PACKAGES = [
    '{}',
]
__version__ = '{}'
project_name = '{}'

class BinaryDistribution(Distribution):
  \"\"\"This class is needed in order to create OS specific wheels.\"\"\"

  def has_ext_modules(self):
    return True

setup(
    name=project_name,
    version=__version__,
    description=('TensorFlow IO'),
    author='Google Inc.',
    author_email='opensource@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='tensorflow io machine learning',
)
"""

# Note: Change to tensorflow == 1.13.0 once 1.13.0 is released
package = 'tensorflow>=1.13.0,<1.14.0'
version = '0.4.0'
project = 'tensorflow-io'
if '--package-version' in sys.argv:
  print(package)
  sys.exit(0)

# Note: import setuptools later to avoid unnecessary dependency
from setuptools import sandbox

if '--nightly' in sys.argv:
  nightly_idx = sys.argv.index('--nightly')
  version = version + ".dev" + sys.argv[nightly_idx + 1]
  project = 'tensorflow-io-nightly'
  sys.argv.remove('--nightly')
  sys.argv.pop(nightly_idx)

if '--preview' in sys.argv:
  preview_idx = sys.argv.index('--preview')
  version = version + ".dev" + sys.argv[preview_idx + 1]
  package = 'tf-nightly-2.0-preview'
  project = 'tensorflow-io-2.0-preview'
  sys.argv.remove('--preview')
  sys.argv.pop(preview_idx)


rootpath = tempfile.mkdtemp()
print("setup.py - create {} and copy tensorflow_io".format(rootpath))
shutil.copytree("tensorflow_io", os.path.join(rootpath, "tensorflow_io"))

print("setup.py - create {}/MANIFEST.in".format(rootpath))
with open(os.path.join(rootpath, "MANIFEST.in"), "w") as f:
  f.write("recursive-include tensorflow_io *.so")

print("setup.py - create {}/setup.py with required = '{}', project_name = '{}' and __version__ = {}".format(rootpath, package, project, version))
with open(os.path.join(rootpath, "setup.py"), "w") as f:
  f.write(content.format(package, version, project))

datapath = None
if '--data' in sys.argv:
  data_idx = sys.argv.index('--data')
  datapath = sys.argv[data_idx + 1]
  sys.argv.remove('--data')
  sys.argv.pop(data_idx)
else:
  datapath = os.environ.get('TFIO_DATAPATH')

if datapath is not None:
  for rootname, _, filenames in os.walk(os.path.join(datapath, "tensorflow_io")):
    if not fnmatch.fnmatch(rootname, "*test*") and not fnmatch.fnmatch(rootname, "*runfiles*"):
      for filename in fnmatch.filter(filenames, "*.so"):
        src = os.path.join(rootname, filename)
        dst = os.path.join(rootpath, os.path.relpath(os.path.join(rootname, filename), datapath))
        print("setup.py - copy {} to {}".format(src, dst))
        shutil.copyfile(src, dst)

print("setup.py - run sandbox.run_setup {} {}".format(os.path.join(rootpath, "setup.py"), sys.argv[1:]))
sandbox.run_setup(os.path.join(rootpath, "setup.py"), sys.argv[1:])

if not os.path.exists("dist"):
  os.makedirs("dist")
for f in os.listdir(os.path.join(rootpath, "dist")):
  print("setup.py - copy {} to {}".format(os.path.join(rootpath, "dist", f), os.path.join("dist", f)))
  shutil.copyfile(os.path.join(rootpath, "dist", f), os.path.join("dist", f))
print("setup.py - remove {}".format(rootpath))
shutil.rmtree(rootpath)
print("setup.py - complete")
