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

import os
import sys

from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
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

class InstallCommand(InstallCommandBase):
  \"\"\"Override the dir where the headers go.\"\"\"

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self)
    self.install_headers = os.path.join(self.install_purelib, 'tensorflow_core',
                                        'include')
    self.install_lib = self.install_platlib
    return ret


class InstallHeaders(Command):
  \"\"\"Override how headers are copied.

  The install_headers that comes with setuptools copies all files to
  the same directory. But we need the files to be in a specific directory
  hierarchy for -I <include_dir> to work correctly.
  \"\"\"
  description = 'install C/C++ header files'

  user_options = [('install-dir=', 'd',
                   'directory to install header files to'),
                  ('force', 'f',
                   'force installation (overwrite existing files)'),
                 ]

  boolean_options = ['force']

  def initialize_options(self):
    self.install_dir = None
    self.force = 0
    self.outfiles = []

  def finalize_options(self):
    self.set_undefined_options('install',
                               ('install_headers', 'install_dir'),
                               ('force', 'force'))

  def mkdir_and_copy_file(self, header):
    install_dir = os.path.join(self.install_dir, os.path.dirname(header))
    # Get rid of some extra intervening directories so we can have fewer
    # directories for -I
    install_dir = re.sub('/google/protobuf_archive/src', '', install_dir)
    install_dir = re.sub('/include/tensorflow_core/', '/include/tensorflow/',
                         install_dir)

    # Copy external code headers into tensorflow_core/include.
    # A symlink would do, but the wheel file that gets created ignores
    # symlink within the directory hierarchy.
    # NOTE(keveman): Figure out how to customize bdist_wheel package so
    # we can do the symlink.
    external_header_locations = [
        'tensorflow_core/include/external/eigen_archive/',
        'tensorflow_core/include/external/com_google_absl/',
    ]
    for location in external_header_locations:
      if location in install_dir:
        extra_dir = install_dir.replace(location, '')
        if not os.path.exists(extra_dir):
          self.mkpath(extra_dir)
        self.copy_file(header, extra_dir)

    if not os.path.exists(install_dir):
      self.mkpath(install_dir)
    return self.copy_file(header, install_dir)

  def run(self):
    hdrs = self.distribution.headers
    if not hdrs:
      return

    self.mkpath(self.install_dir)
    for header in hdrs:
      (out, _) = self.mkdir_and_copy_file(header)
      self.outfiles.append(out)

  def get_inputs(self):
    return self.distribution.headers or []

  def get_outputs(self):
    return self.outfiles


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
    cmdclass={},
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
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

package = 'tensorflow==2.0.0rc2'
version = '0.9.0'
project = 'tensorflow-io'
if '--package-version' in sys.argv:
  print(package)
  sys.exit(0)

# Note: import setuptools later to avoid unnecessary dependency
from setuptools import sandbox # pylint: disable=wrong-import-position

if '--nightly' in sys.argv:
  nightly_idx = sys.argv.index('--nightly')
  version = version + ".dev" + sys.argv[nightly_idx + 1]
  project = 'tensorflow-io-nightly'
  sys.argv.remove('--nightly')
  sys.argv.pop(nightly_idx)

rootpath = tempfile.mkdtemp()
print("setup.py - create {} and copy tensorflow_io".format(rootpath))
shutil.copytree("tensorflow_io", os.path.join(rootpath, "tensorflow_io"))

print("setup.py - create {}/MANIFEST.in".format(rootpath))
with open(os.path.join(rootpath, "MANIFEST.in"), "w") as f:
  f.write("recursive-include tensorflow_io *.so")

print("setup.py - create {}/setup.py with required = '{}', "
      "project_name = '{}' and __version__ = {}".format(
          rootpath, package, project, version))
cmdclass = "{'install_headers':InstallHeaders,'install':InstallCommand,}"
with open(os.path.join(rootpath, "setup.py"), "w") as f:
  f.write(content.format(package, version, project, cmdclass))

datapath = None
if '--data' in sys.argv:
  data_idx = sys.argv.index('--data')
  datapath = sys.argv[data_idx + 1]
  sys.argv.remove('--data')
  sys.argv.pop(data_idx)
else:
  datapath = os.environ.get('TFIO_DATAPATH')

if datapath is not None:
  for rootname, _, filenames in os.walk(
      os.path.join(datapath, "tensorflow_io")):
    if (not fnmatch.fnmatch(rootname, "*test*") and
        not fnmatch.fnmatch(rootname, "*runfiles*")):
      for filename in [
          f for f in filenames if fnmatch.fnmatch(
              f, "*.so") or fnmatch.fnmatch(f, "*.py")]:
        src = os.path.join(rootname, filename)
        dst = os.path.join(
            rootpath,
            os.path.relpath(os.path.join(rootname, filename), datapath))
        print("setup.py - copy {} to {}".format(src, dst))
        shutil.copyfile(src, dst)

print("setup.py - run sandbox.run_setup {} {}".format(
    os.path.join(rootpath, "setup.py"), sys.argv[1:]))
sandbox.run_setup(os.path.join(rootpath, "setup.py"), sys.argv[1:])

if not os.path.exists("dist"):
  os.makedirs("dist")
for f in os.listdir(os.path.join(rootpath, "dist")):
  print("setup.py - copy {} to {}".format(
      os.path.join(rootpath, "dist", f), os.path.join("dist", f)))
  shutil.copyfile(os.path.join(rootpath, "dist", f), os.path.join("dist", f))
print("setup.py - remove {}".format(rootpath))
shutil.rmtree(rootpath)
print("setup.py - complete")
