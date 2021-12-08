# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import warnings
import functools


def deprecate(message):
    """a decorator to aid in deprecating python api endpoints and
    suggesting alternatives.

    Args:
      message: A description of the deprecation activity and an
        alternative endpoint that can be used (if applicable).

    Example:

    ```
    from tensorflow_io.python.utils import deprecation

    @deprecation.deprecate("Use new_func instead")
    def old_func():
      return 1
    ```
    """

    def deprecated_wrapper(func_or_class):
        """Deprecation wrapper."""
        if isinstance(func_or_class, type):
            # If a class is deprecated, you actually want to wrap the constructor.
            cls = func_or_class
            if cls.__new__ is object.__new__:
                func = cls.__init__
                constructor_name = "__init__"
            else:
                # for classes where __new__ has been overwritten
                func = cls.__new__
                constructor_name = "__new__"
        else:
            cls = None
            constructor_name = None
            func = func_or_class

        @functools.wraps(func)
        def new_func(*args, **kwargs):  # pylint: disable=missing-docstring
            warnings.warn(message=message)
            return func(*args, **kwargs)

        if cls is None:
            return new_func
        else:
            # Insert the wrapped function as the constructor
            setattr(cls, constructor_name, new_func)
            return cls

    return deprecated_wrapper
