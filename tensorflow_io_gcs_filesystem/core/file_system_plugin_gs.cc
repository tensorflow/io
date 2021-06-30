/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_io_gcs_filesystem/core/file_system_plugin_gs.h"

#include "absl/strings/ascii.h"

#if defined(_MSC_VER)
#define TFIO_PLUGIN_EXPORT __declspec(dllexport)
#else
#define TFIO_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

// Please see:
// tensorflow/tensorflow/c/experimental/filesystem/filesystem_interface.h
// for definition of `TF_InitPlugin`
TFIO_PLUGIN_EXPORT void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  info->plugin_memory_allocate = tensorflow::io::plugin_memory_allocate;
  info->plugin_memory_free = tensorflow::io::plugin_memory_free;
  info->num_schemes = 1;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      tensorflow::io::plugin_memory_allocate(info->num_schemes *
                                             sizeof(info->ops[0])));
  // Load plugins only when the environment variable is set
  tensorflow::io::gs::ProvideFilesystemSupportFor(&info->ops[0], "gs");
}
