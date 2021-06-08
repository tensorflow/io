/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_io/core/filesystems/filesystem_plugins.h"

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
  const char* env_value = getenv("TF_USE_MODULAR_FILESYSTEM");
  std::string load_plugin = env_value ? absl::AsciiStrToLower(env_value) : "";

  info->plugin_memory_allocate = tensorflow::io::plugin_memory_allocate;
  info->plugin_memory_free = tensorflow::io::plugin_memory_free;
  info->num_schemes = 7;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      tensorflow::io::plugin_memory_allocate(info->num_schemes *
                                             sizeof(info->ops[0])));
  tensorflow::io::az::ProvideFilesystemSupportFor(&info->ops[0], "az");
  tensorflow::io::http::ProvideFilesystemSupportFor(&info->ops[1], "http");
  tensorflow::io::http::ProvideFilesystemSupportFor(&info->ops[2], "https");
  // Load plugins only when the environment variable is set
  if (load_plugin == "true" || load_plugin == "1") {
    tensorflow::io::s3::ProvideFilesystemSupportFor(&info->ops[3], "s3");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[4], "hdfs");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[5], "viewfs");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[6], "har");
  } else {
    tensorflow::io::s3::ProvideFilesystemSupportFor(&info->ops[3], "s3e");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[4], "hdfse");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[5], "viewfse");
    tensorflow::io::hdfs::ProvideFilesystemSupportFor(&info->ops[6], "hare");
  }
}
