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

#ifndef TENSORFLOW_IO_CORE_FILESYSTEMS_FILESYSTEM_PLUGINS_H
#define TENSORFLOW_IO_CORE_FILESYSTEMS_FILESYSTEM_PLUGINS_H

#include <stdlib.h>

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"

namespace tensorflow {
namespace io {

static void* plugin_memory_allocate(size_t size) { return calloc(1, size); }
static void plugin_memory_free(void* ptr) { free(ptr); }

namespace az {

void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops, const char* uri);

}  // namespace az

namespace hdfs {

void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops, const char* uri);

}  // namespace hdfs

namespace http {

void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops, const char* uri);

}  // namespace http

namespace s3 {

void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops, const char* uri);

}  // namespace s3

namespace dfs {

void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops, const char* uri);

}  // namespace dfs

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_FILESYSTEMS_FILESYSTEM_PLUGINS_H
