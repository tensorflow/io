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

#ifndef TENSORFLOW_IO_CORE_KERNELS_ARROW_STREAM_CLIENT_H_
#define TENSORFLOW_IO_CORE_KERNELS_ARROW_STREAM_CLIENT_H_

#include "arrow/io/api.h"

namespace tensorflow {
namespace data {

// Class to wrap a socket as a readable Arrow InputStream
class ArrowStreamClient : public arrow::io::InputStream {
 public:
  ArrowStreamClient(const std::string& endpoint);
  ~ArrowStreamClient() override;

  arrow::Status Connect();
  arrow::Status Close() override;
  bool closed() const override;
  arrow::Result<int64_t> Tell() const override;
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override;

 private:
  const std::string endpoint_;
  int sock_;
  int64_t pos_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_ARROW_STREAM_CLIENT_H_
