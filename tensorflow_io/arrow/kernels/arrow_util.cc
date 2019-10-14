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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_io/arrow/kernels/arrow_util.h"

namespace tensorflow {

Status ParseEndpoint(std::string endpoint, std::string* endpoint_type,
                     std::string* endpoint_value) {
  size_t sep_pos = endpoint.find(':');

  // Check for a proper format
  if (sep_pos == std::string::npos) {
    return errors::InvalidArgument(
        "Expected endpoint to be in format <endpoint_type>://<endpoint_value> "
        "or <host>:<port> for tcp IPv4, but got: " + endpoint);
  }

  // If IPv4 and no endpoint type specified, descriptor is entire endpoint
  if (endpoint.substr(sep_pos + 1, 2) != "//") {
    *endpoint_type = "";
    *endpoint_value = endpoint;
    return Status::OK();
  }

  // Parse string as <endpoint_type>://<endpoint_value>
  *endpoint_type = endpoint.substr(0, sep_pos);
  *endpoint_value = endpoint.substr(sep_pos + 3);

  return Status::OK();
}

Status ParseHost(std::string host, std::string* host_address, std::string* host_port) {
  size_t sep_pos = host.find(':');
  if (sep_pos == std::string::npos || sep_pos == host.length()) {
    return errors::InvalidArgument(
        "Expected host to be in format <host>:<port> but got: " + host);
  }

  *host_address = host.substr(0, sep_pos);
  *host_port = host.substr(sep_pos + 1);

  return Status::OK();
}

}  // namespace tensorflow
