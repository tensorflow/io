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

#include "tensorflow_io/core/kernels/bigquery/bigquery_lib.h"

namespace tensorflow {
namespace {
::tensorflow::error::Code GcpErrorCodeToTfErrorCode(::grpc::StatusCode code) {
  switch (code) {
    case ::grpc::StatusCode::OK:
      return ::tensorflow::error::OK;
    case ::grpc::StatusCode::CANCELLED:
      return ::tensorflow::error::CANCELLED;
    case ::grpc::StatusCode::INVALID_ARGUMENT:
      return ::tensorflow::error::INVALID_ARGUMENT;
    case ::grpc::StatusCode::DEADLINE_EXCEEDED:
      return ::tensorflow::error::DEADLINE_EXCEEDED;
    case ::grpc::StatusCode::NOT_FOUND:
      return ::tensorflow::error::NOT_FOUND;
    case ::grpc::StatusCode::ALREADY_EXISTS:
      return ::tensorflow::error::ALREADY_EXISTS;
    case ::grpc::StatusCode::PERMISSION_DENIED:
      return ::tensorflow::error::PERMISSION_DENIED;
    case ::grpc::StatusCode::UNAUTHENTICATED:
      return ::tensorflow::error::UNAUTHENTICATED;
    case ::grpc::StatusCode::RESOURCE_EXHAUSTED:
      return ::tensorflow::error::RESOURCE_EXHAUSTED;
    case ::grpc::StatusCode::FAILED_PRECONDITION:
      return ::tensorflow::error::FAILED_PRECONDITION;
    case ::grpc::StatusCode::ABORTED:
      return ::tensorflow::error::ABORTED;
    case ::grpc::StatusCode::OUT_OF_RANGE:
      return ::tensorflow::error::OUT_OF_RANGE;
    case ::grpc::StatusCode::UNIMPLEMENTED:
      return ::tensorflow::error::UNIMPLEMENTED;
    case ::grpc::StatusCode::INTERNAL:
      return ::tensorflow::error::INTERNAL;
    case ::grpc::StatusCode::UNAVAILABLE:
      return ::tensorflow::error::UNAVAILABLE;
    case ::grpc::StatusCode::DATA_LOSS:
      return ::tensorflow::error::DATA_LOSS;
    case ::grpc::StatusCode::UNKNOWN:  // Fallthrough
    default:
      return ::tensorflow::error::UNKNOWN;
  }
}
}  // namespace

Status GrpcStatusToTfStatus(const ::grpc::Status& status) {
  if (status.ok()) {
    return Status::OK();
  }
  return Status(GcpErrorCodeToTfErrorCode(status.error_code()),
                strings::StrCat("Error reading from Cloud BigQuery: ",
                                status.error_message()));
}

string GrpcStatusToString(const ::grpc::Status& status) {
  if (status.ok()) {
    return "OK";
  }
  return strings::StrCat("Status code: ",
                         ::tensorflow::error::Code_Name(
                             GcpErrorCodeToTfErrorCode(status.error_code())),
                         " error message:", status.error_message(),
                         " error details: ", status.error_details());
}

Status GetDataFormat(string data_format_str,
                     apiv1beta1::DataFormat* data_format) {
  if (data_format_str == "ARROW") {
    *data_format = apiv1beta1::DataFormat::ARROW;
  } else if (data_format_str == "AVRO") {
    *data_format = apiv1beta1::DataFormat::AVRO;
  } else {
    return errors::Internal("Unsupported data format: " + data_format_str);
  }
  return Status::OK();
}

}  // namespace tensorflow
