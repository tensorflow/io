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
absl::StatusCode GcpErrorCodeToTfErrorCode(::grpc::StatusCode code) {
  switch (code) {
    case ::grpc::StatusCode::OK:
      return absl::StatusCode::kOk;
    case ::grpc::StatusCode::CANCELLED:
      return absl::StatusCode::kCancelled;
    case ::grpc::StatusCode::INVALID_ARGUMENT:
      return absl::StatusCode::kInvalidArgument;
    case ::grpc::StatusCode::DEADLINE_EXCEEDED:
      return absl::StatusCode::kDeadlineExceeded;
    case ::grpc::StatusCode::NOT_FOUND:
      return absl::StatusCode::kNotFound;
    case ::grpc::StatusCode::ALREADY_EXISTS:
      return absl::StatusCode::kAlreadyExists;
    case ::grpc::StatusCode::PERMISSION_DENIED:
      return absl::StatusCode::kPermissionDenied;
    case ::grpc::StatusCode::UNAUTHENTICATED:
      return absl::StatusCode::kUnauthenticated;
    case ::grpc::StatusCode::RESOURCE_EXHAUSTED:
      return absl::StatusCode::kResourceExhausted;
    case ::grpc::StatusCode::FAILED_PRECONDITION:
      return absl::StatusCode::kFailedPrecondition;
    case ::grpc::StatusCode::ABORTED:
      return absl::StatusCode::kAborted;
    case ::grpc::StatusCode::OUT_OF_RANGE:
      return absl::StatusCode::kOutOfRange;
    case ::grpc::StatusCode::UNIMPLEMENTED:
      return absl::StatusCode::kUnimplemented;
    case ::grpc::StatusCode::INTERNAL:
      return absl::StatusCode::kInternal;
    case ::grpc::StatusCode::UNAVAILABLE:
      return absl::StatusCode::kUnavailable;
    case ::grpc::StatusCode::DATA_LOSS:
      return absl::StatusCode::kDataLoss;
    case ::grpc::StatusCode::UNKNOWN:  // Fallthrough
    default:
      return absl::StatusCode::kUnknown;
  }
}
}  // namespace

Status GrpcStatusToTfStatus(const ::grpc::Status& status) {
  if (status.ok()) {
    return OkStatus();
  }
  return Status(static_cast<tensorflow::errors::Code>(
                GcpErrorCodeToTfErrorCode(status.error_code())),
                strings::StrCat("Error reading from Cloud BigQuery: ",
                                status.error_message()));
}

string GrpcStatusToString(const ::grpc::Status& status) {
  if (status.ok()) {
    return "OK";
  }
  return strings::StrCat("Status code: ",
                         ::tensorflow::error::Code_Name(
                             static_cast<tensorflow::error::Code>(
                             GcpErrorCodeToTfErrorCode(status.error_code()))),
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
  return OkStatus();
}

}  // namespace tensorflow
