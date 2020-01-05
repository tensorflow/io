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

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

#include "api/Compiler.hh"
#include "api/DataFile.hh"
#include "api/Generic.hh"
#include "api/Stream.hh"
#include "api/Validator.hh"

namespace tensorflow {
namespace data {
namespace {

class DecodeJSONOp : public OpKernel {
 public:
  explicit DecodeJSONOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
  }

  void Compute(OpKernelContext* context) override {
    // TODO: support batch (1-D) input
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<string>()();

    const Tensor* names_tensor;
    OP_REQUIRES_OK(context, context->input("names", &names_tensor));

    OP_REQUIRES(context, (names_tensor->NumElements() == shapes_.size()),
                errors::InvalidArgument(
                    "shapes and names should have same number: ",
                    shapes_.size(), " vs. ", names_tensor->NumElements()));
    rapidjson::Document d;
    d.Parse(input.c_str());
    OP_REQUIRES(context, d.IsObject(),
                errors::InvalidArgument("not a valid JSON object"));
    for (size_t i = 0; i < shapes_.size(); i++) {
      Tensor* value_tensor;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, shapes_[i], &value_tensor));
      rapidjson::Value* entry =
          rapidjson::Pointer(names_tensor->flat<string>()(i).c_str()).Get(d);
      OP_REQUIRES(context, (entry != nullptr),
                  errors::InvalidArgument("no value for ",
                                          names_tensor->flat<string>()(i)));
      if (entry->IsArray()) {
        OP_REQUIRES(context, entry->Size() == value_tensor->NumElements(),
                    errors::InvalidArgument(
                        "number of elements in JSON does not match spec: ",
                        entry->Size(), " vs. ", value_tensor->NumElements()));

        switch (value_tensor->dtype()) {
          case DT_INT32:
            for (int64 j = 0; j < entry->Size(); j++) {
              value_tensor->flat<int32>()(j) = (*entry)[j].GetInt();
            }
            break;
          case DT_INT64:
            for (int64 j = 0; j < entry->Size(); j++) {
              value_tensor->flat<int64>()(j) = (*entry)[j].GetInt64();
            }
            break;
          case DT_FLOAT:
            for (int64 j = 0; j < entry->Size(); j++) {
              value_tensor->flat<float>()(j) = (*entry)[j].GetDouble();
            }
            break;
          case DT_DOUBLE:
            for (int64 j = 0; j < entry->Size(); j++) {
              value_tensor->flat<double>()(j) = (*entry)[j].GetDouble();
            }
            break;
          case DT_STRING:
            for (int64 j = 0; j < entry->Size(); j++) {
              value_tensor->flat<string>()(j) = (*entry)[j].GetString();
            }
            break;
          default:
            OP_REQUIRES(
                context, false,
                errors::InvalidArgument("data type not supported: ",
                                        DataTypeString(value_tensor->dtype())));
            break;
        }

      } else {
        switch (value_tensor->dtype()) {
          case DT_INT32:
            value_tensor->scalar<int32>()() = entry->GetInt();
            break;
          case DT_INT64:
            value_tensor->scalar<int64>()() = entry->GetInt64();
            break;
          case DT_FLOAT:
            value_tensor->scalar<float>()() = entry->GetDouble();
            break;
          case DT_DOUBLE:
            value_tensor->scalar<double>()() = entry->GetDouble();
            break;
          case DT_STRING:
            value_tensor->scalar<string>()() = entry->GetString();
            break;
          default:
            OP_REQUIRES(
                context, false,
                errors::InvalidArgument("data type not supported: ",
                                        DataTypeString(value_tensor->dtype())));
            break;
        }
      }
    }
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::vector<TensorShape> shapes_ GUARDED_BY(mu_);
};

class DecodeAvroOp : public OpKernel {
 public:
  explicit DecodeAvroOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
  }

  void Compute(OpKernelContext* context) override {
    // TODO: support batch (1-D) input
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<string>()();

    const Tensor* names_tensor;
    OP_REQUIRES_OK(context, context->input("names", &names_tensor));

    OP_REQUIRES(context, (names_tensor->NumElements() == shapes_.size()),
                errors::InvalidArgument(
                    "shapes and names should have same number: ",
                    shapes_.size(), " vs. ", names_tensor->NumElements()));

    const Tensor* schema_tensor;
    OP_REQUIRES_OK(context, context->input("schema", &schema_tensor));
    const string& schema = schema_tensor->scalar<string>()();

    std::unordered_map<string, Tensor*> values;
    for (int64 i = 0; i < names_tensor->NumElements(); i++) {
      Tensor* value_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(static_cast<int64>(i),
                                                       input_tensor->shape(),
                                                       &value_tensor));
      values[names_tensor->flat<string>()(i)] = value_tensor;
    }
    avro::ValidSchema avro_schema;
    std::istringstream ss(schema);
    string error;
    OP_REQUIRES(context, (avro::compileJsonSchema(ss, avro_schema, error)),
                errors::Unimplemented("Avro schema error: ", error));

    avro::GenericDatum datum(avro_schema);
    const string& entry = input;
    std::unique_ptr<avro::InputStream> in =
        avro::memoryInputStream((const uint8_t*)entry.data(), entry.size());

    avro::DecoderPtr d = avro::binaryDecoder();
    d->init(*in);
    avro::decode(*d, datum);

    OP_REQUIRES_OK(context, ProcessEntry(values, "", datum));
  }
  Status ProcessEntry(std::unordered_map<string, Tensor*>& values,
                      const string& name, const avro::GenericDatum& datum) {
    switch (datum.type()) {
      case avro::AVRO_BOOL:
      case avro::AVRO_INT:
      case avro::AVRO_LONG:
      case avro::AVRO_FLOAT:
      case avro::AVRO_DOUBLE:
      case avro::AVRO_STRING:
      case avro::AVRO_BYTES:
      case avro::AVRO_FIXED:
      case avro::AVRO_ENUM:
        return ProcessPrimitive(values, name, datum);
      case avro::AVRO_RECORD:
        return ProcessRecord(values, name, datum);
      default:
        return errors::InvalidArgument("data type not supported: ",
                                       datum.type());
    }

    return Status::OK();
  }

  Status ProcessRecord(std::unordered_map<string, Tensor*>& values,
                       const string& name, const avro::GenericDatum& datum) {
    const avro::GenericRecord& record = datum.value<avro::GenericRecord>();
    for (size_t i = 0; i < record.fieldCount(); i++) {
      string entry = name + "/" + record.schema()->nameAt(i);
      const avro::GenericDatum& field = record.fieldAt(i);
      TF_RETURN_IF_ERROR(ProcessEntry(values, entry, field));
    }
    return Status::OK();
  }
  Status ProcessPrimitive(std::unordered_map<string, Tensor*>& values,
                          const string& name, const avro::GenericDatum& datum) {
    std::unordered_map<string, Tensor*>::const_iterator lookup =
        values.find(name);
    if (lookup == values.end()) {
      return errors::InvalidArgument("unable to find: ", name);
    }
    Tensor* value_tensor = lookup->second;
    switch (datum.type()) {
      case avro::AVRO_BOOL:
        value_tensor->scalar<bool>()() = datum.value<bool>();
        break;
      case avro::AVRO_INT:
        value_tensor->scalar<int32>()() = datum.value<int32_t>();
        break;
      case avro::AVRO_LONG:
        value_tensor->scalar<int64>()() = datum.value<int64_t>();
        break;
      case avro::AVRO_FLOAT:
        value_tensor->scalar<float>()() = datum.value<float>();
        break;
      case avro::AVRO_DOUBLE:
        value_tensor->scalar<double>()() = datum.value<double>();
        break;
      case avro::AVRO_STRING: {
        // make a concrete explicit copy as otherwise avro may override the
        // underlying buffer.
        const string& datum_value = datum.value<string>();
        value_tensor->scalar<string>()().resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&value_tensor->scalar<string>()()[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_BYTES: {
        const std::vector<uint8_t>& datum_value =
            datum.value<std::vector<uint8_t>>();
        value_tensor->scalar<string>()().resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&value_tensor->scalar<string>()()[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_FIXED: {
        const std::vector<uint8_t>& datum_value =
            datum.value<avro::GenericFixed>().value();
        value_tensor->scalar<string>()().resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&value_tensor->scalar<string>()()[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_ENUM:
        value_tensor->scalar<string>()() =
            datum.value<avro::GenericEnum>().symbol();
        break;
      default:
        return errors::InvalidArgument("data type not supported: ",
                                       datum.type());
    }
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::vector<TensorShape> shapes_ GUARDED_BY(mu_);
};

class EncodeAvroOp : public OpKernel {
 public:
  explicit EncodeAvroOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* names_tensor;
    OP_REQUIRES_OK(context, context->input("names", &names_tensor));

    // Make sure input and names have the same elements
    OP_REQUIRES(
        context, (context->num_inputs() - 2 == names_tensor->NumElements()),
        errors::InvalidArgument("number of elements different: inputs (",
                                context->num_inputs() - 2, ") vs. names(",
                                names_tensor->NumElements(), ")"));

    // Make sure input have the same elements;
    for (int64 i = 1; i < context->num_inputs() - 2; i++) {
      OP_REQUIRES(
          context,
          (context->input(0).NumElements() == context->input(i).NumElements()),
          errors::InvalidArgument("number of elements different: input 0 (",
                                  context->input(0).NumElements(),
                                  ") vs. input ", i, " (",
                                  context->input(i).NumElements(), ")"));
    }

    std::unordered_map<string, const Tensor*> values;
    for (int64 i = 0; i < names_tensor->NumElements(); i++) {
      values[names_tensor->flat<string>()(i)] = &context->input(i);
    }

    const Tensor* schema_tensor;
    OP_REQUIRES_OK(context, context->input("schema", &schema_tensor));
    const string& schema = schema_tensor->scalar<string>()();

    avro::ValidSchema avro_schema;
    std::istringstream ss(schema);
    string error;
    OP_REQUIRES(context, (avro::compileJsonSchema(ss, avro_schema, error)),
                errors::Unimplemented("Avro schema error: ", error));

    Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, context->input(0).shape(), &value_tensor));

    for (int64 entry_index = 0; entry_index < context->input(0).NumElements();
         entry_index++) {
      std::ostringstream ss;
      std::unique_ptr<avro::OutputStream> o = avro::ostreamOutputStream(ss);

      avro::GenericDatum datum(avro_schema);
      OP_REQUIRES_OK(context, ProcessEntry(values, "", datum));

      //      avro::GenericRecord& record = datum.value<avro::GenericRecord>();

      avro::EncoderPtr e = avro::binaryEncoder();
      e->init(*o);
      avro::encode(*e, datum);
      o->flush();
      value_tensor->flat<string>()(entry_index) = ss.str();
    }
  }

  Status ProcessEntry(const std::unordered_map<string, const Tensor*>& values,
                      const string& name, avro::GenericDatum& datum) {
    switch (datum.type()) {
      case avro::AVRO_BOOL:
      case avro::AVRO_INT:
      case avro::AVRO_LONG:
      case avro::AVRO_FLOAT:
      case avro::AVRO_DOUBLE:
      case avro::AVRO_STRING:
      case avro::AVRO_BYTES:
      case avro::AVRO_FIXED:
      case avro::AVRO_ENUM:
        return ProcessPrimitive(values, name, datum);
      // case avro::AVRO_RECORD:
      //  return ProcessRecord(values, name, datum);
      default:
        return errors::InvalidArgument("data type not supported: ",
                                       datum.type());
    }

    return Status::OK();
  }

  Status ProcessRecord(const std::unordered_map<string, const Tensor*>& values,
                       const string& name, avro::GenericDatum& datum) {
    avro::GenericRecord& record = datum.value<avro::GenericRecord>();
    for (size_t i = 0; i < record.fieldCount(); i++) {
      string entry = name + "/" + record.schema()->nameAt(i);
      avro::GenericDatum& field = record.fieldAt(i);
      TF_RETURN_IF_ERROR(ProcessEntry(values, entry, field));
    }
    return Status::OK();
  }
  Status ProcessPrimitive(
      const std::unordered_map<string, const Tensor*>& values,
      const string& name, avro::GenericDatum& datum) {
    std::unordered_map<string, const Tensor*>::const_iterator lookup =
        values.find(name);
    if (lookup == values.end()) {
      return errors::InvalidArgument("unable to find: ", name);
    }
    const Tensor* value_tensor = lookup->second;
    switch (datum.type()) {
      case avro::AVRO_BOOL:
        datum.value<bool>() = value_tensor->scalar<bool>()();
        break;
      case avro::AVRO_INT:
        datum.value<int32_t>() = value_tensor->scalar<int32>()();
        break;
      case avro::AVRO_LONG:
        datum.value<int64_t>() = value_tensor->scalar<int64>()();
        break;
      case avro::AVRO_FLOAT:
        datum.value<float>() = value_tensor->scalar<float>()();
        break;
      case avro::AVRO_DOUBLE:
        datum.value<double>() = value_tensor->scalar<double>()();
        break;
      case avro::AVRO_STRING: {
        // make a concrete explicit copy as otherwise avro may override the
        // underlying buffer.
        const string& datum_value = value_tensor->scalar<string>()();
        datum.value<string>().resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&datum.value<string>()[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_BYTES: {
        const string& datum_value = value_tensor->scalar<string>()();
        datum.value<std::vector<uint8_t>>().resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&datum.value<std::vector<uint8_t>>()[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_FIXED: {
        const string& datum_value = value_tensor->scalar<string>()();
        datum.value<avro::GenericFixed>().value().resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&datum.value<avro::GenericFixed>().value()[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_ENUM:
        datum.value<avro::GenericEnum>().set(value_tensor->scalar<string>()());
        break;
      default:
        return errors::InvalidArgument("data type not supported: ",
                                       datum.type());
    }
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>DecodeJSON").Device(DEVICE_CPU), DecodeJSONOp);
REGISTER_KERNEL_BUILDER(Name("IO>DecodeAvroV").Device(DEVICE_CPU),
                        DecodeAvroOp);
REGISTER_KERNEL_BUILDER(Name("IO>EncodeAvroV").Device(DEVICE_CPU),
                        EncodeAvroOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
