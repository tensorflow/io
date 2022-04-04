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

#include "api/Compiler.hh"
#include "api/DataFile.hh"
#include "api/Generic.hh"
#include "api/Stream.hh"
#include "api/Validator.hh"
#include "rapidjson/document.h"
#include "rapidjson/pointer.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace data {
namespace {

class DecodeJSONOp : public OpKernel {
 public:
  explicit DecodeJSONOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<tstring>()();

    const Tensor* names_tensor;
    OP_REQUIRES_OK(context, context->input("names", &names_tensor));

    OP_REQUIRES(
        context, (names_tensor->NumElements() == context->num_outputs()),
        errors::InvalidArgument("names should have same number as outputs: ",
                                names_tensor->NumElements(), " vs. ",
                                context->num_outputs()));
    rapidjson::Document d;
    d.Parse(input.c_str());
    OP_REQUIRES(context, d.IsObject(),
                errors::InvalidArgument("not a valid JSON object"));
    for (size_t i = 0; i < names_tensor->NumElements(); i++) {
      rapidjson::Value* entry =
          rapidjson::Pointer(names_tensor->flat<tstring>()(i).c_str()).Get(d);
      OP_REQUIRES(context, (entry != nullptr),
                  errors::InvalidArgument("no value for ",
                                          names_tensor->flat<tstring>()(i)));

      Tensor* value_tensor;
      std::vector<int64> tensor_shape_vector;
      if (entry->IsArray()) {
        getTensorShape(entry, tensor_shape_vector);
      } else {
        tensor_shape_vector.push_back(1);
      }
      OP_REQUIRES_OK(
          context, context->allocate_output(i, TensorShape(tensor_shape_vector),
                                            &value_tensor));
      int64 flat_index;
      flat_index = 0;
      switch (value_tensor->dtype()) {
        case DT_INT32:
          writeToTensor(entry, value_tensor, flat_index, writeInt32);
          break;
        case DT_INT64:
          writeToTensor(entry, value_tensor, flat_index, writeInt64);
          break;
        case DT_FLOAT:
          writeToTensor(entry, value_tensor, flat_index, writeFloat);
          break;
        case DT_DOUBLE:
          writeToTensor(entry, value_tensor, flat_index, writeDouble);
          break;
        case DT_STRING:
          writeToTensor(entry, value_tensor, flat_index, writeString);
          break;
        case DT_BOOL:
          writeToTensor(entry, value_tensor, flat_index, writeBool);
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

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);

  // Tensor Shape

  static void getTensorShape(rapidjson::Value* entry,
                             std::vector<int64>& tensor_shape_vector) {
    if (entry->IsArray()) {
      tensor_shape_vector.push_back(entry->Size());
      if (entry->Size() != 0) {
        getTensorShape(&(*entry)[0], tensor_shape_vector);
      }
    }
  }

  // Single Entry Tensor Writes

  static void writeInt32(rapidjson::Value* entry, Tensor* value_tensor,
                         int64& flat_index) {
    value_tensor->flat<int32>()(flat_index) = (*entry).GetInt();
  }

  static void writeInt64(rapidjson::Value* entry, Tensor* value_tensor,
                         int64& flat_index) {
    value_tensor->flat<int64>()(flat_index) = (*entry).GetInt64();
  }

  static void writeFloat(rapidjson::Value* entry, Tensor* value_tensor,
                         int64& flat_index) {
    value_tensor->flat<float>()(flat_index) = (*entry).GetDouble();
  }

  static void writeDouble(rapidjson::Value* entry, Tensor* value_tensor,
                          int64& flat_index) {
    value_tensor->flat<double>()(flat_index) = (*entry).GetDouble();
  }

  static void writeString(rapidjson::Value* entry, Tensor* value_tensor,
                          int64& flat_index) {
    value_tensor->flat<tstring>()(flat_index) = (*entry).GetString();
  }

  static void writeBool(rapidjson::Value* entry, Tensor* value_tensor,
                        int64& flat_index) {
    value_tensor->flat<bool>()(flat_index) = (*entry).GetBool();
  }

  // Full Tensor Write

  template <class T>
  static void writeToTensor(rapidjson::Value* entry, Tensor* value_tensor,
                            int64& flat_index, T write_func) {
    if (entry->IsArray()) {
      for (int64 i = 0; i < entry->Size(); i++) {
        writeToTensor(&(*entry)[i], value_tensor, flat_index, write_func);
      }
    } else {
      write_func(entry, value_tensor, flat_index);
      flat_index++;
    }
  }
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

    const Tensor* names_tensor;
    OP_REQUIRES_OK(context, context->input("names", &names_tensor));

    OP_REQUIRES(context, (names_tensor->NumElements() == shapes_.size()),
                errors::InvalidArgument(
                    "shapes and names should have same number: ",
                    shapes_.size(), " vs. ", names_tensor->NumElements()));

    const Tensor* schema_tensor;
    OP_REQUIRES_OK(context, context->input("schema", &schema_tensor));
    const string& schema = schema_tensor->scalar<tstring>()();

    std::unordered_map<string, Tensor*> values;
    for (int64 i = 0; i < names_tensor->NumElements(); i++) {
      Tensor* value_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(static_cast<int64>(i),
                                                       input_tensor->shape(),
                                                       &value_tensor));
      values[names_tensor->flat<tstring>()(i)] = value_tensor;
    }
    avro::ValidSchema avro_schema;
    std::istringstream ss(schema);
    string error;
    OP_REQUIRES(context, (avro::compileJsonSchema(ss, avro_schema, error)),
                errors::Unimplemented("Avro schema error: ", error));

    for (int64 entry_index = 0; entry_index < context->input(0).NumElements();
         entry_index++) {
      avro::GenericDatum datum(avro_schema);
      const string& entry = input_tensor->flat<tstring>()(entry_index);
      std::unique_ptr<avro::InputStream> in =
          avro::memoryInputStream((const uint8_t*)entry.data(), entry.size());

      avro::DecoderPtr d = avro::binaryDecoder();
      d->init(*in);
      avro::decode(*d, datum);
      OP_REQUIRES_OK(context, ProcessEntry(entry_index, values, "", datum));
    }
  }
  Status ProcessEntry(const int64 index,
                      std::unordered_map<string, Tensor*>& values,
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
        return ProcessPrimitive(index, values, name, datum);
      case avro::AVRO_NULL:
        return ProcessNull(index, values, name, datum);
      case avro::AVRO_RECORD:
        return ProcessRecord(index, values, name, datum);
      default:
        return errors::InvalidArgument("data type not supported: ",
                                       datum.type());
    }

    return Status::OK();
  }

  Status ProcessRecord(const int64 index,
                       std::unordered_map<string, Tensor*>& values,
                       const string& name, const avro::GenericDatum& datum) {
    const avro::GenericRecord& record = datum.value<avro::GenericRecord>();
    for (size_t i = 0; i < record.fieldCount(); i++) {
      string entry = name + "/" + record.schema()->nameAt(i);
      const avro::GenericDatum& field = record.fieldAt(i);
      TF_RETURN_IF_ERROR(ProcessEntry(index, values, entry, field));
    }
    return Status::OK();
  }
  Status ProcessPrimitive(const int64 index,
                          std::unordered_map<string, Tensor*>& values,
                          const string& name, const avro::GenericDatum& datum) {
    std::unordered_map<string, Tensor*>::const_iterator lookup =
        values.find(name);
    if (lookup == values.end()) {
      return errors::InvalidArgument("unable to find: ", name);
    }
    Tensor* value_tensor = lookup->second;
    switch (datum.type()) {
      case avro::AVRO_BOOL:
        value_tensor->flat<bool>()(index) = datum.value<bool>();
        break;
      case avro::AVRO_INT:
        value_tensor->flat<int32>()(index) = datum.value<int32_t>();
        break;
      case avro::AVRO_LONG:
        value_tensor->flat<int64>()(index) = datum.value<int64_t>();
        break;
      case avro::AVRO_FLOAT:
        value_tensor->flat<float>()(index) = datum.value<float>();
        break;
      case avro::AVRO_DOUBLE:
        value_tensor->flat<double>()(index) = datum.value<double>();
        break;
      case avro::AVRO_STRING: {
        // make a concrete explicit copy as otherwise avro may override the
        // underlying buffer.
        const string& datum_value = datum.value<string>();
        value_tensor->flat<tstring>()(index).resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&value_tensor->flat<tstring>()(index)[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_BYTES: {
        const std::vector<uint8_t>& datum_value =
            datum.value<std::vector<uint8_t>>();
        value_tensor->flat<tstring>()(index).resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&value_tensor->flat<tstring>()(index)[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_FIXED: {
        const std::vector<uint8_t>& datum_value =
            datum.value<avro::GenericFixed>().value();
        value_tensor->flat<tstring>()(index).resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&value_tensor->flat<tstring>()(index)[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_ENUM:
        value_tensor->flat<tstring>()(index) =
            datum.value<avro::GenericEnum>().symbol();
        break;
      default:
        return errors::InvalidArgument("data type not supported: ",
                                       datum.type());
    }
    return Status::OK();
  }
  Status ProcessNull(const int64 index,
                     std::unordered_map<string, Tensor*>& values,
                     const string& name, const avro::GenericDatum& datum) {
    std::unordered_map<string, Tensor*>::const_iterator lookup =
        values.find(name);
    if (lookup == values.end()) {
      return errors::InvalidArgument("unable to find: ", name);
    }
    Tensor* value_tensor = lookup->second;
    switch (value_tensor->dtype()) {
      case DT_BOOL:
        value_tensor->flat<bool>()(index) = false;
        break;
      case DT_INT32:
        value_tensor->flat<int32>()(index) = 0;
        break;
      case DT_INT64:
        value_tensor->flat<int64>()(index) = 0;
        break;
      case DT_FLOAT:
        value_tensor->flat<float>()(index) = 0.0;
        break;
      case DT_DOUBLE:
        value_tensor->flat<double>()(index) = 0;
        break;
      case DT_STRING:
        value_tensor->flat<tstring>()(index) = "";
        break;
      default:
        return errors::InvalidArgument("data type not supported: ",
                                       value_tensor->dtype());
    }
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::vector<TensorShape> shapes_ TF_GUARDED_BY(mu_);
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
      values[names_tensor->flat<tstring>()(i)] = &context->input(i);
    }

    const Tensor* schema_tensor;
    OP_REQUIRES_OK(context, context->input("schema", &schema_tensor));
    const string& schema = schema_tensor->scalar<tstring>()();

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
      OP_REQUIRES_OK(context, ProcessEntry(entry_index, values, "", datum));

      avro::EncoderPtr e = avro::binaryEncoder();
      e->init(*o);
      avro::encode(*e, datum);
      o->flush();
      value_tensor->flat<tstring>()(entry_index) = ss.str();
    }
  }

  Status ProcessEntry(const int64 index,
                      const std::unordered_map<string, const Tensor*>& values,
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
        return ProcessPrimitive(index, values, name, datum);
      case avro::AVRO_RECORD:
        return ProcessRecord(index, values, name, datum);
      default:
        return errors::InvalidArgument("data type not supported: ",
                                       datum.type());
    }

    return Status::OK();
  }

  Status ProcessRecord(const int64 index,
                       const std::unordered_map<string, const Tensor*>& values,
                       const string& name, avro::GenericDatum& datum) {
    avro::GenericRecord& record = datum.value<avro::GenericRecord>();
    for (size_t i = 0; i < record.fieldCount(); i++) {
      string entry = name + "/" + record.schema()->nameAt(i);
      avro::GenericDatum& field = record.fieldAt(i);
      TF_RETURN_IF_ERROR(ProcessEntry(index, values, entry, field));
    }
    return Status::OK();
  }
  Status ProcessPrimitive(
      const int64 index,
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
        datum.value<bool>() = value_tensor->flat<bool>()(index);
        break;
      case avro::AVRO_INT:
        datum.value<int32_t>() = value_tensor->flat<int32>()(index);
        break;
      case avro::AVRO_LONG:
        datum.value<int64_t>() = value_tensor->flat<int64>()(index);
        break;
      case avro::AVRO_FLOAT:
        datum.value<float>() = value_tensor->flat<float>()(index);
        break;
      case avro::AVRO_DOUBLE:
        datum.value<double>() = value_tensor->flat<double>()(index);
        break;
      case avro::AVRO_STRING: {
        // make a concrete explicit copy as otherwise avro may override the
        // underlying buffer.
        const string& datum_value = value_tensor->flat<tstring>()(index);
        datum.value<string>().resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&datum.value<string>()[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_BYTES: {
        const string& datum_value = value_tensor->flat<tstring>()(index);
        datum.value<std::vector<uint8_t>>().resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&datum.value<std::vector<uint8_t>>()[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_FIXED: {
        const string& datum_value = value_tensor->flat<tstring>()(index);
        datum.value<avro::GenericFixed>().value().resize(datum_value.size());
        if (datum_value.size() > 0) {
          memcpy(&datum.value<avro::GenericFixed>().value()[0], &datum_value[0],
                 datum_value.size());
        }
      } break;
      case avro::AVRO_ENUM:
        datum.value<avro::GenericEnum>().set(
            value_tensor->flat<tstring>()(index));
        break;
      default:
        return errors::InvalidArgument("data type not supported: ",
                                       datum.type());
    }
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>DecodeJSON").Device(DEVICE_CPU), DecodeJSONOp);
REGISTER_KERNEL_BUILDER(Name("IO>DecodeAvro").Device(DEVICE_CPU), DecodeAvroOp);
REGISTER_KERNEL_BUILDER(Name("IO>EncodeAvro").Device(DEVICE_CPU), EncodeAvroOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
