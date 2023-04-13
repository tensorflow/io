#ifndef TENSORFLOW_DATA_CORE_KERNELS_AVRO_ATDS_DECODER_BASE_H_
#define TENSORFLOW_DATA_CORE_KERNELS_AVRO_ATDS_DECODER_BASE_H_

#include "api/Decoder.hh"
#include "api/GenericDatum.hh"
#include "api/Node.hh"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow_io/core/kernels/avro/atds/sparse_value_buffer.h"

namespace tensorflow {
namespace atds {

enum class FeatureType { dense, sparse, varlen, opaque_contextual };

static const std::map<avro::Type, DataType> avro_to_tf_datatype = {
    {avro::AVRO_INT, DT_INT32},     {avro::AVRO_LONG, DT_INT64},
    {avro::AVRO_STRING, DT_STRING}, {avro::AVRO_BYTES, DT_STRING},
    {avro::AVRO_FLOAT, DT_FLOAT},   {avro::AVRO_DOUBLE, DT_DOUBLE},
    {avro::AVRO_BOOL, DT_BOOL}};

/*
 * Decoders decode avro features into Tensors.
 * All decoder implementations must implement the operator overload '()'.
 * Decoders are invoked in a multithreaded context(controlled by
 * `num_parallel_calls`). Therefore the implementations must be threadsafe.
 * TODO: Add static analysis to check thread-safety(BDP-7562)
 * */
class DecoderBase {
 public:
  virtual ~DecoderBase() {}

  virtual Status operator()(avro::DecoderPtr&, std::vector<Tensor>&,
                            sparse::ValueBuffer&,
                            std::vector<avro::GenericDatum>&, size_t) = 0;
};

/*
 * Template Metadata class must implement the following public members.
 *   FeatureType type
 *   string name
 *   DataType dtype
 *   PartialTensorShape shape
 * */
template <typename Metadata>
std::unique_ptr<DecoderBase> CreateFeatureDecoder(const avro::NodePtr&,
                                                  const Metadata&);

template <typename Metadata>
Status ValidateSchema(const avro::NodePtr&, const Metadata&);

}  // namespace atds
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_CORE_KERNELS_AVRO_ATDS_DECODER_BASE_H_
