#ifndef TENSORFLOW_DATA_CORE_KERNELS_AVRO_ATDS_OPAQUE_CONTEXTUAL_FEATURE_DECODER_H_
#define TENSORFLOW_DATA_CORE_KERNELS_AVRO_ATDS_OPAQUE_CONTEXTUAL_FEATURE_DECODER_H_

#include "api/Decoder.hh"
#include "api/Generic.hh"
#include "api/Specific.hh"
#include "tensorflow_io/core/kernels/avro/atds/decoder_base.h"

namespace tensorflow {
namespace atds {
namespace opaque_contextual {

class FeatureDecoder : public DecoderBase {
 public:
  explicit FeatureDecoder(size_t datum_index) : datum_index_(datum_index) {}

  Status operator()(avro::DecoderPtr& decoder,
                    std::vector<Tensor>& dense_tensors,
                    sparse::ValueBuffer& buffer,
                    std::vector<avro::GenericDatum>& skipped_data,
                    size_t offset) {
    avro::decode(*decoder, skipped_data[datum_index_]);
    return OkStatus();
  }

 private:
  const size_t datum_index_;
};

}  // namespace opaque_contextual
}  // namespace atds
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_CORE_KERNELS_AVRO_ATDS_OPAQUE_CONTEXTUAL_FEATURE_DECODER_H_
