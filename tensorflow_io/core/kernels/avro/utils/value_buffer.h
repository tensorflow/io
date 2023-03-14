/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#ifndef TENSORFLOW_DATA_VALUE_BUFFER_H_
#define TENSORFLOW_DATA_VALUE_BUFFER_H_

#include <sstream>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {

// Constants in namespace data for begin and finish mark
static constexpr size_t kBeginMark = std::numeric_limits<size_t>::max() - 1;
static constexpr size_t kFinishMark = std::numeric_limits<size_t>::max();

// An abstract representation of a store for values
// This representation preserves the order in which elements have been added
// Uses begin and finish marks around elements to infer the full shape of an N-D
// tensor
class ValueStore;
using ValueStoreUniquePtr = std::unique_ptr<ValueStore>;
class ValueStore {
 public:
  // Virtual destructor ensures the derived class's destructor is called and
  // clean up its memory.
  virtual ~ValueStore() = default;

  // Make a dense tensor from this value store given the default values and the
  // resolved shape Assumes the tensor has been initialized and allocated!
  virtual Status MakeDense(Tensor* tensor, const TensorShape& resolved_shape,
                           const Tensor& defaults) const = 0;

  // Make a sparse tensor with values and indices from this value store
  // Assumes the tensor has been initialized and allocated!
  virtual Status MakeSparse(Tensor* values, Tensor* indices) const = 0;

  // Resolve a shape given a partial shape from the user and a shape from the
  // defaults
  // TODO(fraudies): Remove once avro dataset is gone
  virtual Status ResolveDenseShape(TensorShape* shape,
                                   const PartialTensorShape& partial_shape,
                                   const TensorShape& default_shape) const = 0;

  // Resolve the user shape, given the batch_size, default_shape, and data shape
  // In particular checks that
  // - batch_size + user shape is compatible with non-scalar default shapes
  // - batch_size + user shape is compatible with data shape
  virtual Status ResolveDenseShapeWithBatch(
      TensorShape* shape, const PartialTensorShape& user_shape,
      const TensorShape& default_shape, size_t batch_size) const = 0;

  // Get the shape for sparse values if this value store where represented as
  // sparse tensor
  virtual Status GetSparseValueShape(TensorShape* shape) const = 0;

  // Get the shape for sparse indices if this value store where represented as
  // sparse tensor
  virtual Status GetSparseIndexShape(TensorShape* shape) const = 0;

  // Get the maximum index for each dimension and assign it to the dense_shape
  virtual Status GetDenseShapeForSparse(Tensor* dense_shape) const = 0;

  // Check if values match at the reverse index between this store and the
  // provided one
  virtual bool ValuesMatchAtReverseIndex(const ValueStore& store,
                                         size_t reverse_index) const = 0;

  // Check if the value matches at the reverse index
  virtual bool ValueMatchesAtReverseIndex(const tstring& value,
                                          size_t reverse_index) const = 0;

  // Is empty when this store has no values
  virtual bool IsEmpty() const = 0;

  // Place a begin mark
  virtual void BeginMark() = 0;

  // Place a finish mark
  virtual void FinishMark() = 0;

  // Output a human readable string representation with limit number of elements
  virtual string ToString(size_t limit = 10) const = 0;
};

// The shape builder keeps track of the number of values each dimension has
// Assumes that values are added in sequence
class ShapeBuilder {
 public:
  ShapeBuilder();

  virtual ~ShapeBuilder() = default;

  // Places a begin mark
  void BeginMark();

  // Places a finish mark
  void FinishMark();

  // Increment the counter for the elements
  void Increment();

  // Get the number of dimensions
  size_t GetNumberOfDimensions() const;

  // Get the dense shape for this shape builder
  void GetDenseShape(TensorShape* shape) const;

  // True if the shape builder registered all elements for the given shape
  bool HasAllElements(const TensorShape& shape) const;

  // Get the copy information for the given tensor output shape
  Status GetCopyInfo(std::vector<std::pair<size_t, size_t> >* copy_info,
                     const TensorShape& output_shape) const;

  // Get the fill information for the given tensor output shape
  Status GetFillInfo(std::vector<std::pair<size_t, size_t> >* fill_info,
                     const TensorShape& output_shape) const;

  // Get the indices as tensor for this shape
  Status GetIndices(Tensor* indices) const;

  // Get a human readable string for this shape builder
  string ToString() const;

  // Merge this shape builder with another. This assumes that both this and
  // the other shape builder are valid and initialized
  void Merge(const ShapeBuilder& other);

 private:
  // Reconcile the output shape of a tensor with the shape within this shape
  // builder Used by copy and fill info This will handle flattening, where the
  // user did not provide a shape but the batch shape defines fully (e.g. batch
  // size = 3). Then the output shape is [3] but the shape builder parsed this
  // as [3 1] one data point per entry. This method will reconcile the shape to
  // [3 1] If we use this parser to patch assume the user provided a shape of
  // [3, 2] but the shape builder only had one entry per row then their shape
  // will be [3, 1]. This will reconcile the shapes to [3, 2]
  void ReconcileShape(TensorShape* reconciled, const TensorShape& shape) const;

  // Get the cumulative product of dimensions without the last one
  std::vector<size_t> CumulativeProductOfDimensionsWithOneAtEnd(
      const TensorShape& shape) const;

  // A shape buffer is scalar if its length is <= 3, these are B ? F and B F
  // (others are invalid)
  inline bool IsScalar() const { return element_info_.size() == 3; }

  // Contains begin marks, finish marks, and the element counter for each
  // dimension
  std::vector<size_t> element_info_;

  // intermediate counter used when creating a buffer
  size_t element_counter_;

  // True if we had a begin mark otherwise false
  bool has_begin_;
};

// The value buffer holds the actual values and implements a value store
template <typename T>
class ValueBuffer : public ValueStore {
 public:
  ValueBuffer() = default;
  ValueBuffer(const std::vector<ValueStoreUniquePtr>& others);

  virtual ~ValueBuffer() = default;

  inline void BeginMark() override { shape_builder_.BeginMark(); }

  inline void FinishMark() override { shape_builder_.FinishMark(); }

  // Add a primitive value (e.g. bool, int) to the buffer by copy
  inline void Add(T value) {
    values_.push_back(value);
    shape_builder_.Increment();
  }

  // Add a non-primitive value (e.g. string) by reference to the buffer
  inline void AddByRef(const T& value) {
    values_.push_back(value);
    shape_builder_.Increment();
  }

  // Return the last item in the buffer
  inline const T back() const { return values_.back(); }

  // Index in reverse order, index 1 indexes the last element
  inline const T ReverseIndex(size_t index) const {
    return values_[values_.size() - index];
  }

  // Resolve the partial shape provided by the user into a fully defined shape
  // for a dense tensor
  // TODO(fraudies) Deprecated, remove once make_avro_dataset is gone
  Status ResolveDenseShape(TensorShape* shape,
                           const PartialTensorShape& partial_shape,
                           const TensorShape& default_shape) const override;

  Status ResolveDenseShapeWithBatch(TensorShape* shape,
                                    const PartialTensorShape& user_shape,
                                    const TensorShape& default_shape,
                                    size_t batch_size) const override;

  Status GetSparseValueShape(TensorShape* shape) const override;

  Status GetSparseIndexShape(TensorShape* shape) const override;

  Status GetDenseShapeForSparse(Tensor* dense_shape) const override;

  Status MakeDense(Tensor* tensor, const TensorShape& resolved_shape,
                   const Tensor& defaults) const override;

  Status MakeSparse(Tensor* values, Tensor* indices) const override;

  virtual bool ValuesMatchAtReverseIndex(const ValueStore& store,
                                         size_t reverse_index) const override;

  virtual bool ValueMatchesAtReverseIndex(const tstring& value,
                                          size_t reverse_index) const override;

  inline bool IsEmpty() const override { return GetNumberOfElements() == 0; }

  virtual string ToString(size_t limit) const override;

 private:
  // Returns the number of elements
  inline size_t GetNumberOfElements() const { return values_.size(); }

  // Fill in from the buffer Assumes tensor has been initialized
  Status FillInFromBuffer(Tensor* tensor) const;

  // Assumes tensor has been initialized
  Status FillInFromDefault(Tensor* tensor, const Tensor& defaults) const;

  // Is empty shape for this partial shape
  inline static bool IsEmptyShape(const PartialTensorShape& partial_shape) {
    return partial_shape.dims() < 1;
  }

  // Is this a one element tensor of shape 1 with one value
  inline static bool IsOneElementTensor(const TensorShape& tensor_shape) {
    return tensor_shape.dims() == 1 && tensor_shape.dim_size(0) == 1;
  }

  // Is non trivial tensor that has >= 1 dimension and the dimension(0) > 1
  // value
  inline static bool IsNonTrivialTensor(const TensorShape& tensor_shape) {
    VLOG(15) << "Checking if " << tensor_shape << " is non-trivial";
    // Check that any of the dimensions is > 1
    for (size_t i_dim = 0; i_dim < tensor_shape.dims(); ++i_dim) {
      VLOG(15) << "Dimension " << i_dim << " is "
               << tensor_shape.dim_size(i_dim);
      if (tensor_shape.dim_size(i_dim) > 1) {
        return true;
      }
    }
    return false;
  }

  // For up to 4 values use inline memory
  gtl::InlinedVector<T, 4> values_;

  // The shape builder for this value buffer
  ShapeBuilder shape_builder_;
};

// -------------------------------------------------------------------------------------------------
// Template specializations for value buffer
// -------------------------------------------------------------------------------------------------
typedef ValueBuffer<bool> BoolValueBuffer;
typedef ValueBuffer<int32> IntValueBuffer;
typedef ValueBuffer<int64> LongValueBuffer;
typedef ValueBuffer<float> FloatValueBuffer;
typedef ValueBuffer<double> DoubleValueBuffer;
typedef ValueBuffer<tstring> StringValueBuffer;

// Unfortunately, need to provide type information for casting
// Note, this code has not been designed with merge in scope
Status MergeAs(ValueStoreUniquePtr& merged,
               const std::vector<ValueStoreUniquePtr>& buffers, DataType dtype);

// -------------------------------------------------------------------------------------------------
// copy or move data depending on the data type
// -------------------------------------------------------------------------------------------------
template <typename InputIterT, typename OutputIterT>
inline void CopyOrMoveBlock(const InputIterT b, const InputIterT e,
                            OutputIterT t) {
  std::copy(b, e, t);
}
template <>
inline void CopyOrMoveBlock(const string* b, const string* e, string* t) {
  std::move(b, e, t);
}

// -------------------------------------------------------------------------------------------------
// Implementation of the value buffer
// -------------------------------------------------------------------------------------------------
template <typename T>
ValueBuffer<T>::ValueBuffer(const std::vector<ValueStoreUniquePtr>& others) {
  // Compute the total number of elements
  size_t n_total = 0;
  for (size_t i = 0; i < others.size(); ++i) {
    ValueBuffer<T>* buffer = reinterpret_cast<ValueBuffer<T>*>(others[i].get());
    n_total += buffer->values_.size();
  }
  values_.resize(n_total);
  VLOG(5) << "Allocate space for " << n_total << " elements in buffer";

  // Get the target start location
  auto target = values_.begin();
  for (size_t i = 0; i < others.size(); ++i) {
    ValueBuffer<T>* buffer = reinterpret_cast<ValueBuffer<T>*>(others[i].get());
    auto begin = buffer->values_.begin();
    size_t n_elements = buffer->values_.size();
    CopyOrMoveBlock(begin, begin + n_elements, target);
    target += n_elements;
    shape_builder_.Merge(buffer->shape_builder_);
  }
}

template <typename T>
Status ValueBuffer<T>::ResolveDenseShapeWithBatch(
    TensorShape* shape, const PartialTensorShape& user_shape,
    const TensorShape& default_shape, size_t batch_size) const {
  // Assume non-trivial default shape is consistent with user shape
  // Assume that all items have the full batch size TODO(fraudies) Make true for
  // null values
  if (IsNonTrivialTensor(default_shape)) {
    // Note, data shape does not have to be compatible because we will pad
    *shape = default_shape;
  } else {
    // Default is trivial, get shape from data and ensure it's consistent with
    // users batched shape
    TensorShape data_shape;
    shape_builder_.GetDenseShape(&data_shape);
    PartialTensorShape batched_user_shape(
        PartialTensorShape({static_cast<long long>(batch_size)})
            .Concatenate(user_shape));
    if (!batched_user_shape.IsCompatibleWith(data_shape)) {
      return errors::InvalidArgument(
          "Batched user shape", batched_user_shape,
          " is incompatible with data shape: ", *shape);
    }
    *shape = data_shape;
  }
  return OkStatus();
}

// TODO(fraudies): Move validation of user defined shape and defaults into the
// avro dataset To resolve the proper shape for a dense tensor we honor: 1st the
// user provided partial shape 2nd the user provided defaults 3rd the value
// buffer's end indices In that order!
template <typename T>
Status ValueBuffer<T>::ResolveDenseShape(
    TensorShape* shape, const PartialTensorShape& user_shape,
    const TensorShape& default_shape) const {
  bool defaultIsNonTrivialTensor = IsNonTrivialTensor(default_shape);

  // Honor user defined shape if fully defined
  if (user_shape.IsFullyDefined() && user_shape.AsTensorShape(shape) &&
      IsNonTrivialTensor(*shape)) {
    VLOG(3) << "Fully defined input shape";

    // If the default is not scalar
  } else if (defaultIsNonTrivialTensor) {
    VLOG(3) << "Default is non trivial tensor";

    PartialTensorShape tmp_shape;
    // Honor any partially defined shape from user and supplement with that from
    // default
    if (user_shape.MergeWith(default_shape, &tmp_shape) == OkStatus()) {
      // Merged convert partial shape into shape
      if (!tmp_shape.AsTensorShape(shape)) {
        return errors::InvalidArgument("Expected ", tmp_shape,
                                       " to be fully defined"
                                       " and convertible into a dense shape.");
      }
    } else {
      // Could not merge, then use default
      *shape = default_shape;
    }

    // If the shape is not defined by the user nor the default, infer from
    // provided data
  } else {
    TensorShape dense_shape;
    shape_builder_.GetDenseShape(&dense_shape);

    VLOG(3) << "Get dense shape from data " << dense_shape;

    PartialTensorShape tmp_shape;
    // Honor any partially defined shape from user and supplement with that from
    // data
    if (user_shape.MergeWith(dense_shape, &tmp_shape) == OkStatus()) {
      if (!tmp_shape.AsTensorShape(shape)) {
        return errors::InvalidArgument("Expected ", tmp_shape,
                                       " to be fully defined"
                                       " and convertible into a dense shape.");
      }
    } else {
      // Could not merge, then use dense shape
      *shape = dense_shape;
    }
  }

  return OkStatus();
}

template <typename T>
Status ValueBuffer<T>::GetSparseValueShape(TensorShape* shape) const {
  (*shape).AddDim(GetNumberOfElements());
  return OkStatus();
}

template <typename T>
Status ValueBuffer<T>::GetSparseIndexShape(TensorShape* shape) const {
  (*shape).AddDim(GetNumberOfElements());
  size_t n_dim = shape_builder_.GetNumberOfDimensions();
  // Only add this dimension if it is necessary, which is > 1
  if (n_dim > 1) {
    (*shape).AddDim(shape_builder_.GetNumberOfDimensions());
  }
  return OkStatus();
}

// Assumes dense_shape has been allocated appropriate space -- not checked
template <typename T>
Status ValueBuffer<T>::GetDenseShapeForSparse(Tensor* dense_shape) const {
  TensorShape shape;
  shape_builder_.GetDenseShape(&shape);
  VLOG(3) << "Dense shape for buffer is: " << shape;

  auto tensor_flat = (*dense_shape).flat<int64>();
  size_t n_dim = shape.dims();
  for (size_t i_dim = 0; i_dim < n_dim; ++i_dim) {
    tensor_flat(i_dim) = shape.dim_size(i_dim);
  }
  return OkStatus();
}

template <typename T>
Status ValueBuffer<T>::MakeDense(Tensor* tensor,
                                 const TensorShape& resolved_shape,
                                 const Tensor& defaults) const {
  // Get the dense shape
  bool doFillFromDefault = !shape_builder_.HasAllElements(resolved_shape);

  // Check that shape matches, with the dimensions
  if (doFillFromDefault) {
    // fill in the default -- note might fill all values
    TF_RETURN_IF_ERROR(FillInFromDefault(tensor, defaults));
  }

  // Fill in the values into the tensor from the buffer
  TF_RETURN_IF_ERROR(FillInFromBuffer(tensor));

  return OkStatus();
}

// Builds the tensor for the values and the indices from the value buffer
// Assumes that values is pre-allocated with space for n_elements
// Assumes that indices is pre-allocated with space for n_elements x n_dim
template <typename T>
Status ValueBuffer<T>::MakeSparse(Tensor* values, Tensor* indices) const {
  // Copy values
  auto tensor_data = (*values).flat<T>().data();
  auto buffer_data = values_.begin();
  size_t n_elements(GetNumberOfElements());
  CopyOrMoveBlock(buffer_data, buffer_data + n_elements, tensor_data);

  // Create indices
  TF_RETURN_IF_ERROR(shape_builder_.GetIndices(indices));

  return OkStatus();
}

template <typename T>
Status ValueBuffer<T>::FillInFromBuffer(Tensor* tensor) const {
  TensorShape shape = (*tensor).shape();
  // shape_builder_.GetDenseShape(&shape); //(*tensor).shape();
  auto tensor_data = (*tensor).flat<T>().data();
  auto buffer_data = values_.begin();

  // These offsets are per fragment of data
  std::vector<std::pair<size_t, size_t> > copy_info;
  TF_RETURN_IF_ERROR(shape_builder_.GetCopyInfo(&copy_info, shape));
  size_t source_offset = 0;
  for (const auto& info : copy_info) {
    size_t target_offset = info.first;
    size_t length = info.second;

    VLOG(3) << "Copy at offset " << source_offset << ": " << length
            << " values to offset " << target_offset;

    CopyOrMoveBlock(buffer_data + source_offset,
                    buffer_data + source_offset + length,
                    tensor_data + target_offset);
    source_offset += length;
  }

  return OkStatus();
}

template <typename T>
Status ValueBuffer<T>::FillInFromDefault(Tensor* tensor,
                                         const Tensor& defaults) const {
  // Don't have any default values
  if (!defaults.IsInitialized()) {
    return errors::InvalidArgument(
        "Need to provide a 'defaults' tensor with values");
  }

  TensorShape shape = (*tensor).shape();
  // TensorShape shape;
  // shape_builder_.GetDenseShape(&shape); //(*tensor).shape();
  auto tensor_data = (*tensor).flat<T>().data();
  auto buffer_data = defaults.flat<T>().data();

  // Defaults is a scalar or one element tensor that need to be initialized
  if (IsOneElementTensor(defaults.shape()) || IsEmptyShape(defaults.shape())) {
    std::fill(tensor_data, tensor_data + shape.num_elements(),
              defaults.flat<T>()(0));
  } else {
    std::vector<std::pair<size_t, size_t> > fill_info;
    TF_RETURN_IF_ERROR(shape_builder_.GetFillInfo(&fill_info, shape));
    for (const auto& info : fill_info) {
      size_t offset = info.first;
      size_t length = info.second;
      CopyOrMoveBlock(buffer_data + offset, buffer_data + offset + length,
                      tensor_data + offset);
    }
  }

  return OkStatus();
}

template <typename T>
bool ValueBuffer<T>::ValuesMatchAtReverseIndex(const ValueStore& store,
                                               size_t reverse_index) const {
  // If both stores are empty, then there is a match
  if (IsEmpty() && store.IsEmpty()) {
    return true;
  }
  // Note, buffer will be nullptr if the types don't match
  const ValueBuffer* buffer = dynamic_cast<const ValueBuffer*>(&store);
  return buffer != nullptr &&
         ReverseIndex(reverse_index) == (*buffer).ReverseIndex(reverse_index);
}

template <typename T>
inline bool ValueBuffer<T>::ValueMatchesAtReverseIndex(
    const tstring& value, size_t reverse_index) const {
  // TODO(fraudies): Check if we want to match other types through parsing of
  // string
  return false;
}

template <>
inline bool ValueBuffer<tstring>::ValueMatchesAtReverseIndex(
    const tstring& value, size_t reverse_index) const {
  // If there is no value to match there is no match
  if (IsEmpty()) {
    return false;
  }
  return ReverseIndex(reverse_index) == value;
}

template <typename T>
string ValueBuffer<T>::ToString(size_t limit) const {
  std::stringstream ss;
  ss << "Shape: " << shape_builder_.ToString() << "Values: ";
  size_t n_print = std::min(values_.size(), limit);
  for (size_t i_print = 0; i_print < n_print; ++i_print) {
    ss << values_[i_print] << ", ";
  }
  if (values_.size() > limit) {
    ss << " ...";
  }
  return ss.str();
}

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALUE_BUFFER_H_
