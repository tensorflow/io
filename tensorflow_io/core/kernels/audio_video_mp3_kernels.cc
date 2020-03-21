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

#include "tensorflow_io/core/kernels/audio_kernels.h"

#define MINIMP3_IMPLEMENTATION
#define MINIMP3_FLOAT_OUTPUT
#include "minimp3_ex.h"

#if defined(__linux__)
#include <dlfcn.h>
#endif

typedef void* lame_t;
typedef enum vbr_mode_e {
  vbr_off = 0,
  vbr_mt,
  vbr_rh,
  vbr_abr,
  vbr_mtrh,
  vbr_max_indicator,
  vbr_default = vbr_mtrh
} vbr_mode;

static lame_t (*lame_init)(void);
static int (*lame_set_num_channels)(lame_t, int);
static int (*lame_set_in_samplerate)(lame_t, int);
static int (*lame_set_VBR)(lame_t, vbr_mode);
static int (*lame_init_params)(lame_t);
static int (*lame_encode_buffer_interleaved_ieee_float)(lame_t gfp,
                                                        const float pcm[],
                                                        const int nsamples,
                                                        unsigned char* mp3buf,
                                                        const int mp3buf_size);
static int (*lame_encode_flush)(lame_t gfp, unsigned char* mp3buf, int size);
static int (*lame_close)(lame_t);

namespace tensorflow {
namespace data {
namespace {

class MP3Stream {
 public:
  MP3Stream(SizedRandomAccessFile* file, int64 size)
      : file(file), size(size), offset(0) {}
  ~MP3Stream() {}

  static size_t ReadCallback(void* buf, size_t size, void* user_data) {
    MP3Stream* p = static_cast<MP3Stream*>(user_data);
    StringPiece result;
    Status status = p->file->Read(p->offset, size, &result, (char*)buf);
    p->offset += result.size();
    return result.size();
  }

  static int SeekCallback(uint64_t position, void* user_data) {
    MP3Stream* p = static_cast<MP3Stream*>(user_data);
    if (position < 0 || position > p->size) {
      return -1;
    }
    p->offset = position;
    return 0;
  }

  SizedRandomAccessFile* file = nullptr;
  int64 size = 0;
  long offset = 0;
};

class MP3ReadableResource : public AudioReadableResourceBase {
 public:
  MP3ReadableResource(Env* env)
      : env_(env), mp3dec_ex_scope_(nullptr, [](mp3dec_ex_t* p) {
          if (p != nullptr) {
            mp3dec_ex_close(p);
          }
        }) {}
  ~MP3ReadableResource() {}

  Status Init(const string& filename, const void* optional_memory,
              const size_t optional_length) override {
    mutex_lock l(mu_);
    file_.reset(new SizedRandomAccessFile(env_, filename, optional_memory,
                                          optional_length));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    stream_.reset(new MP3Stream(file_.get(), file_size_));

    mp3dec_io_.read = MP3Stream::ReadCallback;
    mp3dec_io_.read_data = stream_.get();
    mp3dec_io_.seek = MP3Stream::SeekCallback;
    mp3dec_io_.seek_data = stream_.get();
    memset(&mp3dec_ex_, 0x00, sizeof(mp3dec_ex_));
    if (mp3dec_ex_open_cb(&mp3dec_ex_, &mp3dec_io_, MP3D_SEEK_TO_SAMPLE)) {
      return errors::InvalidArgument("unable to open file ", filename,
                                     " as mp3: ", mp3dec_ex_.last_error);
    }
    mp3dec_ex_scope_.reset(&mp3dec_ex_);
    if (mp3dec_ex_.info.channels == 0) {
      return errors::InvalidArgument("invalid mp3 with channel == 0");
    }
    int64 samples = mp3dec_ex_.samples / mp3dec_ex_.info.channels;
    int64 channels = mp3dec_ex_.info.channels;
    int64 rate = mp3dec_ex_.info.hz;

    shape_ = TensorShape({samples, channels});
    dtype_ = DT_FLOAT;
    rate_ = rate;

    return Status::OK();
  }

  Status Spec(TensorShape* shape, DataType* dtype, int32* rate) override {
    mutex_lock l(mu_);
    *shape = shape_;
    *dtype = dtype_;
    *rate = rate_;
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop,
              std::function<Status(const TensorShape& shape, Tensor** value)>
                  allocate_func) override {
    mutex_lock l(mu_);

    int64 sample_stop =
        (stop < 0) ? (shape_.dim_size(0))
                   : (stop < shape_.dim_size(0) ? stop : shape_.dim_size(0));
    int64 sample_start = (start >= sample_stop) ? sample_stop : start;

    Tensor* value;
    TF_RETURN_IF_ERROR(allocate_func(
        TensorShape({sample_stop - sample_start, shape_.dim_size(1)}), &value));

    if (mp3dec_ex_seek(&mp3dec_ex_, sample_start * shape_.dim_size(1))) {
      return errors::InvalidArgument("seek to ", sample_start,
                                     " failed: ", mp3dec_ex_.last_error);
    }
    size_t returned = mp3dec_ex_read(&mp3dec_ex_, value->flat<float>().data(),
                                     value->NumElements());
    if (returned != value->NumElements()) {
      return errors::InvalidArgument("read ", value->NumElements(), " from ",
                                     sample_start,
                                     " failed: ", mp3dec_ex_.last_error);
    }
    return Status::OK();
  }
  string DebugString() const override { return "MP3ReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  int64 rate_;

  std::unique_ptr<MP3Stream> stream_;
  mp3dec_io_t mp3dec_io_;
  mp3dec_ex_t mp3dec_ex_;
  std::unique_ptr<mp3dec_ex_t, void (*)(mp3dec_ex_t*)> mp3dec_ex_scope_;
};

class AudioDecodeMP3Op : public OpKernel {
 public:
  explicit AudioDecodeMP3Op(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* shape_tensor;
    OP_REQUIRES_OK(context, context->input("shape", &shape_tensor));

    const tstring& input = input_tensor->scalar<tstring>()();

    std::unique_ptr<MP3ReadableResource> resource(
        new MP3ReadableResource(env_));
    OP_REQUIRES_OK(context,
                   resource->Init("memory", input.data(), input.size()));

    int32 rate;
    DataType dtype;
    TensorShape shape;
    OP_REQUIRES_OK(context, resource->Spec(&shape, &dtype, &rate));

    OP_REQUIRES(context, (dtype == context->expected_output_dtype(0)),
                errors::InvalidArgument(
                    "dtype mismatch: ", DataTypeString(dtype), " vs. ",
                    DataTypeString(context->expected_output_dtype(0))));

    PartialTensorShape provided_shape;
    OP_REQUIRES_OK(context, PartialTensorShape::MakePartialShape(
                                shape_tensor->flat<int64>().data(),
                                shape_tensor->NumElements(), &provided_shape));
    OP_REQUIRES(context, (provided_shape.IsCompatibleWith(shape)),
                errors::InvalidArgument(
                    "shape mismatch: ", provided_shape.DebugString(), " vs. ",
                    shape.DebugString()));

    OP_REQUIRES_OK(
        context,
        resource->Read(0, shape.dim_size(0),
                       [&](const TensorShape& shape, Tensor** value) -> Status {
                         TF_RETURN_IF_ERROR(
                             context->allocate_output(0, shape, value));
                         return Status::OK();
                       }));
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

bool LoadLame() {
#if defined(__linux__)
  void* lib = dlopen("libmp3lame.so.0", RTLD_NOW);
  if (lib != nullptr) {
    *(void**)(&lame_init) = dlsym(lib, "lame_init");
    *(void**)(&lame_set_num_channels) = dlsym(lib, "lame_set_num_channels");
    *(void**)(&lame_set_in_samplerate) = dlsym(lib, "lame_set_in_samplerate");
    *(void**)(&lame_set_VBR) = dlsym(lib, "lame_set_VBR");
    *(void**)(&lame_init_params) = dlsym(lib, "lame_init_params");
    *(void**)(&lame_encode_buffer_interleaved_ieee_float) =
        dlsym(lib, "lame_encode_buffer_interleaved_ieee_float");
    *(void**)(&lame_encode_flush) = dlsym(lib, "lame_encode_flush");
    *(void**)(&lame_close) = dlsym(lib, "lame_close");
    if (lame_init != nullptr && lame_set_num_channels != nullptr &&
        lame_set_in_samplerate != nullptr && lame_set_VBR != nullptr &&
        lame_init_params != nullptr &&
        lame_encode_buffer_interleaved_ieee_float != nullptr &&
        lame_encode_flush != nullptr && lame_close != nullptr) {
      return true;
    }
  }
  LOG(WARNING) << "libmp3lame.so.0 or lame functions are not available";
#endif
  return false;
}

class AudioEncodeMP3Op : public OpKernel {
 public:
  explicit AudioEncodeMP3Op(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, lame_available_,
                errors::InvalidArgument("lame library is not available"));
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* rate_tensor;
    OP_REQUIRES_OK(context, context->input("rate", &rate_tensor));

    const int64 rate = rate_tensor->scalar<int64>()();
    const int64 samples = input_tensor->shape().dim_size(0);
    const int64 channels = input_tensor->shape().dim_size(1);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));

    tstring& output = output_tensor->scalar<tstring>()();

    std::unique_ptr<void, void (*)(void*)> lame(nullptr, [](void* p) {
      if (p != nullptr) {
        lame_close(p);
      }
    });
    lame.reset(lame_init());
    OP_REQUIRES(context, (lame.get() != nullptr),
                errors::InvalidArgument("unable to initialize lame"));

    int status;
    status = lame_set_num_channels(lame.get(), channels);
    OP_REQUIRES(context, (status == 0),
                errors::InvalidArgument("unable to set channels: ", status));

    status = lame_set_in_samplerate(lame.get(), rate);
    OP_REQUIRES(context, (status == 0),
                errors::InvalidArgument("unable to set rate: ", status));

    status = lame_set_VBR(lame.get(), vbr_default);
    OP_REQUIRES(context, (status == 0),
                errors::InvalidArgument("unable to set vbr: ", status));

    status = lame_init_params(lame.get());
    OP_REQUIRES(context, (status == 0),
                errors::InvalidArgument("unable to init params ", status));

    const float* pcm = input_tensor->flat<float>().data();

    // worse case according to lame:
    // mp3buf_size in bytes = 1.25*num_samples + 7200
    output.resize(samples * 5 / 4 + 7200);
    unsigned char* mp3buf = (unsigned char*)&output[0];
    int mp3buf_size = output.size();
    status = lame_encode_buffer_interleaved_ieee_float(lame.get(), pcm, samples,
                                                       mp3buf, mp3buf_size);
    OP_REQUIRES(context, (status >= 0),
                errors::InvalidArgument("unable to encode: ", status));

    int encoded = status;

    mp3buf = (unsigned char*)&output[encoded];
    mp3buf_size = output.size() - encoded;
    status = lame_encode_flush(lame.get(), mp3buf, mp3buf_size);
    OP_REQUIRES(context, (status >= 0),
                errors::InvalidArgument("unable to flush: ", status));
    encoded = encoded + status;
    // cur to the encoded length
    output.resize(encoded);
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);

  static bool lame_available_;
};

bool AudioEncodeMP3Op::lame_available_ = LoadLame();

REGISTER_KERNEL_BUILDER(Name("IO>AudioDecodeMP3").Device(DEVICE_CPU),
                        AudioDecodeMP3Op);
REGISTER_KERNEL_BUILDER(Name("IO>AudioEncodeMP3").Device(DEVICE_CPU),
                        AudioEncodeMP3Op);

}  // namespace

Status MP3ReadableResourceInit(
    Env* env, const string& filename, const void* optional_memory,
    const size_t optional_length,
    std::unique_ptr<AudioReadableResourceBase>& resource) {
  resource.reset(new MP3ReadableResource(env));
  Status status = resource->Init(filename, optional_memory, optional_length);
  if (!status.ok()) {
    resource.reset(nullptr);
  }
  return status;
}

}  // namespace data
}  // namespace tensorflow
