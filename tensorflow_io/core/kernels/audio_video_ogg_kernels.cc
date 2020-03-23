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

#include "vorbis/codec.h"
#include "vorbis/vorbisenc.h"
#include "vorbis/vorbisfile.h"

namespace tensorflow {
namespace data {
namespace {

class OggVorbisStream {
 public:
  OggVorbisStream(SizedRandomAccessFile* file, int64 size)
      : file(file), size(size), offset(0) {}
  ~OggVorbisStream() {}

  static size_t ReadCallback(void* ptr, size_t size, size_t count,
                             void* stream) {
    OggVorbisStream* p = static_cast<OggVorbisStream*>(stream);
    StringPiece result;
    Status status = p->file->Read(p->offset, size * count, &result, (char*)ptr);
    size_t items = result.size() / size;
    p->offset += items * size;
    return items;
  }

  static int SeekCallback(void* stream, ogg_int64_t offset, int origin) {
    OggVorbisStream* p = static_cast<OggVorbisStream*>(stream);
    switch (origin) {
      case SEEK_SET:
        if (offset < 0 || offset > p->size) {
          return -1;
        }
        p->offset = offset;
        break;
      case SEEK_CUR:
        if (p->offset + offset < 0 || p->offset + offset > p->size) {
          return -1;
        }
        p->offset += offset;
        break;
      case SEEK_END:
        if (p->size + offset < 0 || p->size + offset > p->size) {
          return -1;
        }
        p->offset = p->size + offset;
        break;
      default:
        return -1;
    }
    return 0;
  }

  static long TellCallback(void* stream) {
    OggVorbisStream* p = static_cast<OggVorbisStream*>(stream);
    return p->offset;
  }

  SizedRandomAccessFile* file = nullptr;
  int64 size = 0;
  long offset = 0;
};

static ov_callbacks OggVorbisCallbacks = {
    (size_t(*)(void*, size_t, size_t, void*))OggVorbisStream::ReadCallback,
    (int (*)(void*, ogg_int64_t, int))OggVorbisStream::SeekCallback,
    (int (*)(void*))NULL, (long (*)(void*))OggVorbisStream::TellCallback};

class OggVorbisReadableResource : public AudioReadableResourceBase {
 public:
  OggVorbisReadableResource(Env* env) : env_(env) {}
  ~OggVorbisReadableResource() {}

  Status Init(const string& filename, const void* optional_memory,
              const size_t optional_length) override {
    mutex_lock l(mu_);
    file_.reset(new SizedRandomAccessFile(env_, filename, optional_memory,
                                          optional_length));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    stream_.reset(new OggVorbisStream(file_.get(), file_size_));
    int returned = ov_open_callbacks(stream_.get(), &ogg_vorbis_file_, NULL, -1,
                                     OggVorbisCallbacks);
    if (returned < 0) {
      return errors::InvalidArgument(
          "could not open input as an OggVorbis file: ", returned);
    }

    vorbis_info* vi = ov_info(&ogg_vorbis_file_, -1);
    int64 samples = ov_pcm_total(&ogg_vorbis_file_, -1);
    int64 channels = vi->channels;
    int64 rate = vi->rate;

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

    int returned = ov_pcm_seek(&ogg_vorbis_file_, sample_start);
    if (returned < 0) {
      return errors::InvalidArgument("seek failed: ", returned);
    }

    int64 channels = value->shape().dim_size(1);

    long samples_read = 0;
    long samples_to_read = value->shape().dim_size(0);
    while (samples_read < samples_to_read) {
      float** buffer;
      int bitstream = 0;
      long chunk = ov_read_float(&ogg_vorbis_file_, &buffer,
                                 samples_to_read - samples_read, &bitstream);
      if (chunk < 0) {
        return errors::InvalidArgument("read failed: ", chunk);
      }
      if (chunk == 0) {
        return errors::InvalidArgument("not enough data: ");
      }
      for (int64 c = 0; c < channels; c++) {
        for (int64 i = 0; i < chunk; i++) {
          value->matrix<float>()(samples_read + i, c) = buffer[c][i];
        }
      }
      samples_read += chunk;
    }
    return Status::OK();
  }
  string DebugString() const override { return "OggVorbisReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  int64 rate_;

  OggVorbis_File ogg_vorbis_file_;
  std::unique_ptr<OggVorbisStream> stream_;
};

class AudioDecodeVorbisOp : public OpKernel {
 public:
  explicit AudioDecodeVorbisOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* shape_tensor;
    OP_REQUIRES_OK(context, context->input("shape", &shape_tensor));

    const tstring& input = input_tensor->scalar<tstring>()();

    std::unique_ptr<OggVorbisReadableResource> resource(
        new OggVorbisReadableResource(env_));
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

Status OggVorbisEncodeStreamProcess(vorbis_dsp_state& vd, vorbis_block& vb,
                                    ogg_stream_state& os, ogg_page& og,
                                    ogg_packet& op, tstring* output) {
  int s;
  while (vorbis_analysis_blockout(&vd, &vb) == 1) {
    vorbis_analysis(&vb, NULL);
    vorbis_bitrate_addblock(&vb);

    while (vorbis_bitrate_flushpacket(&vd, &op) == 1) {
      // weld the packet into the bitstream
      ogg_stream_packetin(&os, &op);

      // write out pages (if any)
      while ((s = ogg_stream_flush(&os, &og)) != 0) {
        output->append((const char*)og.header, og.header_len);
        output->append((const char*)og.body, og.body_len);
        if (ogg_page_eos(&og) != 0) {
          break;
        }
      }
    }
  }
  return Status::OK();
}

class AudioEncodeVorbisOp : public OpKernel {
 public:
  explicit AudioEncodeVorbisOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* rate_tensor;
    OP_REQUIRES_OK(context, context->input("rate", &rate_tensor));

    const int64 rate = rate_tensor->scalar<int64>()();
    const int64 samples = input_tensor->shape().dim_size(0);
    const int64 channels = input_tensor->shape().dim_size(1);

    vorbis_info vi;
    vorbis_info_init(&vi);
    std::unique_ptr<vorbis_info, void (*)(vorbis_info*)> vi_scope(
        &vi, [](vorbis_info* p) {
          if (p != nullptr) {
            vorbis_info_clear(p);
          }
        });

    int s;

    s = vorbis_encode_init(&vi, channels, rate, -1, 128000, -1);
    OP_REQUIRES(context, (s == 0),
                errors::InvalidArgument("unable to init encode: ", s));

    // add a comment
    vorbis_comment vc;
    vorbis_comment_init(&vc);
    std::unique_ptr<vorbis_comment, void (*)(vorbis_comment*)> vc_scope(
        &vc, [](vorbis_comment* p) {
          if (p != nullptr) {
            vorbis_comment_clear(p);
          }
        });
    vorbis_comment_add_tag(&vc, "ENCODER", "tensorflow-io");

    // set up the analysis state
    vorbis_dsp_state vd;
    vorbis_analysis_init(&vd, &vi);
    std::unique_ptr<vorbis_dsp_state, void (*)(vorbis_dsp_state*)> vd_scope(
        &vd, [](vorbis_dsp_state* p) {
          if (p != nullptr) {
            vorbis_dsp_clear(p);
          }
        });

    // auxiliary encoding storage
    vorbis_block vb;
    vorbis_block_init(&vd, &vb);
    std::unique_ptr<vorbis_block, void (*)(vorbis_block*)> vb_scope(
        &vb, [](vorbis_block* p) {
          if (p != nullptr) {
            vorbis_block_clear(p);
          }
        });

    // srand(time(NULL));
    ogg_stream_state os;
    s = ogg_stream_init(&os, rand());
    OP_REQUIRES(context, (s == 0),
                errors::InvalidArgument("unable to init ogg stream: ", s));
    std::unique_ptr<ogg_stream_state, void (*)(ogg_stream_state*)> os_scope(
        &os, [](ogg_stream_state* p) {
          if (p != nullptr) {
            ogg_stream_clear(p);
          }
        });

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));

    tstring& output = output_tensor->scalar<tstring>()();

    ogg_page og;

    {
      ogg_packet header;
      ogg_packet header_comm;
      ogg_packet header_code;

      vorbis_analysis_headerout(&vd, &vc, &header, &header_comm, &header_code);
      ogg_stream_packetin(&os, &header);
      ogg_stream_packetin(&os, &header_comm);
      ogg_stream_packetin(&os, &header_code);

      // ensures the actual audio data will start on a new page, as per spec
      while ((s = ogg_stream_flush(&os, &og)) != 0) {
        output.append((const char*)og.header, og.header_len);
        output.append((const char*)og.body, og.body_len);
      }
    }

    // expose the buffer to submit data
    float** buffer = vorbis_analysis_buffer(&vd, samples);

    // uninterleave samples
    for (int64 i = 0; i < samples; i++) {
      for (int64 c = 0; c < channels; c++) {
        buffer[c][i] = input_tensor->matrix<float>()(i, c);
      }
    }

    ogg_packet op;

    // tell the library how much we actually submitted
    vorbis_analysis_wrote(&vd, samples);
    OP_REQUIRES_OK(context,
                   OggVorbisEncodeStreamProcess(vd, vb, os, og, op, &output));

    // end of file
    vorbis_analysis_wrote(&vd, 0);
    OP_REQUIRES_OK(context,
                   OggVorbisEncodeStreamProcess(vd, vb, os, og, op, &output));
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>AudioDecodeVorbis").Device(DEVICE_CPU),
                        AudioDecodeVorbisOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioEncodeVorbis").Device(DEVICE_CPU),
                        AudioEncodeVorbisOp);

}  // namespace

Status OggVorbisReadableResourceInit(
    Env* env, const string& filename, const void* optional_memory,
    const size_t optional_length,
    std::unique_ptr<AudioReadableResourceBase>& resource) {
  resource.reset(new OggVorbisReadableResource(env));
  Status status = resource->Init(filename, optional_memory, optional_length);
  if (!status.ok()) {
    resource.reset(nullptr);
  }
  return status;
}

}  // namespace data
}  // namespace tensorflow
