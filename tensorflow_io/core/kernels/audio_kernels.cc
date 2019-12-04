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

#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/kernels/io_stream.h"
#include "vorbis/codec.h"
#include "vorbis/vorbisfile.h"

namespace tensorflow {
namespace data {
namespace {

// See http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
struct WAVHeader {
  char riff[4];           // RIFF Chunk ID: "RIFF"
  int32 riff_size;        // RIFF chunk size: 4 + n (file size - 8)
  char wave[4];           // WAVE ID: "WAVE"
  char fmt[4];            // fmt Chunk ID: "fmt "
  int32 fmt_size;         // fmt Chunk size: 16, 18, or 40
  int16 wFormatTag;       // Format code: WAVE_FORMAT_PCM (1) for PCM.
                          // WAVE_FORMAT_EXTENSIBLE (0xFFFE) for SubFormat
  int16 nChannels;        // Number of channels
  int32 nSamplesPerSec;   // Sampling rate
  int32 nAvgBytesPerSec;  // Data rate
  int16 nBlockAlign;      // Data block size (bytes)
  int16 wBitsPerSample;   // Bits per sample
};

struct ExtensionHeader {
  int16 cbSize;               // Size of the extension (0 or 22)
  int16 wValidBitsPerSample;  // Number of valid bits
  int32 dwChannelMask;        // Speaker position mask
  char SubFormat[16];         // GUID
};

struct DataHeader {
  char mark[4];
  int32 size;
};
Status ValidateWAVHeader(struct WAVHeader* header) {
  if (memcmp(header->riff, "RIFF", 4) != 0) {
    return errors::InvalidArgument("WAV file must starts with `RIFF`");
  }
  if (memcmp(header->wave, "WAVE", 4) != 0) {
    return errors::InvalidArgument("WAV file must contains riff type `WAVE`");
  }
  if (memcmp(header->fmt, "fmt ", 4) != 0) {
    return errors::InvalidArgument("WAV file must contains `fmt ` mark");
  }
  if (header->fmt_size != 16 && header->fmt_size != 18 &&
      header->fmt_size != 40) {
    return errors::InvalidArgument(
        "WAV file must have `fmt_size ` 16, 18, or 40, received: ",
        header->fmt_size);
  }
  if (header->wFormatTag != 1 &&
      header->wFormatTag != static_cast<int16>(0xFFFE)) {
    return errors::InvalidArgument(
        "WAV file must have `wFormatTag` 1 or 0xFFFE, received: ",
        header->wFormatTag);
  }
  if (header->nChannels <= 0) {
    return errors::InvalidArgument("WAV file have invalide channels: ",
                                   header->nChannels);
  }
  return Status::OK();
}

class AudioReadableResourceBase : public ResourceBase {
 public:
  virtual Status Init(const string& input) = 0;
  virtual Status Read(const int64 start, Tensor* value) = 0;
  virtual Status Spec(TensorShape* shape, DataType* dtype, int32* rate) = 0;
  virtual Status Peek(const int64 start, const int64 stop,
                      TensorShape* shape) = 0;
};

class WAVReadableResource : public AudioReadableResourceBase {
 public:
  WAVReadableResource(Env* env) : env_(env) {}
  ~WAVReadableResource() {}

  Status Init(const string& input) override {
    mutex_lock l(mu_);
    const string& filename = input;
    file_.reset(new SizedRandomAccessFile(env_, filename, nullptr, 0));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    StringPiece result;
    TF_RETURN_IF_ERROR(
        file_->Read(0, sizeof(header_), &result, (char*)(&header_)));
    header_length_ = sizeof(header_);
    int64 fmt_position = 12;
    while (memcmp(header_.fmt, "fmt ", 4) != 0) {
      // Skip JUNK/bext/etc field.
      if (memcmp(header_.fmt, "JUNK", 4) != 0 &&
          memcmp(header_.fmt, "bext", 4) != 0 &&
          memcmp(header_.fmt, "iXML", 4) != 0 &&
          memcmp(header_.fmt, "qlty", 4) != 0 &&
          memcmp(header_.fmt, "mext", 4) != 0 &&
          memcmp(header_.fmt, "levl", 4) != 0 &&
          memcmp(header_.fmt, "link", 4) != 0 &&
          memcmp(header_.fmt, "axml", 4) != 0) {
        return errors::InvalidArgument("unexpected field: ", header_.fmt);
      }
      int32 size_of_chunk = 4 + 4 + header_.fmt_size;
      if (header_.fmt_size % 2 == 1) {
        size_of_chunk += 1;
      }
      fmt_position += size_of_chunk;
      // Re-read the header
      TF_RETURN_IF_ERROR(file_->Read(fmt_position, sizeof(header_) - 12,
                                     &result, (char*)(&header_) + 12));
      header_length_ = fmt_position + sizeof(header_) - 12;
    }

    TF_RETURN_IF_ERROR(ValidateWAVHeader(&header_));
    if (header_.riff_size + 8 != file_size_) {
      // corrupted file?
    }
    int64 filesize = header_.riff_size + 8;
    int64 position = header_length_ + header_.fmt_size - 16;

    int64 nSamples = 0;
    do {
      struct DataHeader head;
      TF_RETURN_IF_ERROR(
          file_->Read(position, sizeof(head), &result, (char*)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        // Data should be block aligned
        // bytes = nSamples * nBlockAlign
        if (head.size % header_.nBlockAlign != 0) {
          return errors::InvalidArgument("data chunk should be block aligned (",
                                         header_.nBlockAlign,
                                         "), received: ", head.size);
        }
        nSamples += head.size / header_.nBlockAlign;
      }
      position += head.size;
    } while (position < filesize);

    switch (header_.wBitsPerSample) {
      case 8:
        dtype_ = DT_INT8;
        break;
      case 16:
        dtype_ = DT_INT16;
        break;
      case 24:
        dtype_ = DT_INT32;
        break;
      default:
        return errors::InvalidArgument("unsupported wBitsPerSample: ",
                                       header_.wBitsPerSample);
    }

    shape_ = TensorShape({nSamples, header_.nChannels});

    rate_ = header_.nSamplesPerSec;

    return Status::OK();
  }

  Status Spec(TensorShape* shape, DataType* dtype, int32* rate) override {
    mutex_lock l(mu_);
    *shape = shape_;
    *dtype = dtype_;
    *rate = rate_;
    return Status::OK();
  }

  Status Peek(const int64 start, const int64 stop,
              TensorShape* shape) override {
    mutex_lock l(mu_);
    int64 sample_stop =
        (stop < 0) ? (shape_.dim_size(0))
                   : (stop < shape_.dim_size(0) ? stop : shape_.dim_size(0));
    int64 sample_start = (start >= sample_stop) ? sample_stop : start;
    *shape = TensorShape({sample_stop - sample_start, shape_.dim_size(1)});
    return Status::OK();
  }

  Status Read(const int64 start, Tensor* value) override {
    mutex_lock l(mu_);
    const int64 sample_start = start;
    const int64 sample_stop = start + value->shape().dim_size(0);

    int64 sample_offset = 0;
    if (header_.riff_size + 8 != file_size_) {
      // corrupted file?
    }
    int64 filesize = header_.riff_size + 8;
    int64 position = header_length_ + header_.fmt_size - 16;
    do {
      StringPiece result;
      struct DataHeader head;
      TF_RETURN_IF_ERROR(
          file_->Read(position, sizeof(head), &result, (char*)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        // Already checked the alignment
        int64 block_sample_start = sample_offset;
        int64 block_sample_stop =
            sample_offset + head.size / header_.nBlockAlign;
        // only read if block_sample_start and block_sample_stop within range
        if (sample_start < block_sample_stop &&
            sample_stop > block_sample_start) {
          int64 read_sample_start =
              (block_sample_start > sample_start ? block_sample_start
                                                 : sample_start);
          int64 read_sample_stop =
              (block_sample_stop < sample_stop ? block_sample_stop
                                               : sample_stop);
          int64 read_bytes_start =
              position +
              (read_sample_start - block_sample_start) * header_.nBlockAlign;
          int64 read_bytes_stop =
              position +
              (read_sample_stop - block_sample_start) * header_.nBlockAlign;
          string buffer;
          buffer.resize(read_bytes_stop - read_bytes_start);
          TF_RETURN_IF_ERROR(file_->Read(read_bytes_start,
                                         read_bytes_stop - read_bytes_start,
                                         &result, &buffer[0]));
          switch (header_.wBitsPerSample) {
            case 8:
              if (header_.wBitsPerSample * header_.nChannels !=
                  header_.nBlockAlign * 8) {
                return errors::InvalidArgument(
                    "unsupported wBitsPerSample and header.nBlockAlign: ",
                    header_.wBitsPerSample, ", ", header_.nBlockAlign);
              }
              memcpy((char*)(value->flat<int8>().data()) +
                         ((read_sample_start - sample_start) *
                          header_.nBlockAlign),
                     &buffer[0], (read_bytes_stop - read_bytes_start));
              break;
            case 16:
              if (header_.wBitsPerSample * header_.nChannels !=
                  header_.nBlockAlign * 8) {
                return errors::InvalidArgument(
                    "unsupported wBitsPerSample and header.nBlockAlign: ",
                    header_.wBitsPerSample, ", ", header_.nBlockAlign);
              }
              memcpy((char*)(value->flat<int16>().data()) +
                         ((read_sample_start - sample_start) *
                          header_.nBlockAlign),
                     &buffer[0], (read_bytes_stop - read_bytes_start));
              break;
            case 24:
              // NOTE: The conversion is from signed integer 24 to signed
              // integer 32 (left shift 8 bits)
              if (header_.wBitsPerSample * header_.nChannels !=
                  header_.nBlockAlign * 8) {
                return errors::InvalidArgument(
                    "unsupported wBitsPerSample and header.nBlockAlign: ",
                    header_.wBitsPerSample, ", ", header_.nBlockAlign);
              }
              for (int64 i = read_sample_start; i < read_sample_stop; i++) {
                for (int64 j = 0; j < header_.nChannels; j++) {
                  char* data_p =
                      (char*)(value->flat<int32>().data() +
                              ((i - sample_start) * header_.nChannels + j));
                  char* read_p = (char*)(&buffer[((i - read_sample_start) *
                                                  header_.nBlockAlign)]) +
                                 3 * j;
                  data_p[3] = read_p[2];
                  data_p[2] = read_p[1];
                  data_p[1] = read_p[0];
                  data_p[0] = 0x00;
                }
              }
              break;
            default:
              return errors::InvalidArgument(
                  "unsupported wBitsPerSample and header.nBlockAlign: ",
                  header_.wBitsPerSample, ", ", header_.nBlockAlign);
          }
        }
        sample_offset = block_sample_stop;
      }
      position += head.size;
    } while (position < filesize);

    return Status::OK();
  }
  string DebugString() const override { return "WAVReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  int64 rate_;

  struct WAVHeader header_;
  int64 header_length_;
};

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

class OggReadableResource : public AudioReadableResourceBase {
 public:
  OggReadableResource(Env* env) : env_(env) {}
  ~OggReadableResource() {}

  Status Init(const string& input) override {
    mutex_lock l(mu_);
    const string& filename = input;
    file_.reset(new SizedRandomAccessFile(env_, filename, nullptr, 0));
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
    dtype_ = DT_INT16;
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

  Status Peek(const int64 start, const int64 stop,
              TensorShape* shape) override {
    mutex_lock l(mu_);
    int64 sample_stop =
        (stop < 0) ? (shape_.dim_size(0))
                   : (stop < shape_.dim_size(0) ? stop : shape_.dim_size(0));
    int64 sample_start = (start >= sample_stop) ? sample_stop : start;
    *shape = TensorShape({sample_stop - sample_start, shape_.dim_size(1)});
    return Status::OK();
  }

  Status Read(const int64 start, Tensor* value) override {
    mutex_lock l(mu_);
    const int64 sample_start = start;
    const int64 sample_stop = start + value->shape().dim_size(0);

    int returned = ov_pcm_seek(&ogg_vorbis_file_, sample_start);
    if (returned < 0) {
      return errors::InvalidArgument("seek failed: ", returned);
    }

    int bitstream = 0;
    long bytes_read = 0;
    long bytes_to_read = value->NumElements() * sizeof(int16);
    while (bytes_read < bytes_to_read) {
      long chunk = ov_read(&ogg_vorbis_file_,
                           (char*)value->flat<int16>().data() + bytes_read,
                           bytes_to_read - bytes_read, 0, 2, 1, &bitstream);
      if (chunk < 0) {
        return errors::InvalidArgument("read failed: ", chunk);
      }
      if (chunk == 0) {
        return errors::InvalidArgument("not enough data: ");
      }
      bytes_read += chunk;
    }
    return Status::OK();
  }
  string DebugString() const override { return "OggReadableResource"; }

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

class AudioReadableResource : public AudioReadableResourceBase {
 public:
  AudioReadableResource(Env* env) : env_(env), resource_(nullptr) {}
  ~AudioReadableResource() {}

  Status Init(const string& input) override {
    mutex_lock l(mu_);
    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(input, &file));
    char header[4];
    StringPiece result;
    TF_RETURN_IF_ERROR(file->Read(0, sizeof(header), &result, header));
    if (memcmp(header, "RIFF", 4) == 0) {
      resource_.reset(new WAVReadableResource(env_));
    } else if (memcmp(header, "OggS", 4) == 0) {
      resource_.reset(new OggReadableResource(env_));
    } else {
      return errors::InvalidArgument("unknown header: ", header);
    }
    return resource_->Init(input);
  }
  Status Spec(TensorShape* shape, DataType* dtype, int32* rate) override {
    mutex_lock l(mu_);
    return resource_->Spec(shape, dtype, rate);
  }
  Status Peek(const int64 start, const int64 stop,
              TensorShape* shape) override {
    mutex_lock l(mu_);
    return resource_->Peek(start, stop, shape);
  }
  Status Read(const int64 start, Tensor* value) override {
    mutex_lock l(mu_);
    return resource_->Read(start, value);
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return resource_->DebugString();
  }

 protected:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<AudioReadableResourceBase> resource_ GUARDED_BY(mu_);
};

class AudioReadableInitOp : public ResourceOpKernel<AudioReadableResource> {
 public:
  explicit AudioReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<AudioReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<AudioReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    OP_REQUIRES_OK(context, resource_->Init(input_tensor->scalar<string>()()));
  }
  Status CreateResource(AudioReadableResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new AudioReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class AudioReadableSpecOp : public OpKernel {
 public:
  explicit AudioReadableSpecOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    AudioReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    TensorShape shape;
    DataType dtype;
    int32 rate;
    OP_REQUIRES_OK(context, resource->Spec(&shape, &dtype, &rate));

    Tensor* shape_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({2}), &shape_tensor));
    shape_tensor->flat<int64>()(0) = shape.dim_size(0);
    shape_tensor->flat<int64>()(1) = shape.dim_size(1);

    Tensor* dtype_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}), &dtype_tensor));
    dtype_tensor->scalar<int64>()() = dtype;

    Tensor* rate_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, TensorShape({}), &rate_tensor));
    rate_tensor->scalar<int32>()() = rate;
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
class AudioReadableReadOp : public OpKernel {
 public:
  explicit AudioReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    AudioReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    int64 stop = stop_tensor->scalar<int64>()();
    TensorShape value_shape;
    OP_REQUIRES_OK(context, resource->Peek(start, stop, &value_shape));

    Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, value_shape, &value_tensor));
    if (value_shape.dim_size(0) > 0) {
      OP_REQUIRES_OK(context, resource->Read(start, value_tensor));
    }
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableInit").Device(DEVICE_CPU),
                        AudioReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableSpec").Device(DEVICE_CPU),
                        AudioReadableSpecOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableRead").Device(DEVICE_CPU),
                        AudioReadableReadOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
