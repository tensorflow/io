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
#include "tensorflow_io/core/kernels/io_stream.h"

#include "FLAC/stream_decoder.h"
#include "speex/speex_resampler.h"
#include "vorbis/codec.h"
#include "vorbis/vorbisfile.h"
#define MINIMP3_IMPLEMENTATION
#include "minimp3_ex.h"

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
  virtual Status Read(
      const int64 start, const int64 stop,
      std::function<Status(const TensorShape& shape, Tensor** value)>
          allocate_func) = 0;
  virtual Status Spec(TensorShape* shape, DataType* dtype, int32* rate) = 0;
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

class FlacStreamDecoder {
 public:
  FlacStreamDecoder(SizedRandomAccessFile* file, int64 size)
      : file(file),
        size(size),
        offset(0),
        samples(0),
        channels(0),
        rate(0),
        bits_per_sample(0) {}
  ~FlacStreamDecoder() {}

  void SetTensor(int64 start, Tensor* value) {
    sample_index = start;
    sample_start = start;
    sample_value = value;
  }

  static FLAC__StreamDecoderReadStatus ReadCallback(
      const FLAC__StreamDecoder* decoder, FLAC__byte buffer[], size_t* bytes,
      void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    if (*bytes > 0) {
      if (p->offset >= p->size) {
        *bytes = 0;
        return FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
      }
      size_t bytes_to_read =
          (p->offset + *bytes) < (p->size) ? (*bytes) : (p->size - p->offset);
      StringPiece result;
      Status status =
          p->file->Read(p->offset, bytes_to_read, &result, (char*)buffer);
      if (result.size() != bytes_to_read) {
        return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
      }
      *bytes = bytes_to_read;
      p->offset += bytes_to_read;
      return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
    }
    return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
  }
  static FLAC__StreamDecoderSeekStatus SeekCallback(
      const FLAC__StreamDecoder* decoder, FLAC__uint64 absolute_byte_offset,
      void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    if (absolute_byte_offset > p->size) {
      return FLAC__STREAM_DECODER_SEEK_STATUS_ERROR;
    }
    p->offset = absolute_byte_offset;
    return FLAC__STREAM_DECODER_SEEK_STATUS_OK;
  }
  static FLAC__StreamDecoderTellStatus TellCallback(
      const FLAC__StreamDecoder* decoder, FLAC__uint64* absolute_byte_offset,
      void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    *absolute_byte_offset = p->offset;
    return FLAC__STREAM_DECODER_TELL_STATUS_OK;
  }
  static FLAC__StreamDecoderLengthStatus LengthCallback(
      const FLAC__StreamDecoder* decoder, FLAC__uint64* stream_length,
      void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    *stream_length = p->size;
    return FLAC__STREAM_DECODER_LENGTH_STATUS_OK;
  }

  static FLAC__bool EofCallback(const FLAC__StreamDecoder* decoder,
                                void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    return p->offset >= p->size;
  }
  static FLAC__StreamDecoderWriteStatus WriteCallback(
      const FLAC__StreamDecoder* decoder, const FLAC__Frame* frame,
      const FLAC__int32* const buffer[], void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    if (frame->header.channels != p->channels) {
      return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
    }
    if (frame->header.number_type != FLAC__FRAME_NUMBER_TYPE_SAMPLE_NUMBER) {
      return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
    }

    if (p->sample_index != frame->header.number.sample_number) {
      return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
    }

    int64 samples_to_read =
        (p->sample_index + frame->header.blocksize) <
                (p->sample_start + p->sample_value->shape().dim_size(0))
            ? (frame->header.blocksize)
            : (p->sample_start + p->sample_value->shape().dim_size(0) -
               p->sample_index);

    switch (p->sample_value->dtype()) {
      case DT_INT16:
        for (int64 channel = 0; channel < frame->header.channels; channel++) {
          for (int64 index = 0; index < samples_to_read; index++) {
            int64 sample_index = p->sample_index + index - p->sample_start;
            p->sample_value->tensor<int16, 2>()(sample_index, channel) =
                buffer[channel][index];
          }
        }
        break;
      default:
        return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
    }
    p->sample_index += samples_to_read;
    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
  }

  static void MetadataCallback(const FLAC__StreamDecoder* decoder,
                               const FLAC__StreamMetadata* metadata,
                               void* client_data) {
    FlacStreamDecoder* p = static_cast<FlacStreamDecoder*>(client_data);
    p->samples = metadata->data.stream_info.total_samples;
    p->channels = metadata->data.stream_info.channels;
    p->rate = metadata->data.stream_info.sample_rate;
    p->bits_per_sample = metadata->data.stream_info.bits_per_sample;
  }

  static void ErrorCallback(const FLAC__StreamDecoder* decoder,
                            FLAC__StreamDecoderErrorStatus status,
                            void* client_data) {
    std::cerr << "ErrorCallback: " << std::endl;
  }

  SizedRandomAccessFile* file;
  int64 size;
  int64 offset;

  int64 samples;
  int64 channels;
  int64 rate;
  int64 bits_per_sample;

  int64 sample_index;
  int64 sample_start;
  Tensor* sample_value;
};

class FlacReadableResource : public AudioReadableResourceBase {
 public:
  FlacReadableResource(Env* env)
      : env_(env), decoder_(nullptr, [](FLAC__StreamDecoder* p) {
          if (p != nullptr) {
            FLAC__stream_decoder_delete(p);
          }
        }) {}
  ~FlacReadableResource() {}

  Status Init(const string& input) override {
    mutex_lock l(mu_);
    const string& filename = input;
    file_.reset(new SizedRandomAccessFile(env_, filename, nullptr, 0));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    decoder_.reset(FLAC__stream_decoder_new());
    stream_decoder_.reset(new FlacStreamDecoder(file_.get(), file_size_));

    FLAC__StreamDecoderInitStatus s = FLAC__stream_decoder_init_stream(
        decoder_.get(), FlacStreamDecoder::ReadCallback,
        FlacStreamDecoder::SeekCallback, FlacStreamDecoder::TellCallback,
        FlacStreamDecoder::LengthCallback, FlacStreamDecoder::EofCallback,
        FlacStreamDecoder::WriteCallback, FlacStreamDecoder::MetadataCallback,
        FlacStreamDecoder::ErrorCallback, stream_decoder_.get());
    if (s != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
      return errors::InvalidArgument("unable to initialize stream: ", s);
    }
    if (!FLAC__stream_decoder_process_until_end_of_metadata(decoder_.get())) {
      return errors::InvalidArgument("unable to read metadata");
    }

    int64 samples = stream_decoder_->samples;
    int64 channels = stream_decoder_->channels;
    int64 rate = stream_decoder_->rate;
    DataType dtype = DT_INVALID;
    switch (stream_decoder_->bits_per_sample) {
      case 16:
        dtype = DT_INT16;
        break;
      default:
        return errors::InvalidArgument("invalid_bits_per_sample: ",
                                       stream_decoder_->bits_per_sample);
    }

    shape_ = TensorShape({samples, channels});
    dtype_ = dtype;
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

    stream_decoder_->SetTensor(sample_start, value);
    if (!FLAC__stream_decoder_seek_absolute(decoder_.get(), sample_start)) {
      return errors::InvalidArgument("unable to seek to: ", sample_start);
    }

    while (stream_decoder_->sample_index < sample_stop) {
      if (!FLAC__stream_decoder_process_single(decoder_.get())) {
        return errors::InvalidArgument("unable to read at: ",
                                       stream_decoder_->sample_index);
      }
    }
    return Status::OK();
  }
  string DebugString() const override { return "FlacReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  int64 rate_;

  std::unique_ptr<FLAC__StreamDecoder, void (*)(FLAC__StreamDecoder*)> decoder_;
  std::unique_ptr<FlacStreamDecoder> stream_decoder_;
};

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
  MP3ReadableResource(Env* env) : env_(env) {}
  ~MP3ReadableResource() {}

  Status Init(const string& input) override {
    mutex_lock l(mu_);

    const string& filename = input;
    file_.reset(new SizedRandomAccessFile(env_, filename, nullptr, 0));
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
    int64 samples = mp3dec_ex_.samples / mp3dec_ex_.info.channels;
    int64 channels = mp3dec_ex_.info.channels;
    int64 rate = mp3dec_ex_.info.hz;

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
    size_t returned = mp3dec_ex_read(&mp3dec_ex_, value->flat<int16>().data(),
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
    } else if (memcmp(header, "fLaC", 4) == 0) {
      resource_.reset(new FlacReadableResource(env_));
    } else {
      resource_.reset(new MP3ReadableResource(env_));
    }
    return resource_->Init(input);
  }
  Status Spec(TensorShape* shape, DataType* dtype, int32* rate) override {
    mutex_lock l(mu_);
    return resource_->Spec(shape, dtype, rate);
  }
  Status Read(const int64 start, const int64 stop,
              std::function<Status(const TensorShape& shape, Tensor** value)>
                  allocate_func) override {
    mutex_lock l(mu_);
    return resource_->Read(start, stop, allocate_func);
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

    OP_REQUIRES_OK(
        context,
        resource->Read(start, stop,
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

class AudioResampleOp : public OpKernel {
 public:
  explicit AudioResampleOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("quality", &quality_));
    OP_REQUIRES(
        context,
        (SPEEX_RESAMPLER_QUALITY_MIN <= quality_ &&
         quality_ <= SPEEX_RESAMPLER_QUALITY_MAX),
        errors::InvalidArgument("quality ", quality_, " not supported, need [",
                                SPEEX_RESAMPLER_QUALITY_MIN, ", ",
                                SPEEX_RESAMPLER_QUALITY_MAX, "]"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* rate_in_tensor;
    OP_REQUIRES_OK(context, context->input("rate_in", &rate_in_tensor));
    const int64 rate_in = rate_in_tensor->scalar<int64>()();

    const Tensor* rate_out_tensor;
    OP_REQUIRES_OK(context, context->input("rate_out", &rate_out_tensor));
    const int64 rate_out = rate_out_tensor->scalar<int64>()();

    int64 samples_in = input_tensor->shape().dim_size(0);
    int64 channels = input_tensor->shape().dim_size(1);

    std::unique_ptr<SpeexResamplerState, void (*)(SpeexResamplerState*)> state(
        nullptr, [](SpeexResamplerState* p) {
          if (p != nullptr) {
            speex_resampler_destroy(p);
          }
        });

    int err = 0;
    state.reset(
        speex_resampler_init(channels, rate_in, rate_out, quality_, &err));
    OP_REQUIRES(
        context, (state.get() != nullptr),
        errors::InvalidArgument("unable to initialize resampler: ", err));

    int64 samples_out = samples_in * rate_out / rate_in;
    Tensor* output_tensor;
    switch (input_tensor->dtype()) {
      case DT_INT16: {
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, TensorShape({samples_out, channels}),
                                    &output_tensor));

        uint32_t processed_in = samples_in;
        uint32_t processed_out = samples_out;
        int returned = speex_resampler_process_interleaved_int(
            state.get(), input_tensor->flat<int16>().data(), &processed_in,
            output_tensor->flat<int16>().data(), &processed_out);
        OP_REQUIRES(context, (returned == 0),
                    errors::InvalidArgument("process error: ", returned));
        OP_REQUIRES(
            context, (processed_out == samples_out),
            errors::InvalidArgument("output buffer mismatch: ", processed_out,
                                    " vs. ", samples_out));
      } break;
      case DT_FLOAT: {
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, TensorShape({samples_out, channels}),
                                    &output_tensor));
        uint32_t processed_in = samples_in;
        uint32_t processed_out = samples_out;
        int returned = speex_resampler_process_interleaved_float(
            state.get(), input_tensor->flat<float>().data(), &processed_in,
            output_tensor->flat<float>().data(), &processed_out);
        OP_REQUIRES(context, (returned == 0),
                    errors::InvalidArgument("process error: ", returned));
        OP_REQUIRES(
            context, (processed_out == samples_out),
            errors::InvalidArgument("output buffer mismatch: ", processed_out,
                                    " vs. ", samples_out));
      } break;
      default:
        OP_REQUIRES_OK(context,
                       errors::InvalidArgument(
                           "Data type ", DataTypeString(input_tensor->dtype()),
                           " not supported"));
    }
  }

 private:
  int64 quality_;
};

REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableInit").Device(DEVICE_CPU),
                        AudioReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableSpec").Device(DEVICE_CPU),
                        AudioReadableSpecOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioReadableRead").Device(DEVICE_CPU),
                        AudioReadableReadOp);

REGISTER_KERNEL_BUILDER(Name("IO>AudioResample").Device(DEVICE_CPU),
                        AudioResampleOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
