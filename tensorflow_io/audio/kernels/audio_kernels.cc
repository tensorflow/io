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

namespace tensorflow {
namespace data {

// See http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
struct WAVHeader {
  char riff[4];              // RIFF Chunk ID: "RIFF"
  int32 riff_size;           // RIFF chunk size: 4 + n (file size - 8)
  char wave[4];              // WAVE ID: "WAVE"
  char fmt[4];               // fmt Chunk ID: "fmt "
  int32 fmt_size;            // fmt Chunk size: 16, 18, or 40
  int16 wFormatTag;          // Format code: WAVE_FORMAT_PCM (1) for PCM. WAVE_FORMAT_EXTENSIBLE (0xFFFE) for SubFormat
  int16 nChannels;           // Number of channels
  int32 nSamplesPerSec;      // Sampling rate
  int32 nAvgBytesPerSec;     // Data rate
  int16 nBlockAlign;         // Data block size (bytes)
  int16 wBitsPerSample;      // Bits per sample
};

struct ExtensionHeader {
  int16 cbSize;              // Size of the extension (0 or 22)
  int16 wValidBitsPerSample; // Number of valid bits
  int32 dwChannelMask;       // Speaker position mask
  char SubFormat[16];        // GUID
};

struct DataHeader {
  char mark[4];
  int32 size;
};
Status ValidateWAVHeader(struct WAVHeader *header) {
  if (memcmp(header->riff, "RIFF", 4) != 0) {
    return errors::InvalidArgument("WAV file must starts with `RIFF`");
  }
  if (memcmp(header->wave, "WAVE", 4) != 0) {
    return errors::InvalidArgument("WAV file must contains riff type `WAVE`");
  }
  if (memcmp(header->fmt, "fmt ", 4) != 0) {
    return errors::InvalidArgument("WAV file must contains `fmt ` mark");
  }
  if (header->fmt_size != 16 && header->fmt_size != 18 && header->fmt_size != 40) {
    return errors::InvalidArgument("WAV file must have `fmt_size ` 16, 18, or 40, received: ", header->fmt_size);
  }
  if (header->wFormatTag != 1 && header->wFormatTag != static_cast<int16>(0xFFFE)) {
    return errors::InvalidArgument("WAV file must have `wFormatTag` 1 or 0xFFFE, received: ", header->wFormatTag);
  }
  if (header->nChannels <= 0) {
    return errors::InvalidArgument("WAV file have invalide channels: ", header->nChannels);
  }
  return Status::OK();
}

class WAVReadable : public IOReadableInterface {
 public:
  WAVReadable(Env* env)
  : env_(env) {}

  ~WAVReadable() {}
  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    if (input.size() > 1) {
      return errors::InvalidArgument("more than 1 filename is not supported");
    }
    const string& filename = input[0];
    file_.reset(new SizedRandomAccessFile(env_, filename, memory_data, memory_size));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    StringPiece result;
    TF_RETURN_IF_ERROR(file_->Read(0, sizeof(header_), &result, (char *)(&header_)));

    TF_RETURN_IF_ERROR(ValidateWAVHeader(&header_));
    if (header_.riff_size + 8 != file_size_) {
      // corrupted file?
    }
    int64 filesize = header_.riff_size + 8;

    int64 position = result.size();

    if (header_.fmt_size != 16) {
      position += header_.fmt_size - 16;
    }

    int64 nSamples = 0;
    do {
      struct DataHeader head;
      TF_RETURN_IF_ERROR(file_->Read(position, sizeof(head), &result, (char *)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        // Data should be block aligned
        // bytes = nSamples * nBlockAlign
        if (head.size % header_.nBlockAlign != 0) {
          return errors::InvalidArgument("data chunk should be block aligned (", header_.nBlockAlign, "), received: ", head.size);
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
      return errors::InvalidArgument("unsupported wBitsPerSample: ", header_.wBitsPerSample);
    }

    shape_ = TensorShape({nSamples, header_.nChannels});

    return Status::OK();
  }
  Status Spec(const string& component, PartialTensorShape* shape, DataType* dtype, bool label) override {
    *shape = shape_;
    *dtype = dtype_;
    return Status::OK();
  }

  Status Extra(const string& component, std::vector<Tensor>* extra) override {
    // Expose a sample `rate`
    Tensor rate(DT_INT32, TensorShape({}));
    rate.scalar<int32>()() = header_.nSamplesPerSec;
    extra->push_back(rate);
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop, const string& component, int64* record_read, Tensor* value, Tensor* label) override {
    (*record_read) = 0;
    if (start >= shape_.dim_size(0)) {
      return Status::OK();
    }
    int64 element_start = start < shape_.dim_size(0) ? start : shape_.dim_size(0);
    int64 element_stop = stop < shape_.dim_size(0) ? stop : shape_.dim_size(0);

    if (element_start > element_stop) {
      return errors::InvalidArgument("dataset selection is out of boundary");
    }
    if (element_start == element_stop) {
      return Status::OK();
    }

    const int64 sample_start = element_start;
    const int64 sample_stop = element_stop;

    int64 sample_offset = 0;
    if (header_.riff_size + 8 != file_size_) {
      // corrupted file?
    }
    int64 filesize = header_.riff_size + 8;
    int64 position = sizeof(header_) + header_.fmt_size - 16;
    do {
      StringPiece result;
      struct DataHeader head;
      TF_RETURN_IF_ERROR(file_->Read(position, sizeof(head), &result, (char *)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        // Already checked the alignment
        int64 block_sample_start = sample_offset;
        int64 block_sample_stop = sample_offset + head.size / header_.nBlockAlign;
        // only read if block_sample_start and block_sample_stop within range
        if (sample_start < block_sample_stop && sample_stop > block_sample_start) {
          int64 read_sample_start = (block_sample_start > sample_start ? block_sample_start : sample_start);
          int64 read_sample_stop = (block_sample_stop < sample_stop ? block_sample_stop : sample_stop);
          int64 read_bytes_start = position + (read_sample_start - block_sample_start) * header_.nBlockAlign;
          int64 read_bytes_stop = position + (read_sample_stop - block_sample_start) * header_.nBlockAlign;
          string buffer;
          buffer.resize(read_bytes_stop - read_bytes_start);
          TF_RETURN_IF_ERROR(file_->Read(read_bytes_start, read_bytes_stop - read_bytes_start, &result, &buffer[0]));
          switch (header_.wBitsPerSample) {
          case 8:
            if (header_.wBitsPerSample * header_.nChannels != header_.nBlockAlign * 8) {
              return errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header_.wBitsPerSample, ", ", header_.nBlockAlign);
            }
            memcpy((char *)(value->flat<int8>().data()) + ((read_sample_start - sample_start) * header_.nBlockAlign), &buffer[0], (read_bytes_stop - read_bytes_start));
            break;
          case 16:
            if (header_.wBitsPerSample * header_.nChannels != header_.nBlockAlign * 8) {
              return errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header_.wBitsPerSample, ", ", header_.nBlockAlign);
            }
            memcpy((char *)(value->flat<int16>().data()) + ((read_sample_start - sample_start) * header_.nBlockAlign), &buffer[0], (read_bytes_stop - read_bytes_start));
            break;
          case 24:
            // NOTE: The conversion is from signed integer 24 to signed integer 32 (left shift 8 bits)
            if (header_.wBitsPerSample * header_.nChannels != header_.nBlockAlign * 8) {
              return errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header_.wBitsPerSample, ", ", header_.nBlockAlign);
            }
            for (int64 i = read_sample_start; i < read_sample_stop; i++) {
              for (int64 j = 0; j < header_.nChannels; j++) {
                char *data_p = (char *)(value->flat<int32>().data() + ((i - sample_start) * header_.nChannels + j));
                char *read_p = (char *)(&buffer[((i - read_sample_start) * header_.nBlockAlign)]) + 3 * j;
                data_p[3] = read_p[2];
                data_p[2] = read_p[1];
                data_p[1] = read_p[0];
                data_p[0] = 0x00;
              }
            }
            break;
          default:
            return errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header_.wBitsPerSample, ", ", header_.nBlockAlign);
          }
        }
        sample_offset = block_sample_stop;
      }
      position += head.size;
    } while (position < filesize);

    (*record_read) = element_stop - element_start;

    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("WAVReadable");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  struct WAVHeader header_;
};

REGISTER_KERNEL_BUILDER(Name("IO>WAVReadableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<WAVReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>WAVReadableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<WAVReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>WAVReadableRead").Device(DEVICE_CPU),
                        IOReadableReadOp<WAVReadable>);

}  // namespace data
}  // namespace tensorflow
