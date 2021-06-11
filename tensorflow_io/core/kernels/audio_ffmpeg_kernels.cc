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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_io/core/kernels/io_stream.h"

extern "C" {

#include <dlfcn.h>

#include "libavcodec/avcodec.h"
}

namespace tensorflow {
namespace data {
void FFmpegInit();
class DecodeAACFunctionState {
 public:
  DecodeAACFunctionState(const int64 codec)
      : codec_parser_context_(nullptr, [](AVCodecParserContext* p) {
          if (p != nullptr) {
            av_parser_close(p);
          }
        }) {
    codec_ = avcodec_find_decoder(AV_CODEC_ID_AAC);
    if (codec_ != nullptr) {
      AVCodecParserContext* codec_parser_context = av_parser_init(codec_->id);
      if (codec_parser_context != nullptr) {
        codec_parser_context_.reset(codec_parser_context);
        return;
      }
      av_parser_close(codec_parser_context);
    }
  }
  ~DecodeAACFunctionState() {}
  bool Valid() {
    if (codec_parser_context_.get() == nullptr) {
      return false;
    }
    return true;
  }
  int64 Call(const int64 rate, const int64 channels, const char* data_in_chunk,
             const int64_t* size_in_chunk, int64_t chunk, int64_t frames,
             char* data_out, int64_t size_out) {
    std::unique_ptr<AVCodecContext, void (*)(AVCodecContext*)> codec_context(
        nullptr, [](AVCodecContext* p) {
          if (p != nullptr) {
            avcodec_free_context(&p);
          }
        });
    codec_context.reset(avcodec_alloc_context3(codec_));
    if (codec_context.get() == nullptr) {
      LOG(ERROR) << "unable to create codec context";
      return -1;
    }
    codec_context->channels = channels;
    codec_context->sample_rate = rate;
    if (avcodec_open2(codec_context.get(), codec_, NULL) < 0) {
      LOG(ERROR) << "unable to open codec context";
      return -1;
    }
    std::unique_ptr<AVPacket, void (*)(AVPacket*)> packet(
        nullptr, [](AVPacket* p) {
          if (p != nullptr) {
            av_packet_free(&p);
          }
        });
    packet.reset(av_packet_alloc());
    if (packet.get() == nullptr) {
      LOG(ERROR) << "unable to create packet";
      return -1;
    }
    std::unique_ptr<AVFrame, void (*)(AVFrame*)> frame(nullptr, [](AVFrame* p) {
      if (p != nullptr) {
        av_frame_free(&p);
      }
    });
    frame.reset(av_frame_alloc());
    if (frame.get() == nullptr) {
      LOG(ERROR) << "unable to create frame";
      return -1;
    }
    int64 offset = 0;
    int64 offset_in = 0;
    for (int64_t i = 0; i < chunk; i++) {
      const int64_t size_in = size_in_chunk[i];
      const char* data_in = &data_in_chunk[offset_in];
      offset_in += size_in;
      int ret = av_parser_parse2(codec_parser_context_.get(),
                                 codec_context.get(), &packet->data,
                                 &packet->size, (const uint8_t*)data_in,
                                 size_in, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
      if (ret < 0) {
        LOG(ERROR) << "unable to parse: " << ret;
        return ret;
      }
      if (ret != size_in) {
        LOG(ERROR) << "size does not match: " << ret << " vs. " << size_in;
        return -1;
      }
      if (packet->size > 0) {
        int64 size_returned = 0;
        ret = Decode(codec_context.get(), packet.get(), frame.get(), channels,
                     &data_out[offset], size_out - offset, &size_returned);
        if (ret < 0) {
          LOG(ERROR) << "unable to decode: " << ret;
          return ret;
        }
        offset += size_returned;
      }
    }

    packet->data = nullptr;
    packet->size = 0;
    int64 size_returned = 0;
    int ret = Decode(codec_context.get(), packet.get(), frame.get(), channels,
                     &data_out[offset], size_out - offset, &size_returned);
    if (ret < 0) {
      LOG(ERROR) << "unable to decode and flush out: " << ret;
      return ret;
    }
    offset += size_returned;
    if (offset != size_out) {
      LOG(WARNING) << "output mismatch: " << offset << " vs. " << size_out
                   << ret;
    }
    return 0;
  }
  int Decode(AVCodecContext* codec_context, AVPacket* packet, AVFrame* frame,
             int64 channels, void* data_out, int64 size_out, int64* offset) {
    int ret = avcodec_send_packet(codec_context, packet);
    if (ret < 0) {
      LOG(ERROR) << "unable to send packet: " << ret;
      return ret;
    }
    while (ret >= 0) {
      ret = avcodec_receive_frame(codec_context, frame);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return 0;
      } else if (ret < 0) {
        LOG(ERROR) << "unable to receive frame: " << ret;
        return ret;
      }
      size_t data_size = av_get_bytes_per_sample(codec_context->sample_fmt);
      if (codec_context->sample_fmt != AV_SAMPLE_FMT_FLTP) {
        LOG(ERROR) << "format not supported: " << codec_context->sample_fmt;
        return -1;
      }
      if (codec_context->channels != channels) {
        LOG(ERROR) << "channels mismatch: " << codec_context->channels
                   << " vs. " << channels;
        ;
        return -1;
      }
      if (data_size < 0) {
        LOG(ERROR) << "unable to get data size: " << data_size;
        return data_size;
      }
      if (((*offset) + frame->nb_samples * codec_context->channels *
                           data_size) > size_out) {
        LOG(WARNING) << "data out run: "
                     << ((*offset) + frame->nb_samples *
                                         codec_context->channels * data_size)
                     << " vs. " << size_out;
      }
      for (int64 i = 0; i < frame->nb_samples; i++) {
        for (int64 ch = 0; ch < codec_context->channels; ch++) {
          if (((*offset) + data_size) <= size_out) {
            memcpy((char*)data_out + (*offset), frame->data[ch] + data_size * i,
                   data_size);
          }
          (*offset) += data_size;
        }
      }
    }
    return 0;
  }

 private:
  AVCodec* codec_;
  std::unique_ptr<AVCodecParserContext, void (*)(AVCodecParserContext*)>
      codec_parser_context_;
};

class EncodeAACFunctionState {
 public:
  EncodeAACFunctionState(const int64 codec, const int64 rate,
                         const int64 channels)
      : rate_(rate),
        channels_(channels),
        codec_context_(nullptr,
                       [](AVCodecContext* p) {
                         if (p != nullptr) {
                           avcodec_free_context(&p);
                         }
                       }),
        packet_(nullptr,
                [](AVPacket* p) {
                  if (p != nullptr) {
                    av_packet_free(&p);
                  }
                }),
        frame_(nullptr, [](AVFrame* p) {
          if (p != nullptr) {
            av_frame_free(&p);
          }
        }) {
    int channel_layout = 0;
    switch (channels) {
      case 1:
        channel_layout = AV_CH_LAYOUT_MONO;
        break;
      case 2:
        channel_layout = AV_CH_LAYOUT_STEREO;
        break;
      default:
        LOG(INFO) << "aac codec does not support channels = " << channels
                  << " yet";
        return;
    }
    codec_ = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (codec_ != nullptr) {
      AVCodecContext* codec_context = avcodec_alloc_context3(codec_);
      if (codec_context != nullptr) {
        // check that the encoder supports AV_SAMPLE_FMT_FLTP input
        const enum AVSampleFormat* p = codec_->sample_fmts;
        while (*p != AV_SAMPLE_FMT_NONE) {
          if (*p == AV_SAMPLE_FMT_FLTP) {
            break;
          }
          p++;
        }
        if (*p == AV_SAMPLE_FMT_FLTP) {
          codec_context->sample_rate = rate;
          codec_context->channels = channels;
          codec_context->channel_layout = channel_layout;
          codec_context->sample_fmt = AV_SAMPLE_FMT_FLTP;
          if (avcodec_open2(codec_context, codec_, NULL) >= 0) {
            LOG(INFO) << "aac codec opened successfully";

            AVPacket* packet = av_packet_alloc();
            AVFrame* frame = av_frame_alloc();
            if (packet != nullptr && frame != nullptr) {
              codec_context_.reset(codec_context);
              packet_.reset(packet);
              frame_.reset(frame);
              return;
            }
          }
        }
        LOG(ERROR) << "unable to support AV_SAMPLE_FMT_FLTP";
      }
      avcodec_free_context(&codec_context);
    }
  }
  ~EncodeAACFunctionState() {}
  bool Valid() {
    if (codec_context_.get() == nullptr) {
      return false;
    }
    return true;
  }
  int64 Call(const float* data_in, const int64_t size_in, char** data_out_chunk,
             int64_t* size_out_chunk, int64_t* chunk) {
    frame_->nb_samples = 1024;
    frame_->format = codec_context_->sample_fmt;
    frame_->channels = codec_context_->channels;

    // allocate the data buffers
    int ret = av_frame_get_buffer(frame_.get(), 0);
    if (ret < 0) {
      return ret;
    }

    buffer_.clear();
    buffer_.reserve(*chunk);

    int64 index = 0;
    while ((index < *chunk) && (index * 1024 * channels_ < size_in)) {
      int ret = av_frame_make_writable(frame_.get());
      if (ret < 0) {
        return ret;
      }
      for (int64 c = 0; c < channels_; c++) {
        for (int64 i = 0; i < 1024; i++) {
          ((float*)frame_->data[c])[i] =
              data_in[(index * 1024 + i) * channels_ + c];
        }
      }
      ret = Encode(codec_context_.get(), packet_.get(), frame_.get(), &buffer_);
      if (ret < 0) {
        return ret;
      }
      index++;
    }
    Encode(codec_context_.get(), packet_.get(), NULL, &buffer_);

    index = 0;
    while (index < buffer_.size() && index < (*chunk)) {
      data_out_chunk[index] = &(buffer_[index])[0];
      size_out_chunk[index] = (buffer_[index]).size();
      index++;
    }
    *chunk = index;
    return 0;
  }
  int Encode(AVCodecContext* codec_context, AVPacket* packet, AVFrame* frame,
             std::vector<string>* buffer) {
    int ret = avcodec_send_frame(codec_context, frame);
    if (ret < 0) {
      return ret;
    }
    while (ret >= 0) {
      ret = avcodec_receive_packet(codec_context, packet);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return 0;
      } else if (ret < 0) {
        LOG(ERROR) << "error encoding audio frame";
        return -1;
      }
      string p;
      buffer->emplace_back(p);
      if (packet->size > 0) {
        buffer->back().resize(packet->size);
        memcpy(&(buffer->back()[0]), packet->data, packet->size);
      }
      av_packet_unref(packet);
    }
    return 0;
  }

 private:
  int64 rate_;
  int64 channels_;
  AVCodec* codec_;
  std::unique_ptr<AVCodecContext, void (*)(AVCodecContext*)> codec_context_;
  std::unique_ptr<AVPacket, void (*)(AVPacket*)> packet_;
  std::unique_ptr<AVFrame, void (*)(AVFrame*)> frame_;
  std::vector<string> buffer_;
};

}  // namespace data
}  // namespace tensorflow

extern "C" {
__attribute__((visibility("default"))) void EncodeAACFunctionFiniFFmpeg(
    void* state) {
  if (state != nullptr) {
    delete static_cast<tensorflow::data::EncodeAACFunctionState*>(state);
  }
}

__attribute__((visibility("default"))) void* EncodeAACFunctionInitFFmpeg(
    const int64_t codec, const int64_t rate, const int64_t channels) {
  tensorflow::data::FFmpegInit();
  tensorflow::data::EncodeAACFunctionState* state =
      new tensorflow::data::EncodeAACFunctionState(codec, rate, channels);
  if (state != nullptr) {
    if (state->Valid()) {
      return state;
    }
    delete state;
  }
  return nullptr;
}
__attribute__((visibility("default"))) int64_t EncodeAACFunctionCallFFmpeg(
    void* state, const float* data_in, const int64_t size_in,
    char** data_out_chunk, int64_t* size_out_chunk, int64_t* chunk) {
  if (state != nullptr) {
    return static_cast<tensorflow::data::EncodeAACFunctionState*>(state)->Call(
        data_in, size_in, data_out_chunk, size_out_chunk, chunk);
  }
  return -1;
}

__attribute__((visibility("default"))) void DecodeAACFunctionFiniFFmpeg(
    void* state) {
  if (state != nullptr) {
    delete static_cast<tensorflow::data::DecodeAACFunctionState*>(state);
  }
}

__attribute__((visibility("default"))) void* DecodeAACFunctionInitFFmpeg(
    const int64_t codec, const int64_t rate, const int64_t channels) {
  tensorflow::data::FFmpegInit();

  tensorflow::data::DecodeAACFunctionState* state =
      new tensorflow::data::DecodeAACFunctionState(codec);
  if (state != nullptr) {
    if (state->Valid()) {
      return state;
    }
    delete state;
  }
  return nullptr;
}

__attribute__((visibility("default"))) int64_t DecodeAACFunctionCallFFmpeg(
    void* state, const int64_t codec, const int64_t rate,
    const int64_t channels, const void* data_in_chunk,
    const int64_t* size_in_chunk, int64_t chunk, int64_t frames, void* data_out,
    int64_t size_out) {
  if (state != nullptr) {
    return static_cast<tensorflow::data::DecodeAACFunctionState*>(state)->Call(
        rate, channels, (const char*)data_in_chunk, size_in_chunk, chunk,
        frames, (char*)data_out, size_out);
  }
  return -1;
}
}
