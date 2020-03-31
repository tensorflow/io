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
  int64 Call(const int64 rate, const int64 channels, const char** data_in_chunk,
             const int64_t* size_in_chunk, int64_t chunk, char* data_out,
             int64_t size_out) {
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
    for (int64_t i = 0; i < chunk; i++) {
      const char* data_in = data_in_chunk[i];
      const int64_t size_in = size_in_chunk[i];
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

}  // namespace data
}  // namespace tensorflow

extern "C" {

void DecodeAACFunctionFiniFFmpeg(void* state) {
  if (state != nullptr) {
    delete static_cast<tensorflow::data::DecodeAACFunctionState*>(state);
  }
}

void* DecodeAACFunctionInitFFmpeg(const int64_t codec, const int64_t rate,
                                  const int64_t channels) {
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

int64_t DecodeAACFunctionCallFFmpeg(void* state, const int64_t codec,
                                    const int64_t rate, const int64_t channels,
                                    const int64_t* frame_in_chunk,
                                    const void** data_in_chunk,
                                    const int64_t* size_in_chunk, int64_t chunk,
                                    void* data_out, int64_t size_out) {
  if (state != nullptr) {
    return static_cast<tensorflow::data::DecodeAACFunctionState*>(state)->Call(
        rate, channels, (const char**)data_in_chunk, size_in_chunk, chunk,
        (char*)data_out, size_out);
  }
  return -1;
}
}
