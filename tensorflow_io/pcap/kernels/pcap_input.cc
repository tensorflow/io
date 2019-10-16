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

#include "kernels/dataset_ops.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"

namespace tensorflow {
namespace data {

class PcapInputStream : public io::BufferedInputStream {
public:

  const uint32_t MAGIC_NUMBER = 0xA1B2C3D4; // magic number for big endian machine with microsecond resolution
  const uint32_t MAGIC_NUMBER_REVERSED = 0xD4C3B2A1; // magic number for little endian machine with microsecond resolution

  const uint16_t PCAP_ERRBUF_SIZE = 256;
  const uint16_t PCAP_TSTAMP_PRECISION_MICRO = 0; // use timestamps with microsecond precision by default

  explicit PcapInputStream(InputStreamInterface* input_stream)
    : io::BufferedInputStream(input_stream, 256 * 1024) {
  }

  Status ReadRecord(double& timestamp, string* packet_data, int64& record_read) {
    string buffer;
    buffer.clear();

    // read packet header
    TF_RETURN_IF_ERROR(ReadNBytes(sizeof(struct PacketHeader), &buffer));
    struct PacketHeader *header = (struct PacketHeader *)buffer.data();

    if (reverse_header_byte_order) {
      // switch byte order to get accurate representation of field values
      EndianSwap(header->ts_sec);
      EndianSwap(header->ts_msec);
      EndianSwap(header->caplen);
      EndianSwap(header->orig_len);
    }

    // Combine date and time in seconds plus milliseconds offset into one composite value
    timestamp = header->ts_sec + (header->ts_msec / 1e6);

    // read packet data
    TF_RETURN_IF_ERROR(ReadNBytes(header->caplen, packet_data));

    record_read = 1; // this method reads one packet at a time from the input buffer

    return Status::OK();
  }

  Status ReadHeader() {
    string buffer;
    // read file header
    TF_RETURN_IF_ERROR(ReadNBytes(sizeof(struct PcapFileHeader), &buffer));
    struct PcapFileHeader *header = (struct PcapFileHeader *)buffer.data();

    if(!ValidateMagicNumber(header->magic_number)) {
      return errors::InvalidArgument("PCAP file must starts with a standard magic number.");
    }

    if (reverse_header_byte_order) {
        // switch byte order to get accurate representation of packet slices
        // snaplen will be needed to reconstruct sliced payloads spread across multiple pcap packets
        EndianSwap(header->snaplen);
    }
    return Status::OK();
  }

private:

  /**
    PcapFileHeader format: (https://wiki.wireshark.org/Development/LibpcapFileFormat)
    magic_number: used to detect the file format itself and the byte ordering. The writing application writes 0xa1b2c3d4 with it's native byte ordering format into this field. The reading application will read either 0xa1b2c3d4 (identical) or 0xd4c3b2a1 (swapped). If the reading application reads the swapped 0xd4c3b2a1 value, it knows that all the following fields will have to be swapped too. For nanosecond-resolution files, the writing application writes 0xa1b23c4d, with the two nibbles of the two lower-order bytes swapped, and the reading application will read either 0xa1b23c4d (identical) or 0x4d3cb2a1 (swapped).
    version_major, version_minor: the version number of this file format (current version is 2.4)
    thiszone: the correction time in seconds between GMT (UTC) and the local timezone of the following packet header timestamps. Examples: If the timestamps are in GMT (UTC), thiszone is simply 0. If the timestamps are in Central European time (Amsterdam, Berlin, ...) which is GMT + 1:00, thiszone must be -3600. In practice, time stamps are always in GMT, so thiszone is always 0.
    sigfigs: in theory, the accuracy of time stamps in the capture; in practice, all tools set it to 0
    snaplen: the "snapshot length" for the capture (typically 65535 or even more, but might be limited by the user), see: incl_len vs. orig_len below
    linktype: link-layer header type, specifying the type of headers at the beginning of the packet (e.g. 1 for Ethernet, see tcpdump.org's link-layer header types page for details); this can be various types such as 802.11, 802.11 with various radio information, PPP, Token Ring, FDDI, etc.
  */
  struct PcapFileHeader {
    uint32_t magic_number;
    uint16_t version_major;
    uint16_t version_minor;
    int32_t thiszone;
    uint32_t sigfigs;
    uint32_t snaplen;
    uint32_t linktype;
  };


  /**
    PacketHeader format:
    ts_sec: the date and time when this packet was captured. This value is in seconds since January 1, 1970 00:00:00 GMT; this is also known as a UN*X time_t. You can use the ANSI C time() function from time.h to get this value, but you might use a more optimized way to get this timestamp value. If this timestamp isn't based on GMT (UTC), use thiszone from the global header for adjustments.
    ts_msec: in regular pcap files, the microseconds when this packet was captured, as an offset to ts_sec. In nanosecond-resolution files, this is, instead, the nanoseconds when the packet was captured, as an offset to ts_sec /!\ Beware: this value shouldn't reach 1 second (in regular pcap files 1 000 000; in nanosecond-resolution files, 1 000 000 000); in this case ts_sec must be increased instead!
    caplen: the number of bytes of packet data actually captured and saved in the file. This value should never become larger than orig_len or the snaplen value of the global header.
    orig_len: the length of the packet as it appeared on the network when it was captured. If caplen and orig_len differ, the actually saved packet size was limited by snaplen.
  */
  struct PacketHeader
  {
	uint32_t ts_sec;
	uint32_t ts_msec;
	uint32_t caplen;
	uint32_t orig_len;
  };

  inline void EndianSwap(uint16_t& x)
  {
    x = (x>>8) |
        (x<<8);
  }

  inline void EndianSwap(uint32_t& x)
  {
    x = (x>>24) |
        ((x<<8) & 0x00FF0000) |
        ((x>>8) & 0x0000FF00) |
        (x<<24);
  }

  bool reverse_header_byte_order = false; // is the pcap file using little endian or big endian byte order

  /**
  Check for the magic numbers of the two most typical pcap formats used in practice.
  */
  bool ValidateMagicNumber(uint32_t magic_number) {
        if (magic_number == MAGIC_NUMBER) {
            return true;
        } else if (magic_number == MAGIC_NUMBER_REVERSED) {
            reverse_header_byte_order = true;
            return true;
        } else {
            return false;
        }
  }

}; // end of class PcapInputStream


class PcapInput: public FileInput<PcapInputStream> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<PcapInputStream>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new PcapInputStream(s));
      TF_RETURN_IF_ERROR(state.get()->ReadHeader());
    }
    // Let's allocate enough space for Tensor, if more than read, replace.
    // The output tensor has two columns (packet_timestamp,packet_data).
    // Hence the shape of the output tensor is (record_to_read,2) unless there are less than record_to_read packets left in the file
    Tensor tensor_packet_ts(ctx->allocator({}), DT_DOUBLE, {record_to_read}); // Tensor column for packet timestamps
    out_tensors->emplace_back(std::move(tensor_packet_ts)); // add timestamp column to the output tensor
    Tensor tensor_packet_data(ctx->allocator({}), DT_STRING, {record_to_read}); // Tensor column for packet data
    out_tensors->emplace_back(std::move(tensor_packet_data)); // add data column to the output tensor

    // read packets from the file up to record_to_read or end of file
    while ((*record_read) < record_to_read) {
      int64 record_count = 0;
      double packet_timestamp;
      string packet_data_buffer;
      Status status = state.get()->ReadRecord(packet_timestamp, &packet_data_buffer, record_count);
      if (!(status.ok() || errors::IsOutOfRange(status))) {
        return status;
      }
      if (record_count > 0) {
        Tensor timestamp_tensor = (*out_tensors)[0];
        timestamp_tensor.flat<double>()(*record_read) = packet_timestamp;
        Tensor data_tensor = (*out_tensors)[1];
        data_tensor.flat<string>()(*record_read) = std::move(packet_data_buffer);
        (*record_read) += record_count;
      } else {
        // no more records available to read
        // record_count == 0
        break;
      }
    }
    return Status::OK();
  }
  Status FromStream(io::InputStreamInterface* s) override {
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    return true;
  }
 protected:
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(PcapInput, "tensorflow::data::PcapInput");

REGISTER_KERNEL_BUILDER(Name("IoPcapInput").Device(DEVICE_CPU),
                        FileInputOp<PcapInput>);
REGISTER_KERNEL_BUILDER(Name("IoPcapDataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<PcapInput, PcapInputStream>);

}  // namespace data
}  // namespace tensorflow
