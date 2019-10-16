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
#include <archive.h>
#include <archive_entry.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow_io/core/kernels/io_stream.h"

namespace tensorflow {
namespace data {
namespace {

class ArchiveRandomAccessFile : public SizedRandomAccessFile {
public:
  ArchiveRandomAccessFile(Env* env, const string& filename, const void* optional_memory_buff, const size_t optional_memory_size) : SizedRandomAccessFile(env, filename, optional_memory_buff, optional_memory_size) {}
  ~ArchiveRandomAccessFile() {}
  static ssize_t CallbackRead(struct archive *a, void *client_data, const void **buff) {
    class ArchiveRandomAccessFile *p = (class ArchiveRandomAccessFile *)client_data;
    StringPiece data(p->callback_read_buffer_, sizeof(p->callback_read_buffer_));
    Status s = p->Read(p->callback_read_offset_, sizeof(p->callback_read_buffer_), &data, p->callback_read_buffer_);
    if (!s.ok()) {
      if (!errors::IsOutOfRange(s)) {
        return -1;
      }
    }
    p->callback_read_offset_ += data.size();
    *buff = p->callback_read_buffer_;
    return data.size();
  }
  // CallbackRead
  char callback_read_buffer_[4096];
  int64 callback_read_offset_ = 0;
};


class ListArchiveEntriesOp : public OpKernel {
 public:
  explicit ListArchiveEntriesOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
    OP_REQUIRES_OK(context, context->GetAttr("filters", &filters_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string filename = filename_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(1);
    const string memory = memory_tensor.scalar<string>()();

    std::unique_ptr<ArchiveRandomAccessFile> file(new ArchiveRandomAccessFile(env_, filename, memory.data(), memory.size()));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    std::unique_ptr<struct archive, void(*)(struct archive *)> archive(archive_read_new(), [](struct archive *a){ archive_read_free(a);});
    for (const string& filter: filters_) {
      if (filter == "none") {
        archive_read_support_filter_none(archive.get());
        archive_read_support_format_raw(archive.get());
      }
      if (filter == "gz") {
        archive_read_support_filter_gzip(archive.get());
        archive_read_support_format_raw(archive.get());
      }
      if (filter == "tar.gz") {
        archive_read_support_filter_gzip(archive.get());
        archive_read_support_format_tar(archive.get());
      }
    }

    OP_REQUIRES(
        context, (archive_read_open(archive.get(), file.get(), NULL, ArchiveRandomAccessFile::CallbackRead, NULL) == ARCHIVE_OK),
        errors::InvalidArgument("unable to open datainput for ", filename, ": ", archive_error_string(archive.get())));

    string format;
    std::vector<string> entries;
    struct archive_entry *entry;
    while (archive_read_next_header(archive.get(), &entry) == ARCHIVE_OK) {
      string entryname = archive_entry_pathname(entry);
      entries.emplace_back(entryname);

      string archive_format(archive_format_name(archive.get()));
      string archive_filter = (archive_filter_count(archive.get()) > 0) ? archive_filter_name(archive.get(), 0) : "";
      // Find out format
      if (format == "") {
        for (const string& filter : filters_) {
          if (filter == "none") {
            if (archive_format == "raw" && archive_filter == "none") {
              format = "none";
              break;
            }
          }
          if (filter == "gz") {
            if (archive_format == "raw" && archive_filter == "gzip") {
              format = "gz";
              break;
            }
          }
          if (filter == "tar.gz") {
            if (archive_format == "GNU tar format" && archive_filter == "gzip") {
              format = "tar.gz";
              break;
            }
          }
        }
        // We are not able to find out the supported
        OP_REQUIRES(context, format != "", errors::InvalidArgument("unsupported archive: ", archive_format, "|", archive_filter));
      }
    }

    Tensor* format_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &format_tensor));
    format_tensor->scalar<string>()() = format;

    Tensor* entries_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({static_cast<int64>(entries.size())}), &entries_tensor));

    for (size_t i = 0; i < entries.size(); i++) {
        entries_tensor->flat<string>()(i) = entries[i];
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::vector<string> filters_ GUARDED_BY(mu_);
};

class ReadArchiveOp : public OpKernel {
 public:
  explicit ReadArchiveOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string filename = filename_tensor.scalar<string>()();

    const Tensor& format_tensor = context->input(1);
    const string format = format_tensor.scalar<string>()();

    const Tensor& entries_tensor = context->input(2);
    std::unordered_map<string, int64> entries;
    for (int64 i = 0; i < entries_tensor.NumElements(); i++) {
      OP_REQUIRES(context, entries.find(entries_tensor.flat<string>()(i)) == entries.end(), errors::InvalidArgument("duplicate entries: ", entries_tensor.flat<string>()(i)));
      entries[entries_tensor.flat<string>()(i)] = i;
    }

    const Tensor& memory_tensor = context->input(3);
    const string memory = memory_tensor.scalar<string>()();

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int64>(entries.size())}), &output_tensor));

    std::unique_ptr<ArchiveRandomAccessFile> file(new ArchiveRandomAccessFile(env_, filename, memory.data(), memory.size()));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    if (format == "none") {
      // Treat none as normal file.
      string output_string;
      output_string.resize(size);
      StringPiece result;
      OP_REQUIRES_OK(context, file->Read(0, size, &result, &output_string[0]));
      output_tensor->flat<string>()(0) = std::move(output_string);
      return;
    }


    std::unique_ptr<struct archive, void(*)(struct archive *)> archive(archive_read_new(), [](struct archive *a){ archive_read_free(a);});
    if (format == "gz") {
      // Treat gz file specially. Looks like libarchive always have issue
      // with text file so use ZlibInputStream. Now libarchive
      // is mostly used for archive (not compressio).
      io::RandomAccessInputStream file_stream(file.get());
      io::ZlibCompressionOptions zlib_compression_options = zlib_compression_options = io::ZlibCompressionOptions::GZIP();
      io::ZlibInputStream compression_stream(&file_stream, 65536, 65536,  zlib_compression_options);
      string output_string;
      Status status = compression_stream.ReadNBytes(INT_MAX, &output_string);
      output_tensor->flat<string>()(0) = std::move(output_string);
      return;
    }

    if (format == "tar.gz") {
      archive_read_support_filter_gzip(archive.get());
      archive_read_support_format_tar(archive.get());
    } else {
      OP_REQUIRES(context, false, errors::InvalidArgument("unsupported format: ", format));
    }

    OP_REQUIRES(
        context, (archive_read_open(archive.get(), file.get(), NULL, ArchiveRandomAccessFile::CallbackRead, NULL) == ARCHIVE_OK),
        errors::InvalidArgument("unable to open datainput for ", filename, ": ", archive_error_string(archive.get())));

    struct archive_entry *entry;
    while (archive_read_next_header(archive.get(), &entry) == ARCHIVE_OK) {
      string entryname = archive_entry_pathname(entry);
      if (entries.find(entryname) != entries.end()) {
        size_t bytes_to_read = archive_entry_size(entry);
        string output_string;
        output_string.resize(bytes_to_read);
        size_t bytes_read = 0;
        while (bytes_read < bytes_to_read) {
          ssize_t size = archive_read_data(archive.get(), &output_string[bytes_read], bytes_to_read - bytes_read);
          if (size == 0) {
            break;
          }
          bytes_read += size;
        }
        output_tensor->flat<string>()(entries[entryname]) = std::move(output_string);
      }
    }

  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IoListArchiveEntries").Device(DEVICE_CPU),
                        ListArchiveEntriesOp);

REGISTER_KERNEL_BUILDER(Name("IoReadArchive").Device(DEVICE_CPU),
                        ReadArchiveOp);


}  // namespace
}  // namespace data
}  // namespace tensorflow
