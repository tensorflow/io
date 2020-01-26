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

#include "tensorflow_io/core/kernels/azfs_kernels.h"
#include "gtest/gtest.h"

#define EXPECT_OK(val) EXPECT_EQ(val, Status::OK())

namespace tensorflow {
namespace io {
namespace {

using namespace tensorflow::io;

class AzBlobFileSystemTest : public ::testing::Test {
 protected:
  AzBlobFileSystemTest() {}

  std::string PathTo(const std::string& path) {
    return "az://devstoreaccount1/aztest" + path;
  }

  Status WriteString(const std::string& fname, const std::string& content) {
    std::unique_ptr<WritableFile> writer;
    TF_RETURN_IF_ERROR(fs.NewWritableFile(fname, &writer));
    TF_RETURN_IF_ERROR(writer->Append(content));
    TF_RETURN_IF_ERROR(writer->Close());
    return Status::OK();
  }

  Status ReadAll(const std::string& fname, std::string* content) {
    std::unique_ptr<RandomAccessFile> reader;
    TF_RETURN_IF_ERROR(fs.NewRandomAccessFile(fname, &reader));

    uint64 file_size = 0;
    TF_RETURN_IF_ERROR(fs.GetFileSize(fname, &file_size));

    StringPiece result;
    char* strdata = &(*content)[0];
    TF_RETURN_IF_ERROR(reader->Read(0, file_size, &result, strdata));
    if (file_size != result.size()) {
      return errors::DataLoss("expected ", file_size, " got ", result.size(),
                              " bytes");
    }

    *content = std::string(result);

    return Status::OK();
  }

  void SetUp() override {
    // Create container
    (void)fs.CreateDir(PathTo(""));
  }

  void TearDown() override {
    // Delete container
    // fs.DeleteDir(PathTo(""));
  }

  AzBlobFileSystem fs;
};

TEST_F(AzBlobFileSystemTest, ContainerShouldBeDirectory) {
  auto container_path = PathTo("");
  auto is_dir = fs.IsDirectory(container_path);
  EXPECT_EQ(is_dir, Status::OK());
}

TEST_F(AzBlobFileSystemTest, NewRandomAccessFile) {
  const std::string fname = PathTo("/RandomAccessFile");
  const std::string content = "abcdefghijklmn";

  size_t size = 4, offset = 2;
  const auto content_substring = content.substr(offset, size);

  EXPECT_OK(WriteString(fname, content));

  std::unique_ptr<RandomAccessFile> reader;
  EXPECT_OK(fs.NewRandomAccessFile(fname, &reader));

  StringPiece result;
  EXPECT_OK(reader->Read(0, content.size(), &result, nullptr));
  EXPECT_EQ(content, result);

  EXPECT_OK(reader->Read(offset, size, &result, nullptr));
  EXPECT_EQ(content_substring, result);
}

TEST_F(AzBlobFileSystemTest, NewWritableFile) {
  std::unique_ptr<WritableFile> writer;
  const std::string fname = PathTo("/WritableFile");
  EXPECT_OK(fs.NewWritableFile(fname, &writer));
  EXPECT_OK(writer->Append("content1,"));
  EXPECT_OK(writer->Append("content2"));
  EXPECT_OK(writer->Flush());
  EXPECT_OK(writer->Sync());
  EXPECT_OK(writer->Close());

  std::string content;
  EXPECT_OK(ReadAll(fname, &content));
  EXPECT_EQ("content1,content2", content);
}

TEST_F(AzBlobFileSystemTest, NewAppendableFile) {
  std::unique_ptr<WritableFile> writer;

  const std::string fname = PathTo("/AppendableFile");
  EXPECT_OK(WriteString(fname, "test"));

  EXPECT_OK(fs.NewAppendableFile(fname, &writer));
  EXPECT_OK(writer->Append("content"));
  EXPECT_OK(writer->Close());
}

TEST_F(AzBlobFileSystemTest, NewReadOnlyMemoryRegionFromFile) {
  const auto fname = PathTo("/MemoryFile");
  const auto content = "content";
  EXPECT_OK(WriteString(fname, content));
  std::unique_ptr<ReadOnlyMemoryRegion> region;
  EXPECT_OK(fs.NewReadOnlyMemoryRegionFromFile(fname, &region));

  const char* data = static_cast<const char*>(region->data());
  auto length = region->length();
  std::string read_content(data, length);
  EXPECT_EQ(content, read_content);
}

TEST_F(AzBlobFileSystemTest, FileExists) {
  const std::string fname = PathTo("/FileExists");
  EXPECT_EQ(error::Code::NOT_FOUND, fs.FileExists(fname).code());
  EXPECT_OK(WriteString(fname, "test"));
  EXPECT_OK(fs.FileExists(fname));
  EXPECT_OK(fs.DeleteFile(fname));
}

TEST_F(AzBlobFileSystemTest, GetChildren) {
  const auto base = PathTo("/GetChildren");
  EXPECT_OK(fs.CreateDir(base));

  const auto file = base + "/TestFile.csv";
  EXPECT_OK(WriteString(file, "test"));

  const auto subdir = base + "/SubDir";
  EXPECT_OK(fs.CreateDir(subdir));
  const auto subfile = subdir + "/TestSubFile.csv";
  EXPECT_OK(WriteString(subfile, "test"));

  std::vector<string> children;
  EXPECT_OK(fs.GetChildren(base, &children));
  std::sort(children.begin(), children.end());
  EXPECT_EQ(std::vector<string>({"SubDir", "TestFile.csv"}), children);
}

TEST_F(AzBlobFileSystemTest, DeleteFile) {
  const std::string fname = PathTo("/DeleteFile");
  EXPECT_OK(WriteString(fname, "test"));
  EXPECT_OK(fs.DeleteFile(fname));
}

TEST_F(AzBlobFileSystemTest, DeleteRecursively) {
  const std::string fname = PathTo("/recursive");

  for (const auto& ext : {".txt", ".md"}) {
    for (int i = 0; i < 3; ++i) {
      const auto this_fname = fname + "/" + std::to_string(i) + ext;
      (void)WriteString(this_fname, "");
    }
  }

  std::vector<std::string> txt_files;
  (void)fs.GetMatchingPaths(fname + "/*.txt", &txt_files);
  EXPECT_EQ(3, txt_files.size());

  int64 undeleted_files, undeleted_dirs;
  EXPECT_OK(fs.DeleteRecursively(fname, &undeleted_files, &undeleted_dirs));
}

TEST_F(AzBlobFileSystemTest, GetFileSize) {
  const std::string fname = PathTo("/GetFileSize");
  EXPECT_OK(WriteString(fname, "test"));
  uint64 file_size = 0;
  EXPECT_OK(fs.GetFileSize(fname, &file_size));
  EXPECT_EQ(4, file_size);
}

TEST_F(AzBlobFileSystemTest, CreateDir) {
  const auto dir = PathTo("/CreateDir");
  EXPECT_OK(fs.CreateDir(dir));

  const auto file = dir + "/CreateDirFile.csv";
  EXPECT_OK(WriteString(file, "test"));
  FileStatistics stat;
  EXPECT_OK(fs.Stat(dir, &stat));
  EXPECT_TRUE(stat.is_directory);
}

TEST_F(AzBlobFileSystemTest, DeleteDir) {
  const auto dir = PathTo("/DeleteDir");
  const auto file = dir + "/DeleteDirFile.csv";
  EXPECT_OK(WriteString(file, "test"));
  EXPECT_OK(fs.DeleteDir(dir));

  FileStatistics stat;
  // Still OK here as virtual directories always exist
  EXPECT_OK(fs.Stat(dir, &stat));
}

TEST_F(AzBlobFileSystemTest, RenameFile) {
  const auto fname1 = PathTo("/RenameFile1");
  const auto fname2 = PathTo("/RenameFile2");
  EXPECT_OK(WriteString(fname1, "test"));
  EXPECT_OK(fs.RenameFile(fname1, fname2));
  std::string content;
  EXPECT_OK(ReadAll(fname2, &content));
  EXPECT_EQ("test", content);
}

TEST_F(AzBlobFileSystemTest, RenameFile_Overwrite) {
  const auto fname1 = PathTo("/RenameFile1");
  const auto fname2 = PathTo("/RenameFile2");

  EXPECT_OK(WriteString(fname2, "test"));
  EXPECT_OK(fs.FileExists(fname2));

  EXPECT_OK(WriteString(fname1, "test"));
  EXPECT_OK(fs.RenameFile(fname1, fname2));
  std::string content;
  EXPECT_OK(ReadAll(fname2, &content));
  EXPECT_EQ("test", content);
}

TEST_F(AzBlobFileSystemTest, StatFile) {
  const auto fname = PathTo("/StatFile");
  EXPECT_OK(WriteString(fname, "test"));
  FileStatistics stat;
  EXPECT_OK(fs.Stat(fname, &stat));
  EXPECT_EQ(4, stat.length);
  EXPECT_FALSE(stat.is_directory);
}

TEST_F(AzBlobFileSystemTest, GetMatchingPaths_NoWildcard) {
  const auto fname = PathTo("/path/subpath/file2.txt");

  EXPECT_OK(WriteString(fname, "test"));
  std::vector<std::string> results;
  EXPECT_OK(fs.GetMatchingPaths(fname, &results));
  EXPECT_EQ(std::vector<std::string>({fname}), results);
}

TEST_F(AzBlobFileSystemTest, GetMatchingPaths_FilenameWildcard) {
  const auto fname1 = PathTo("/path/subpath/file1.txt");
  const auto fname2 = PathTo("/path/subpath/file2.txt");
  const auto fname3 = PathTo("/path/subpath/another.txt");

  EXPECT_OK(WriteString(fname1, "test"));
  EXPECT_OK(WriteString(fname2, "test"));
  EXPECT_OK(WriteString(fname3, "test"));

  const auto pattern = PathTo("/path/subpath/file*.txt");
  std::vector<std::string> results;
  EXPECT_OK(fs.GetMatchingPaths(pattern, &results));
  EXPECT_EQ(std::vector<std::string>({fname1, fname2}), results);
}

}  // namespace
}  // namespace io
}  // namespace tensorflow
