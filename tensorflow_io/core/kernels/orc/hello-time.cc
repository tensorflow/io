#include <ctime>
#include <iostream>
#include <orc/Exceptions.hh>
#include <orc/OrcFile.hh>
#include <orc/Reader.hh>

#include "orc/orc-config.hh"

void print_localtime() {
  std::time_t result = std::time(nullptr);
  std::cout << std::asctime(std::localtime(&result));
}

// Sample ORC file:
// https://github.com/harbby/cmake_ExternalProject_demo/blob/main/iris.orc
int main(int argc, char const *argv[]) {
  std::list<uint64_t> read_cols = {0, 1, 2, 3, 4};
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " some_orc_file.orc" << std::endl;
    return 1;
  }
  std::string file_path = argv[1];

  orc::RowReaderOptions row_reader_opts;
  row_reader_opts.include(read_cols);

  orc::ReaderOptions reader_opts;
  std::unique_ptr<orc::Reader> reader =
      orc::createReader(orc::readFile(file_path), reader_opts);
  std::unique_ptr<orc::RowReader> row_reader =
      reader->createRowReader(row_reader_opts);

  std::unique_ptr<orc::ColumnVectorBatch> batch =
      row_reader->createRowBatch(24);

  // double field
  auto *fields = dynamic_cast<orc::StructVectorBatch *>(batch.get());
  auto *col0 = dynamic_cast<orc::DoubleVectorBatch *>(fields->fields[0]);
  double *buffer1 = col0->data.data();

  // string field
  auto *col4 = dynamic_cast<orc::StringVectorBatch *>(fields->fields[4]);
  char **buffer2 = col4->data.data();
  int64_t *lengths = col4->length.data();

  while (row_reader->next(*batch)) {
    for (uint32_t r = 0; r < batch->numElements; ++r) {
      std::cout << "line " << buffer1[r] << ","
                << std::string(buffer2[r], lengths[r]) << "\n";
    }
  }

  return 0;
}
