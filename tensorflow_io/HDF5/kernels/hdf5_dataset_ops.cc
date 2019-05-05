#include "kernels/dataset_ops.h"
#include "H5Cpp.h"
#include "boost/multi_array.hpp"
class H5InputStream{
public:
  explicit H5InputStream(const string& filename, const string& dataset)
  :h5file_(nullptr),
  dataset_(nullptr),
  filespace_(nullptr){
    std::vector<string> result;
    try{
      h5file_ = std::make_unique<H5File>(filename, H5F_ACC_RDONLY);
      dataset_ = std::move(new DataSet(h5file_->openDataSet(H5std_string(dataset))));
      filespace_ = std::move(new DataSpace(h5file_->getSpace()));
    }catch(FileIException error){
      throw errors::InvalidArgument("Error Opening File: " + filename);
    }
    catch(IGroupException error){
    throw errors::InvalidArgument("Error Opening Dataset: "+ dataset); 
    }
  }
  ~H5InputStream(){
    if(h5file_ != nullptr){
      h5file_->close();
    }
    dataset_ = nullptr;
  }
  Status ReadRecord(Tensor* out){
    DataType dtype = dataset_->getDataType();
    int rank_ = filespace_->getSimpleExtentNdims();
    hsize_t dims[rank_];
    filespace_->getSimpleExtentDims(dims);
    hid_t type_ = H5Tget_native_type(dtype.getId(), H5T_DIR_ASCEND);
    // Checking for integer
    boost::multi_array<string, rank_> array_type;
    if(H5Tequal(type_, H5T_NATIVE_INT)){
        boost::multi_array<int, rank_> array_type;
      }
    //Checking for uint
    if(H5Tequal(type_, H5T_NATIVE_UINT))
      boost::multi_array<unsigned int, rank_> array_type;
    //Checking for short
    if(H5Tequal(type_, H5T_NATIVE_SHORT))
        boost::multi_array<short int, rank_> array_type;
    //Checking for unsigned short
    if(H5Tequal(type_, H5T_NATIVE_USHORT))
        boost::multi_array<unsigned short int, rank_> array_type;
    //Checking for long
    if(H5Tequal(type_, H5T_NATIVE_LONG))
        boost::multi_array<long int, rank_> array_type;
    //Checking for unsigned long
    if(H5Tequal(type_, H5T_NATIVE_ULONG))
      boost::multi_array<unsigned long int, rank_> array_type;
    //Checking for long long
    if(H5Tequal(type_, H5T_NATIVE_LLONG))
      boost::multi_array<long long int, rank_> array_type;
    //Checking for unsigned long long
    if(H5Tequal(type_, H5T_NATIVE_ULLONG))
      boost::multi_array<unsigned long long int, rank_> array_type;
    //Checking for float
    if(H5Tequal(type_, H5T_NATIVE_FLOAT))
      boost::multi_array<float, rank_> array_type;
    //Checking for double
    if(H5Tequal(type_, H5T_NATIVE_DOUBLE))
      boost::multi_array<double, rank_> array_type;
    //Checking for Long Double
    if(H5Tequal(type_, H5T_NATIVE_LDOUBLE))
      boost::multi_array<long double, rank_> array_type;
    
    boost::array<array::index, rank_> shape = dims;
    array_type data(shape_);
    DataSpace memspace(rank_, dims);
    dataset_->read(data.data(), dtype,memspace, *filespace_);
    memcpy(flat_<std::remove_pointer<data.data()>::type>.data(), data.data(), sizeof(data.data()));
    return Status:OK();
  }
}
