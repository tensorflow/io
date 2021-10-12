#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

REGISTER_OP("BigtableTest")
    .Attr("project_id: string")
    //    .Attr("instance_id: string")
    //    .Attr("table_id: string")
    //    .Attr("columns: list(string) >= 1")
    .Output("zeroed: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

// Register the op definition for MyReaderDataset.
//
// Dataset ops always have a single output, of type `variant`, which represents
// the constructed `Dataset` object.
//
// Add any attrs and input tensors that define the dataset here.
REGISTER_OP("BigtableDataset")
    .Attr("project_id: string")
    .Attr("instance_id: string")
    .Attr("table_id: string")
    .Attr("columns: list(string) >= 1")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);