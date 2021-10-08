#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BigtableTest")
    .Attr("project_id: string")
    .Attr("instance_id: string")
    .Attr("table_id: string")
    .Attr("columns: list(string) >= 1")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
});

