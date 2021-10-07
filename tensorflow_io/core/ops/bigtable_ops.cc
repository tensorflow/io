#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BigtableTest")
.Input("to_zero: int32")
.Output("zeroed: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
c->set_output(0, c->input(0));
return Status::OK();
});

