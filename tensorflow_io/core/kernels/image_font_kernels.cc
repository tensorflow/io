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
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

// OpenSans Regular font, Apache License 2.0
// Generated through xxd -i.
// OpenSans_Regular_ttf and  OpenSans_Regular_ttf_len
#include "third_party/font/font_opensans_regular.h"

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H

// Needed in macOS
extern "C" {
int z_verbose = 0;
void z_error(char* message) { (message); }
}

namespace tensorflow {
namespace {

std::vector<std::vector<float>> DefaultColorTable(int depth) {
  std::vector<std::vector<float>> color_table;
  color_table.emplace_back(std::vector<float>({1, 1, 0, 1}));      // 0: yellow
  color_table.emplace_back(std::vector<float>({0, 0, 1, 1}));      // 1: blue
  color_table.emplace_back(std::vector<float>({1, 0, 0, 1}));      // 2: red
  color_table.emplace_back(std::vector<float>({0, 1, 0, 1}));      // 3: lime
  color_table.emplace_back(std::vector<float>({0.5, 0, 0.5, 1}));  // 4: purple
  color_table.emplace_back(std::vector<float>({0.5, 0.5, 0, 1}));  // 5: olive
  color_table.emplace_back(std::vector<float>({0.5, 0, 0, 1}));    // 6: maroon
  color_table.emplace_back(std::vector<float>({0, 0, 0.5, 1}));  // 7: navy blue
  color_table.emplace_back(std::vector<float>({0, 1, 1, 1}));    // 8: aqua
  color_table.emplace_back(std::vector<float>({1, 0, 1, 1}));    // 9: fuchsia

  if (depth == 1) {
    for (int64 i = 0; i < color_table.size(); i++) {
      color_table[i][0] = 1;
    }
  }
  return color_table;
}

bool GlyphBit(const FT_Face& face, int64 x, int64 y) {
  int pitch = abs(face->glyph->bitmap.pitch);

  unsigned char* row = &face->glyph->bitmap.buffer[pitch * y];
  unsigned char byte = row[x >> 3];

  return (byte & (128 >> (x & 7))) != 0;
}

static FT_Library library;
Status InitializeFreeTypeLibrary() {
  static mutex init_lock(LINKER_INITIALIZED);
  static bool init(false);

  std::lock_guard<mutex> lock(init_lock);

  if (!init) {
    if (FT_Init_FreeType(&library)) {
      return errors::Internal("could not init FreeType library");
    }
    init = true;
  }
  return Status::OK();
}

}  // namespace

template <class T>
class DrawBoundingBoxesV3Op : public OpKernel {
 public:
  explicit DrawBoundingBoxesV3Op(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("font_size", &font_size_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& images = context->input(0);
    const Tensor& boxes = context->input(1);
    const int64 depth = images.dim_size(3);

    OP_REQUIRES(context, images.dims() == 4,
                errors::InvalidArgument("The rank of the images should be 4"));
    OP_REQUIRES(
        context, boxes.dims() == 3,
        errors::InvalidArgument("The rank of the boxes tensor should be 3"));
    OP_REQUIRES(context, images.dim_size(0) == boxes.dim_size(0),
                errors::InvalidArgument("The batch sizes should be the same"));

    OP_REQUIRES(
        context, depth == 4 || depth == 1 || depth == 3,
        errors::InvalidArgument("Channel depth should be either 1 (GRY), "
                                "3 (RGB), or 4 (RGBA)"));

    const int64 batch_size = images.dim_size(0);
    const int64 height = images.dim_size(1);
    const int64 width = images.dim_size(2);
    std::vector<std::vector<float>> color_table;
    if (context->num_inputs() >= 3) {
      const Tensor& colors_tensor = context->input(2);
      OP_REQUIRES(context, colors_tensor.shape().dims() == 2,
                  errors::InvalidArgument("colors must be a 2-D matrix",
                                          colors_tensor.shape().DebugString()));
      if (colors_tensor.NumElements() != 0) {
        OP_REQUIRES(
            context, colors_tensor.shape().dim_size(1) >= depth,
            errors::InvalidArgument("colors must have equal or more ",
                                    "channels than the image provided: ",
                                    colors_tensor.shape().DebugString()));
        color_table.clear();

        auto colors = colors_tensor.matrix<float>();
        for (int64 i = 0; i < colors.dimension(0); i++) {
          std::vector<float> color_value(4);
          for (int64 j = 0; j < 4; j++) {
            color_value[j] = colors(i, j);
          }
          color_table.emplace_back(color_value);
        }
      }
    }
    if (color_table.empty()) {
      color_table = DefaultColorTable(depth);
    }

    FT_Face face;
    std::vector<string> texts;
    int64 font_size = font_size_ > 0 ? font_size_ : 32;
    if (context->num_inputs() >= 4) {
      const Tensor& texts_tensor = context->input(3);
      if (texts_tensor.NumElements() > 0) {
        OP_REQUIRES(context, texts_tensor.dims() == 1,
                    errors::InvalidArgument(
                        "The rank of the texts tensor should be 1"));
        OP_REQUIRES(
            context, images.dim_size(0) == texts_tensor.dim_size(0),
            errors::InvalidArgument("The batch sizes should be the same"));

        texts.reserve(texts_tensor.NumElements());
        for (int64 i = 0; i < texts_tensor.NumElements(); ++i) {
          texts.emplace_back(texts_tensor.flat<tstring>()(i));
        }
        OP_REQUIRES_OK(context, InitializeFreeTypeLibrary());
        OP_REQUIRES(context,
                    FT_New_Memory_Face(library, OpenSans_Regular_ttf,
                                       OpenSans_Regular_ttf_len, 0, &face) == 0,
                    errors::Internal("could not init FreeType Face"));
        OP_REQUIRES(context, FT_Set_Pixel_Sizes(face, 0, font_size) == 0,
                    errors::Internal("could not set pixel size"));

        std::unordered_set<FT_ULong> bytes;
        for (int64 box_index = 0; box_index < texts.size(); box_index++) {
          for (int64 byte_index = 0; byte_index < texts[box_index].size();
               byte_index++) {
            FT_ULong byte = texts[box_index][byte_index];
            bytes.insert(byte);
          }
        }
        int64 face_max_rows = 0;
        for (FT_ULong byte : bytes) {
          FT_UInt glyph_index = FT_Get_Char_Index(face, byte);
          OP_REQUIRES(
              context,
              FT_Load_Glyph(face, glyph_index,
                            FT_LOAD_RENDER | FT_LOAD_MONOCHROME |
                                FT_LOAD_TARGET_MONO) == 0,
              errors::InvalidArgument("could not load glyph for byte: ", byte));
          OP_REQUIRES(context,
                      FT_Render_Glyph(face->glyph, FT_RENDER_MODE_MONO) == 0,
                      errors::InvalidArgument(
                          "could not render glyph for byte: ", byte));

          face_max_rows = face_max_rows > face->glyph->bitmap.rows
                              ? face_max_rows
                              : face->glyph->bitmap.rows;
        }
      }
    }

    Tensor* output;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({batch_size, height, width, depth}), &output));

    output->tensor<T, 4>() = images.tensor<T, 4>();
    auto canvas = output->tensor<T, 4>();

    for (int64 b = 0; b < batch_size; ++b) {
      const int64 num_boxes = boxes.dim_size(1);
      const auto tboxes = boxes.tensor<T, 3>();
      for (int64 bb = 0; bb < num_boxes; ++bb) {
        int64 color_index = bb % color_table.size();
        const int64 min_box_row =
            static_cast<float>(tboxes(b, bb, 0)) * (height - 1);
        const int64 min_box_row_clamp = std::max<int64>(min_box_row, int64{0});
        const int64 max_box_row =
            static_cast<float>(tboxes(b, bb, 2)) * (height - 1);
        const int64 max_box_row_clamp =
            std::min<int64>(max_box_row, height - 1);
        const int64 min_box_col =
            static_cast<float>(tboxes(b, bb, 1)) * (width - 1);
        const int64 min_box_col_clamp = std::max<int64>(min_box_col, int64{0});
        const int64 max_box_col =
            static_cast<float>(tboxes(b, bb, 3)) * (width - 1);
        const int64 max_box_col_clamp = std::min<int64>(max_box_col, width - 1);

        if (min_box_row > max_box_row || min_box_col > max_box_col) {
          LOG(WARNING) << "Bounding box (" << min_box_row << "," << min_box_col
                       << "," << max_box_row << "," << max_box_col
                       << ") is inverted and will not be drawn.";
          continue;
        }
        if (min_box_row >= height || max_box_row < 0 || min_box_col >= width ||
            max_box_col < 0) {
          LOG(WARNING) << "Bounding box (" << min_box_row << "," << min_box_col
                       << "," << max_box_row << "," << max_box_col
                       << ") is completely outside the image"
                       << " and will not be drawn.";
          continue;
        }

        // At this point, {min,max}_box_{row,col}_clamp are inside the
        // image.
        CHECK_GE(min_box_row_clamp, 0);
        CHECK_GE(max_box_row_clamp, 0);
        CHECK_LT(min_box_row_clamp, height);
        CHECK_LT(max_box_row_clamp, height);
        CHECK_GE(min_box_col_clamp, 0);
        CHECK_GE(max_box_col_clamp, 0);
        CHECK_LT(min_box_col_clamp, width);
        CHECK_LT(max_box_col_clamp, width);

        // At this point, the min_box_row and min_box_col are either
        // in the image or above/left of it, and max_box_row and
        // max_box_col are either in the image or below/right or it.
        CHECK_LT(min_box_row, height);
        CHECK_GE(max_box_row, 0);
        CHECK_LT(min_box_col, width);
        CHECK_GE(max_box_col, 0);

        // Draw top line.
        if (min_box_row >= 0) {
          for (int64 j = min_box_col_clamp; j <= max_box_col_clamp; ++j)
            for (int64 c = 0; c < depth; c++) {
              canvas(b, min_box_row, j, c) =
                  static_cast<T>(color_table[color_index][c]);
            }
        }
        // Draw bottom line.
        if (max_box_row < height) {
          for (int64 j = min_box_col_clamp; j <= max_box_col_clamp; ++j)
            for (int64 c = 0; c < depth; c++) {
              canvas(b, max_box_row, j, c) =
                  static_cast<T>(color_table[color_index][c]);
            }
        }
        // Draw left line.
        if (min_box_col >= 0) {
          for (int64 i = min_box_row_clamp; i <= max_box_row_clamp; ++i)
            for (int64 c = 0; c < depth; c++) {
              canvas(b, i, min_box_col, c) =
                  static_cast<T>(color_table[color_index][c]);
            }
        }
        // Draw right line.
        if (max_box_col < width) {
          for (int64 i = min_box_row_clamp; i <= max_box_row_clamp; ++i)
            for (int64 c = 0; c < depth; c++) {
              canvas(b, i, max_box_col, c) =
                  static_cast<T>(color_table[color_index][c]);
            }
        }

        // Draw text
        for (int64 box_index = 0; box_index < texts.size(); box_index++) {
          int64 box_col_offset = 0;
          for (int64 byte_index = 0; byte_index < texts[box_index].size();
               byte_index++) {
            FT_ULong byte = texts[box_index][byte_index];
            FT_UInt glyph_index = FT_Get_Char_Index(face, byte);
            FT_Load_Glyph(
                face, glyph_index,
                FT_LOAD_RENDER | FT_LOAD_MONOCHROME | FT_LOAD_TARGET_MONO);
            FT_Render_Glyph(face->glyph, FT_RENDER_MODE_MONO);
            for (int64 i = 0; i < face->glyph->bitmap.rows; i++) {
              for (int64 j = 0; j < face->glyph->bitmap.width; j++) {
                int64 x = j;
                int64 y = i;
                if (GlyphBit(face, x, y)) {
                  int64 col = face->glyph->bitmap_left + box_col_offset + j +
                              min_box_col;
                  int64 row =
                      i + min_box_row + font_size - face->glyph->bitmap.rows;
                  for (int64 c = 0; c < depth; c++) {
                    canvas(b, row, col, c) =
                        static_cast<T>(color_table[color_index][c]);
                  }
                }
              }
            }
            box_col_offset += (face->glyph->advance.x >> 6);
          }
        }
      }
    }
  }

 private:
  int64 font_size_;
};

#define REGISTER_CPU_KERNEL(T)                           \
  REGISTER_KERNEL_BUILDER(Name("IO>DrawBoundingBoxesV3") \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<T>("T"),   \
                          DrawBoundingBoxesV3Op<T>);

TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);

}  // namespace tensorflow
