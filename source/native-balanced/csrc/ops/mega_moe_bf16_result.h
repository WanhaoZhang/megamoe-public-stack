#pragma once

#include <tuple>

#include <torch/extension.h>

namespace ascend_deepep::ops {

// Common BF16 training contract:
//   y, local expert counts, compact state retained for backward.
using MegaMoeBF16ForwardResult = std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>;

}  // namespace ascend_deepep::ops
