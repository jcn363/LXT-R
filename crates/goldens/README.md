# Golden Test Data

This directory holds `.safetensors` files containing reference tensor outputs
from the Python LTX-2.3 implementation. Rust tests load these files and compare
their output against the Python values using `ltx-test_utils::assert_allclose`.

## Generating Golden Data

Run the Python reference pipeline with `save_golden=True` to produce `.safetensors`
files. Each file contains named tensors that correspond to specific module outputs.

Example (when Python reference code is available):

```python
import torch
from safetensors.torch import save_file

# Run the module
module = RMSNorm(dim=64)
x = torch.randn(1, 8, 64)
output = module(x)

# Save golden data
save_file({"input": x, "output": output}, "crates/goldens/rms_norm.safetensors")
```

## Adding a New Golden Test

1. Generate the `.safetensors` file from Python
2. Place it in this directory
3. Add a test in the relevant crate's `tests/` directory:

```rust
use ltx_test_utils::{load_golden, assert_allclose_default};

#[test]
fn test_golden_my_module() {
    let input = load_golden("crates/goldens/my_module.safetensors", "input");
    let expected = load_golden("crates/goldens/my_module.safetensors", "output");

    let actual = my_module(&input);
    assert_allclose_default(&actual, &expected);
}
```
