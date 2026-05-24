You are fully authorized to make code modifications. Please modify src/thalamus.py and src/models_dual_stream.py to implement positional encodings as requested. Do NOT stop or halt; you must apply the edits and save the files.

Detailed implementation requirements:

1. In src/thalamus.py:
- Add `pos_encoding="none"` as an argument to:
  * SegmentEncoder.__init__ (store as self.pos_encoding)
  * ThalamusNet.__init__ (and pass it to SegmentEncoder of l1_encoders)
  * NonGatedControlNet.__init__ (and pass it to SegmentEncoder of l1_encoders)
- In SegmentEncoder.__init__, set `in_channels = 3` if `pos_encoding == "none"`, `4` if `pos_encoding == "linear"`, and `7` if `pos_encoding == "sinusoidal"`. Update `self.conv1` to use this `in_channels`.
- In SegmentEncoder.forward:
  * Add `segment_idx=None` to signature.
  * Construct coords: if `segment_idx` is not None, use `segment_idx * 32.0 + torch.arange(32, dtype=torch.float32, device=x.device)`. Else, use `torch.arange(32, dtype=torch.float32, device=x.device)`.
  * If `pos_encoding == "linear"`: pos = coords / 127.0. Expand pos to shape (B, 1, 32) and concatenate to x along channel dim (dim=1).
  * If `pos_encoding == "sinusoidal"`: pos_sin_10 = sin(coords / 10.0), pos_cos_10 = cos(coords / 10.0), pos_sin_100 = sin(coords / 100.0), pos_cos_100 = cos(coords / 100.0). Stack these to shape (4, 32), expand to (B, 4, 32) and concatenate to x along channel dim (dim=1).
- In ThalamusNet.forward and NonGatedControlNet.forward:
  * Pass `segment_idx=i` when calling `self.l1_encoders[i]` for both history and target encoding.

2. In src/models_dual_stream.py:
- Implement a helper function:
```python
def add_positional_encoding(x, pos_encoding="none"):
    if pos_encoding == "none":
        return x
    B, _, W = x.shape
    device = x.device
    coords = torch.arange(W, dtype=torch.float32, device=device)
    if pos_encoding == "linear":
        pos_linear = coords / max(1.0, float(W - 1))
        pos_linear = pos_linear.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        return torch.cat([x, pos_linear], dim=1)
    elif pos_encoding == "sinusoidal":
        pos_sin_10 = torch.sin(coords / 10.0)
        pos_cos_10 = torch.cos(coords / 10.0)
        pos_sin_100 = torch.sin(coords / 100.0)
        pos_cos_100 = torch.cos(coords / 100.0)
        pos_embeds = torch.stack([pos_sin_10, pos_cos_10, pos_sin_100, pos_cos_100], dim=0)
        pos_embeds = pos_embeds.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([x, pos_embeds], dim=1)
    return x
```
- Add `pos_encoding="none"` as an argument to NonParametricEncoder.__init__ (store as self.pos_encoding) and NonParametricJEPASpatial.__init__ (and pass it to NonParametricEncoder).
- In NonParametricEncoder.__init__, set `in_channels = 3` if `pos_encoding == "none"`, `4` if `pos_encoding == "linear"`, and `7` if `pos_encoding == "sinusoidal"`. Update `self.conv1` to use `in_channels`.
- In NonParametricEncoder.forward_spatial, apply `x = add_positional_encoding(x, self.pos_encoding)` before calling `self.conv1(x)`.
- In NonParametricJEPASpatial.clone(), propagate `pos_encoding=self.pos_encoding`.

3. Run `pytest` to make sure all existing tests still compile and pass perfectly. Verify everything compiles successfully.