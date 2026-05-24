Modify src/thalamus.py and src/models_dual_stream.py to implement positional encodings as follows:
1. For src/thalamus.py:
- Add a `pos_encoding` parameter to SegmentEncoder, ThalamusNet, and NonGatedControlNet (default is 'none').
- Update SegmentEncoder to support linear ('linear') and sinusoidal ('sinusoidal') encodings. When segment_idx is provided, coordinates are segment_idx * 32 + local_idx.
  * For linear: 4th channel of pos_linear = coords / 127.0.
  * For sinusoidal: 4 channels of sine/cos with frequencies 10 and 100.
- Update ThalamusNet and NonGatedControlNet to pass segment_idx=i to the encoders during forward passes.
- Adjust convolutional in_channels in SegmentEncoder (3 for 'none', 4 for 'linear', 7 for 'sinusoidal').

2. For src/models_dual_stream.py:
- Implement a helper function `add_positional_encoding(x, pos_encoding="none")` where x is shape (B, 3, 128).
  * For 'linear': 4th channel of coords / 127.0.
  * For 'sinusoidal': 4 channels of sine/cos with frequencies 10 and 100.
- Add `pos_encoding` parameter to NonParametricEncoder and NonParametricJEPASpatial (default 'none').
- Update NonParametricEncoder to use pos_encoding in forward_spatial, applying `add_positional_encoding` before Conv1d.
- Adjust convolutional in_channels in NonParametricEncoder (3 for 'none', 4 for 'linear', 7 for 'sinusoidal').
- Ensure clone() in NonParametricJEPASpatial propagates the pos_encoding parameter.

3. Run pytest in the project directory to verify all compilation and tests pass successfully.