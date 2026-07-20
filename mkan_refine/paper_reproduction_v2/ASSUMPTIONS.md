# Implementation assumptions

- OpenAI CLIP replaces the paper/supplied Hugging Face loader because the server has a local OpenAI ViT-L/14@336px checkpoint and network downloads are prohibited.
- B-spline default grid size 5 and order 3 are bounded assumptions, not confirmed paper constants.
- Affine LayerNorm followed by a `tanh` map into the configured grid range provides stable grid-domain normalization. Its inclusion is explicit in config and can be ablated only on validation.
- Text and vision context projections are separate KAN layers.
- The token energy scorer is shared across directions by default to stay close to supplied `inference.py`; an unshared scorer is a bounded validation-only variant.
- Classifier hidden width 512 and dropout 0.3 come from supplied `inference.py`.
- Adaptive grid updates are implemented but disabled for the first baseline. Update frequency is a bounded validation-only search dimension.
- No label smoothing, bias adjustment, image fallback, or test-informed ensemble selection is used in the baseline.
