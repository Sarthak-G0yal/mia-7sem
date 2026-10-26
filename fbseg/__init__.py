"""Feedback-controlled adaptive smoothing for OCT defect segmentation.

The B-scan is smoothed with a per-pixel strength map lambda in [0, 1] that moves along a scale space
[raw, learned denoiser D(x), D(x) blurred with sigma 1, sigma 2]. A U-Net with a hyperspherical (cosine-prototype) head segments x_s; its
per-pixel prototype confidence and boundary score feed a small controller that updates lambda, and the
loop is unrolled for a few iterations. Smooth more where the segmenter is unsure because of noise,
smooth less where it sees a boundary.
"""

DATA_ROOT = "data/seg"
RUNS_ROOT = "runs_seg"
