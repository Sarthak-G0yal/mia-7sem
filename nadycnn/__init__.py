"""NA-DyCNN: noise-aware dynamic convolution for cross-scanner retinal OCT classification.

Implementation of Chutani & Yalavarthy, J. Biomed. Opt. 31(8), 086006 (2026).
"""

CLASSES = ("CNV", "DME", "DRUSEN", "NORMAL")
NEH_CLASSES = ("CNV", "DRUSEN", "NORMAL")
NEH_CLASS_INDICES = tuple(CLASSES.index(name) for name in NEH_CLASSES)
