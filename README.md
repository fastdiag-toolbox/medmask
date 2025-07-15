# MedMask - Medical Image Mask Processing Library

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A specialized library for efficient compression, storage, and processing of medical image segmentation masks. Designed to dramatically improve medical image analysis workflow through advanced compression and semantic design.

## 🚀 Key Benefits

- **Ultimate Compression**: 50+ compression ratio using Zstandard
- **File Management**: 117 files → 1 archive file
- **Performance**: 16x faster read operations
- **Semantic Mapping**: Built-in name-to-label conversion
- **Overlapping Masks**: Multi-granularity organ combinations
- **Lazy Loading**: Memory-efficient on-demand construction

## 📦 Installation

```bash
pip install medmask
```

**Dependencies**: Python 3.8+, numpy, spacetransformer, zstandard

## ⚡ Usage

### Basic Operations

```python
import numpy as np
from medmask import SegmentationMask, MaskArchive, save_mask, load_mask
from spacetransformer import Space

# Create spatial information 
space = Space(shape=(64, 192, 192), spacing=(2.5, 1.0, 1.0))

# -----------------------------
# Create mask – two approaches
# -----------------------------
# 1. Complete initialization
liver_data = np.zeros((64, 192, 192), dtype=np.uint8)  # (Z,Y,X)
liver_data[20:40, 50:150, 60:140] = 1
mask = SegmentationMask(liver_data, {"liver": 1}, space=space)

# 2. Lazy loading (multiple organs)
combined_mask = SegmentationMask.lazy_init(bit_depth=8, space=space)
combined_mask.add_label(liver_data > 0, label=1, name="liver")
combined_mask.add_label(spleen_data > 0, label=2, name="spleen")

# Query masks
liver_region = mask.get_binary_mask_by_names("liver")
multiple_organs = combined_mask.get_binary_mask_by_names(["liver", "spleen"])
```

### File Operations

```python
# Single mask files (.msk)
save_mask(mask, "liver.msk")
loaded_mask = load_mask("liver.msk")

# Equivalent
mask.save('liver.msk')
loaded_mask = SegmentationMask.load('liver.msk')

# Multi-mask archives (.mska) - for collections
archive = MaskArchive("organs.mska", mode="w", space=space)
archive.add_segmask(liver_mask, "liver")
archive.add_segmask(heart_mask, "heart")

# Read from archive
archive = MaskArchive("organs.mska", mode="r")
liver = archive.load_segmask("liver")
all_names = archive.all_names()
```

### Use Cases

**Single Masks**: Individual organ processing, simple storage
**Combined Masks**: Multiple organs in one mask, unified label management  
**Archives**: Collections of related masks, overlapping hierarchies (e.g., individual ribs + combined rib groups)

## 📊 Performance Comparison

| Metric | Traditional | MedMask | Improvement |
|--------|-------------|---------|-------------|
| **File Count** | 117 .nii.gz files | 1 .mska file | 117:1 |
| **Storage Size** | 5.12 MB | 92 KB | 56.7:1 |
| **Read Time** | 1.869s | 0.117s | 16.0x |
| **File Management** | Complex | Simple | ✓ |

*Test results based on TotalSegmentator 117-organ segmentation data*

## 📁 Project Structure

```
medmask/
├── core/
│   ├── segmask.py       # SegmentationMask class
│   └── mapping.py       # LabelMapping class
├── storage/
│   ├── archivefile.py   # MaskArchive class
│   └── maskfile.py      # MaskFile class
├── compression/
│   └── zstd_codec.py    # Compression implementation
└── utils/
    └── utils.py         # Utility functions
```

## 🧪 Testing

```bash
python -m pytest tests/
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Issues and Pull Requests are welcome! Please follow the standard GitHub workflow.

---

**MedMask** - Making medical image segmentation mask processing simpler and more efficient! 