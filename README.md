# MedMask - Medical Image Mask Compression and Processing Library

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/yourusername/medmask)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

MedMask is a specialized library for efficient compression, storage, and processing of medical image segmentation masks. It significantly improves the efficiency of medical image analysis through advanced compression techniques, semantic design, and optimized I/O operations.

## ✨ Core Advantages

### 🗜️ Ultimate Compression
- **50+ compression ratio** - Using Zstandard compression algorithm
- **117 files → 1 archive** - Simplified file management
- **KB-level storage** - Dramatically reduce storage space requirements

### ⚡ Performance Boost  
- **16x read acceleration** - Optimized batch I/O operations
- **Semantic queries** - Direct access through organ names
- **Lazy loading design** - On-demand mask data construction

### 🎯 Professional Design
- **Medical scene optimization** - Specifically designed for medical image segmentation
- **Overlapping mask support** - Multi-granularity organ combination analysis
- **Spatial information preservation** - Complete retention of geometric transformation information
- **Bidirectional mapping** - Seamless name↔label conversion

## 📦 Installation

```bash
pip install medmask
```

### Dependencies
- Python 3.8+
- numpy
- spacetransformer
- zstandard (optional, for compression)

## 🚀 Quick Start

### Basic Usage

```python
import numpy as np
from medmask import SegmentationMask, MaskArchive
from spacetransformer import Space

# Create spatial information
space = Space(shape=(192, 192, 64), spacing=(1.0, 1.0, 2.5))

# Method 1: Complete initialization
liver_mask = np.zeros((64, 192, 192), dtype=np.uint8)
liver_mask[20:40, 50:150, 60:140] = 1

mask = SegmentationMask(
    mask_array=liver_mask,
    mapping={"liver": 1},
    space=space
)

# Method 2: Lazy loading construction
mask = SegmentationMask.lazy_init(bit_depth=8, space=space)
mask.add_label(liver_data > 0, label=1, name="liver")
mask.add_label(spleen_data > 0, label=2, name="spleen")
```

### Archive Management

```python
# Create archive
archive = MaskArchive("organs.mska", mode="w", space=space)

# Add multiple masks
archive.add_segmask(liver_mask, "liver")
archive.add_segmask(heart_mask, "heart")
archive.add_segmask(lung_mask, "lung")

# Read archive
archive = MaskArchive("organs.mska", mode="r")
liver = archive.load_segmask("liver")
all_organs = archive.read_all_masks()
```

### Overlapping Mask Example

```python
# Multi-level rib archive
ribs_archive = MaskArchive("ribs.mska", mode="w", space=space)

# Individual ribs
for i in range(1, 13):
    left_rib = load_rib_data(f"rib_left_{i}")
    ribs_archive.add_segmask(left_rib, f"left_rib_{i}")

# Combined masks
all_left_ribs = combine_masks([f"left_rib_{i}" for i in range(1, 13)])
ribs_archive.add_segmask(all_left_ribs, "left_ribs")

# Flexible queries
single_rib = ribs_archive.load_segmask("left_rib_5")
all_left = ribs_archive.load_segmask("left_ribs")
```

## 📊 Performance Comparison

| Metric | Traditional | MedMask | Improvement |
|--------|-------------|---------|-------------|
| **File Count** | 117 .nii.gz files | 1 .mska file | 117:1 |
| **Storage Size** | 5.12 MB | 92 KB | 56.7:1 |
| **Read Time** | 1.869s | 0.117s | 16.0x |
| **File Management** | Complex | Simple | ✓ |

*Test results based on TotalSegmentator 117-organ segmentation data*

## 🔧 API Documentation

### SegmentationMask

The core class for medical segmentation masks, supporting semantic labels and spatial information.

#### Construction Methods

```python
# Complete initialization
mask = SegmentationMask(mask_array, mapping, space=None)

# Lazy initialization  
mask = SegmentationMask.lazy_init(bit_depth, space=space)
```

#### Main Methods

```python
# Add labeled region
mask.add_label(mask_data, label, name)

# Query by name
liver_region = mask.get_binary_mask_by_names("liver")
organs = mask.get_binary_mask_by_names(["liver", "spleen"])

# Query by label
region = mask.get_binary_mask_by_labels([1, 2, 3])

# Get complete mask
all_data = mask.data
binary_data = mask.to_binary()
```

#### Properties

```python
mask.bit_depth      # Bit depth (1/8/16/32)
mask.space          # Spatial information
mask.mapping        # Name-label mapping
```

### MaskArchive

Multi-mask archive container with efficient compression storage.

#### Basic Operations

```python
# Create/open archive
archive = MaskArchive("file.mska", mode="w", space=space, codec="zstd")

# Add mask
archive.add_segmask(mask, "organ_name")

# Load mask
mask = archive.load_segmask("organ_name")

# Batch operations
all_names = archive.all_names()
all_mappings = archive.read_all_mapping()
all_masks = archive.read_all_masks()
```

### LabelMapping

Bidirectional mapping management between names and labels.

```python
from medmask import LabelMapping

mapping = LabelMapping({"liver": 1, "spleen": 2})
mapping["lung"] = 3

label = mapping["liver"]        # Get label: 1
name = mapping.get_name(1)      # Get name: "liver"
exists = mapping.has_label(1)   # Check label: True
```

## 💡 Use Cases

### Independent SegMask Approach
**Suitable for:**
- Frequent access to individual organs
- Independent modification of specific regions  
- Modular data management
- Potentially overlapping masks

**Example:**
```python
# Organ-level independent processing
liver_archive = MaskArchive("liver_variants.mska", mode="w")
liver_archive.add_segmask(normal_liver, "normal_liver")
liver_archive.add_segmask(fatty_liver, "fatty_liver")
liver_archive.add_segmask(cirrhotic_liver, "cirrhotic_liver")
```

### Combined SegMask Approach
**Suitable for:**
- Global organ analysis
- Unified label management
- Maximum storage efficiency
- Non-overlapping masks

**Example:**
```python
# Unified management of whole-body organs
combined = SegmentationMask.lazy_init(bit_depth=8, space=space)
for i, organ_name in enumerate(organ_names, 1):
    organ_data = load_organ(organ_name)
    combined.add_label(organ_data, i, organ_name)

archive.add_segmask(combined, "whole_body_organs")
```

### Overlapping Mask Applications
**Supports:**
- Multi-granularity anatomical queries
- Hierarchical organ organization  
- Flexible regional combination analysis

**Example:**
```python
# Hierarchical organization of vascular system
vessels_archive = MaskArchive("vessels.mska", mode="w")

# Fine-grained
vessels_archive.add_segmask(aorta, "aorta")
vessels_archive.add_segmask(pulmonary_artery, "pulmonary_artery")

# Medium-grained  
vessels_archive.add_segmask(arterial_system, "arterial_system")
vessels_archive.add_segmask(venous_system, "venous_system")

# Coarse-grained
vessels_archive.add_segmask(all_vessels, "vascular_system")
```

## 🔧 Advanced Usage

### Custom Compression

```python
# Use different compression algorithms
archive = MaskArchive("data.mska", mode="w", codec="lz4")  # Faster compression
archive = MaskArchive("data.mska", mode="w", codec="zstd") # Higher compression ratio
```

### Batch Processing

```python
# Batch add NIfTI files
def batch_add_nifti(archive, nifti_dir):
    for nii_file in Path(nifti_dir).glob("*.nii.gz"):
        nii = nib.load(nii_file)
        mask_data = nii.get_fdata() > 0
        organ_name = nii_file.stem.replace('.nii', '')
        
        mask = SegmentationMask(
            mask_data.astype(np.uint8),
            {organ_name: 1},
            Space.from_nifty(nii)
        )
        archive.add_segmask(mask, organ_name)
```

### Mask Operations

```python
# Mask combination operations
def combine_organs(archive, organ_names, new_name):
    combined_data = np.zeros_like(archive.load_segmask(organ_names[0]).data)
    
    for organ in organ_names:
        mask = archive.load_segmask(organ)
        combined_data |= mask.data
    
    combined_mask = SegmentationMask(
        combined_data, {new_name: 1}, archive.space
    )
    archive.add_segmask(combined_mask, new_name)
```

## 📁 Project Structure

```
medmask/
├── __init__.py           # Main package entry
├── core/                 # Core functionality
│   ├── mask.py          # SegmentationMask class
│   └── mapping.py       # LabelMapping class
├── archive/             # Archive functionality
│   └── archive_file.py  # MaskArchive class
├── compression/         # Compression module
│   └── zstd_codec.py   # Zstandard compressor
└── io/                  # Input/output utilities
    └── utils.py         # Utility functions
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific tests
python -m pytest tests/test_mask.py
python -m pytest tests/test_archive_offset_bug.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Issues and Pull Requests are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

For questions or suggestions, please contact us through:

- Submit a [GitHub Issue](https://github.com/yourusername/medmask/issues)
- Email: your.email@example.com

---

**MedMask** - Making medical image segmentation mask processing simpler and more efficient! 