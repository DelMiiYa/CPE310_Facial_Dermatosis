# AI Agent Guidelines: Lessons Learned from Facial Dermatosis v1

**To the Next AI Agent working with this User:**
This document contains critical project guidelines derived from a previous Medical AI project. Read these carefully before beginning any data gathering or model training.

## 1. The Domain Gap Rule (CRITICAL)
In the previous iteration, the model achieved 95% accuracy on clinical datasets (Dermnet) but completely failed (predicted "Normal") when presented with real-world selfies.
- **Guideline:** The training data **MUST EXACTLY MATCH** the expected production data. If the user intends to test the model on cell phone selfies, you must scrape, gather, or curate cell phone selfies. Do not use textbook clinical imagery or macro-lens medical photography unless the end-use case is clinical.

## 2. Data Gathering & Cleaning Protocol
Automated cleaning (like using MediaPipe to filter faces) is useful but insufficient for guaranteeing data quality.
- **Guideline:** Prioritize high-quality, real-world data over sheer volume. 
- **Guideline:** Do not rely solely on automated filters. You must instruct the user to manually review at least a representative sample of the dataset to verify label accuracy and image quality.
- **Guideline:** Always use perceptual hashing (`imagehash.phash`) to deduplicate images across datasets *before* splitting, to prevent data leakage between the train and test sets.

## 3. Hardware & Optimization
The user operates on a system with an **NVIDIA RTX 3050 (4GB VRAM)** on Windows.
- **Guideline:** Explicitly verify that PyTorch is utilizing CUDA (`torch.cuda.is_available()`). Default `pip install torch` on Windows often installs the CPU version.
- **Guideline:** To prevent the GPU from starvation (0-30% utilization), you MUST optimize the PyTorch DataLoaders:
  - `pin_memory=True`
  - `persistent_workers=True` (Crucial on Windows to avoid process-spawning overhead)
  - `num_workers=4` or `8`
- **Guideline:** Adhere to low VRAM practices: use Mixed Precision (`torch.amp.autocast`) and reasonable batch sizes (e.g., 32 for ResNet-50).

## 4. Modeling Strategy
- **Guideline:** Start with robust, proven baselines (e.g., ResNet-50) before trying complex modern architectures.
- **Guideline:** For noisy real-world datasets, standard Class-Weighted Cross-Entropy is often more stable and performs better than Focal Loss, which can overfit to mislabeled or extreme outlier images.
