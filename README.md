# 🌟 Breaking Weather Degradation: Wavelet-Contrastive WDRNet for Unsupervised All-Weather Understanding 

Adverse weather conditions such as rain, fog, snow, and low-light significantly degrade visibility in traffic scenes, challenging both human drivers and autonomous vehicles. To address this, we propose **WDRNet (Wavelet Deweathering Restoration Network)**, an **unsupervised all-weather image enhancement network** that leverages a **Spatial–Frequency Wavelet Pyramid Discriminator (SPF-WPD)** and a **Wavelet High-frequency Contrastive Consistency Loss (WHCC)**. **WDRNet** effectively removes large-area weather degradations while preserving high-frequency details such as road edges, vehicle boundaries, and building contours, producing **high-quality, structurally consistent, and visually faithful images** without requiring paired training data.

---
## 📰 News  

- **Dec 10, 2025**: Our paper is under submission to **ICME 2026** 📑  
- **Nov 8, 2025**: Codes are released!  🚀
- **Nov 8, 2025**: Homepage is released!💥
---

## 🏗️ Model Architecture  
<div align="center">
  <img src="images/model.png" alt="Architecture diagram" width="800" />
</div>

### ✨ Key Features  

1. **Unsupervised All-Weather Enhancement**: WDRNet is specifically designed for real-world traffic scenes, balancing **low-frequency weather removal** with **high-frequency texture preservation**.

2. **Spatial–Frequency Wavelet Pyramid Discriminator (SPF-WPD)**: This discriminator models large-area weather degradations across **multiple spatial and frequency scales**, improving robustness under diverse adverse conditions.

3. **Wavelet High-frequency Contrastive Consistency Loss (WHCC)**: Combines **multi-level feature contrastive learning** with **wavelet high-frequency consistency** to effectively preserve fine details such as **road edges, vehicle boundaries, and building contours**.
---

## 📊 Experimental Results  
<div align="center">
  <img src="images/compare.png" alt="Architecture diagram" width="1200" />
</div>

AERNet achieves outstanding results on real-world datasets, including **ACDC** and **Dark Zurich nighttime**, demonstrating strong structural preservation, perceptual quality, and robust generalization.

- **Quantitative Comparison**  
  - Outperforms advanced methods such as Santa, Retinexformer, CycleGAN, UNIT, UNSB, SCI, and PIE.  
  - Achieves the highest **SSIM** and **LPIPS**, along with competitive **PSNR**, indicating superior structural preservation, perceptual quality, and fidelity.

- **Qualitative Evaluation**  
  - Produces clearer and more realistic images under extreme weather compared to existing methods.  
  - Effectively removes snow, suppresses reflections in rain, enhances visibility in fog, and brightens nighttime scenes while preserving details, structure, and natural colors.  

---


## 🚀 Training & Testing (to be updated)

###  Dataset  
Supported datasets include:  
- [ACDC](https://acdc.vision.ee.ethz.ch/)  
- [Dark Zurich](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/?utm_source=chatgpt.com)

### Organize the dataset
The datasets should be organized as follows:
```bash
├── ACDC
│   ├── trainA  # Contains adverse weather images
│   └── trainB  # Contains normal weather images
```


###  Usage 
```bash
# Training
python train.py --dataroot ./datasets/weather --name AllWeather --mode WDRNet

# Testing
python test.py --dataroot ./datasets/weather --name  AllWeather --mode WDRNet --phase train

```
---

## 📚 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{WDRNet2025,
  title={WDRNet: Wavelet-Based Deweathering Restoration for Unsupervised All-Weather Image Enhancement},
  year={2025}
}

---
