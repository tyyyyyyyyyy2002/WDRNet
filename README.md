# 🌟 AERNet: All-Weather Enhancement Network  

Adverse weather conditions such as rain, nighttime, fog, and snow significantly degrade visibility in traffic scenes, posing challenges for both human drivers and autonomous vehicles. To address this, we propose **AERNet**, an **unsupervised all-weather enhancement network** that leverages **Weather Type Transfer (WTT)**, **Stability Regularizer (SR)**, and **Clear-Domain Alignment (CDA)**. AERNet enhances degraded images into high-quality, structurally consistent, and visually stable outputs, without requiring paired training data.  

![Architecture diagram](images/gif.gif)  

---
## 📰 News  

- **Sep 18, 2025**: Our paper is under submission to **ICASSP 2026** 📑  
- **Jul 8, 2025**: Codes are released!  🚀
- **Jul 8, 2025**: Homepage is released!💥
---

## 🏗️ Model Architecture  
<div align="center">
  <img src="images/model.png" alt="Architecture diagram" width="800" />
</div>

### ✨ Key Features  

- ** AERNet Framework**  
  An **unsupervised Network** designed for all-weather image enhancement, demonstrating competitive performance across multiple real-world and benchmark datasets.  

- ** Weather Type Transfer (WTT)**  
  Introduces a **data-driven framework** for unsupervised all-weather enhancement, using **dual-level discriminators** to tackle the challenge of limited paired data.  

- ** Stability Regularizer (SR)**  
  Ensures temporal and structural stability across weather transitions, preventing flickering and maintaining detail preservation.  

- ** Clear-Domain Alignment (CDA)**
  A module that disentangles **structural features** from **weather-induced effects**, adaptively modulating perturbation features via weather-label embeddings to achieve accurate, detail-preserving, and robust image enhancements.

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
python train.py --config configs/aernet.yaml

# Testing
python test.py --input ./samples/inputs --output ./samples/results

```
---

## 📚 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{AERNet2025,
  title={AERNet: Unsupervised All-Weather Enhancement via Weather Type Transfer and Clear-Domain Alignment},
  author={Yunyi Tang, Bowei Fang, Chunyu Zhao, Haoran Liu, Fei Yan and Tao Deng},
  year={2025}
}

---
