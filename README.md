# UniDCF: A Foundation Model for Comprehensive Dentocraniofacial Hard Tissue Reconstruction

This repository contains the inference code for our paper:

> **UniDCF: A Foundation Model for Comprehensive Dentocraniofacial Hard Tissue Reconstruction**  
> **Authors:** [Chunxia Ren], et al.  

UniDCF is a unified model designed to reconstruct complete dentocraniofacial structures from partial data using deep learning. This repository provides the official implementation for point cloud completion and denoising inference.

---

## 📦 Package Installation

We recommend using a Python virtual environment:

```bash
# Create virtual environment (optional)
python3 -m venv unidcf_env
source unidcf_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
The key packages include:

- `PyTorch >= 1.10`
- `Open3D >= 0.17.0`
- `timm`
- `numpy`, `opencv-python`, `tqdm`, `pyyaml`

> For GPU support, ensure you have the correct CUDA version installed (e.g., CUDA 11.8).

---

### 📁 Download Pretrained Models & Test Data

Due to GitHub's file size limitations, large files such as **pretrained model checkpoints** and **test point cloud data** are not included in this repository.

Please download the required files from the following Baidu NetDisk link:

> 🔗 **[Baidu NetDisk Download (百度网盘)](https://pan.baidu.com/s/1g2JvFd49RjuyLfLDjdU7pg?pwd=ds4z)**  
> 📦 Extraction code: `ds4z` (if required)

Once downloaded, please place the files in the following directories:

```
UniDCF-Inference/
├── data/
│   ├── pcd/           ← Point cloud (.ply) input files
│   └── ima/           ← Corresponding x/y/z normal maps
├── experiments/
│   └── UniDCF/        ← Pretrained checkpoints (.pth / .pt)
```

If you encounter issues or need a mirror link (e.g., Google Drive), feel free to open an issue or contact us.

---

## 🧪 Running Inference

### **1. Prepare Input Data**

Organize your data with the following structure:

```
data/
├── pcd/                 # Point cloud inputs (.ply files) 
	 ├── sample1.ply
└── ima/                 # Associated  images     
	├── sample1_x.png     
	├── sample1_y.png     
	├── sample1_z.png`
```

### **2. Run the Inference Script**

```
python inference.py --pc_root ./data/pcd/ --ima_root ./data/ima/ --visualize
```

#### Optional Arguments:

|Argument|Description|Default|
|---|---|---|
|`--pc`|Path to a single .ply file|_(blank)_|
|`--model_config`|Path to YAML model config|`./cfgs/UniDCF_models/UniDCF.yaml`|
|`--unidcf_n_ckpt`|Path to pretrained UniDCF weights|`./experiments/UniDCF/...`|
|`--denoise_ckpt`|Path to pretrained denoiser checkpoint|`./experiments/UniDCF/...`|
|`--visualize`|Enable 3D visualization (Open3D)|`False`|
|`--out_pc_root`|Output directory for results|`./inference_result/`|

---

## 🖼️ 3D Visualization

If `--visualize` is enabled, the input and predicted point clouds will be displayed in an interactive 3D window:
- **Blue** = Input partial point cloud
- **Red** = Reconstructed + denoised point cloud
---

## 📜 License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this code with proper attribution.

---

## 🙌 Citation

If you find this work useful, please cite our paper:

`@article{waiting}`

---

## 📬 Contact

For questions or collaborations, feel free to contact:
- 📧 renchunxia@bupt.edu.cn
- 🧑 GitHub: [@rcxia](https://github.com/rcxia)
