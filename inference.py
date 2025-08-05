import argparse
import os
import numpy as np
import time
import open3d as o3d
import torch
from tqdm.auto import tqdm

from tools import builder
from utils.config import cfg_from_yaml_file
from utils.denoise_misc import *
from utils.denoise import *
from utils.transforms import *
from datasets.io import IO
from datasets.data_transforms import Compose
from models.denoise import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_args():
    parser = argparse.ArgumentParser(description="UniDCF inference.")
    parser.add_argument('--model_config', type = str,
                        default=os.path.join(BASE_DIR, 'cfgs/UniDCF_models/UniDCF.yaml'),
                        help = 'Path to model config file')
    parser.add_argument('--unidcf_n_ckpt', type = str,
                        default=os.path.join(BASE_DIR, 'experiments/UniDCF/UniDCF_models/ckpt1.pth'),
                        help = 'Path to pretrained UniDCF-N checkpoint')
    parser.add_argument('--denoise_ckpt', type = str,
                        default=os.path.join(BASE_DIR, 'experiments/UniDCF/UniDCF_models/ckpt2.pt'),
                        help = 'Path to pretrained denoiser checkpoint')
    parser.add_argument('--pc_root', type=str, 
                        default=os.path.join(BASE_DIR, 'data/pcd/'), 
                        help='Point cloud root directory')
    parser.add_argument('--ima_root', type=str, 
                        default=os.path.join(BASE_DIR, 'data/ima/'), 
                        help='Image root directory')
    parser.add_argument('--out_pc_root', type=str,
                       default=os.path.join(BASE_DIR, 'inference_result/'),
                        help='Directory to save output point cloud')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--visualize', default=True, help='Visualize input & output point clouds in 3D')
    args = parser.parse_args()
    return args


def denoise_single(pcd_data, args):
    seed_all(args.seed)
    # Model
    ckpt = torch.load(args.denoise_ckpt, map_location=args.device)
    model = DenoiseNet(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    
    pcl = torch.FloatTensor(np.array(pcd_data))
    pcl, center, scale = NormalizeUnitSphere.normalize(pcl)
    pcl_denoised = patch_based_denoise(model, pcl.to(args.device)).cpu()
    return pcl_denoised * scale + center


def inference_single(model, pc_path, ima_path, data_name, args, out_path):
    x_img = IO.get(os.path.join(ima_path, data_name + "_x.png")).astype(np.float32)
    y_img = IO.get(os.path.join(ima_path, data_name + "_y.png")).astype(np.float32)
    z_img = IO.get(os.path.join(ima_path, data_name + "_z.png")).astype(np.float32)

    pc_array = IO.get(pc_path).astype(np.float32)
    ima_np = np.stack([x_img, y_img, z_img], axis=-1)
    
    transform = Compose([{
        'callback': 'RandomSamplePoints',
        'parameters': {
            'n_points': 14336
        },
        'objects': ['input']
    }, 
    {'callback': 'TransIma',
                'parameters': {'size': 224},
                'objects': ['partialima']},
    {
        'callback': 'ToTensor',
        'objects': ['input']
    }])

    data = transform({'input': pc_array, 'partialima': ima_np})
    pred = model(data['input'].unsqueeze(0).to(args.device),
                 data['partialima'].unsqueeze(0).to(args.device))
    
    dense_pc = pred[-1].squeeze(0).detach().cpu().numpy()
    denoised_pc = denoise_single(dense_pc, args)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(denoised_pc)
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"[INFO] Saved denoised point cloud to: {out_path}")
    # ====== 新增可视化 ======
    if args.visualize:
        visualize_point_clouds(pc_array, denoised_pc)



def run_inference(args):
    start_time = time.time()
    config = cfg_from_yaml_file(args.model_config)
    model = builder.model_builder(config.model)
    builder.load_model(model, args.unidcf_n_ckpt)
    model.to(args.device)
    model.eval()

    os.makedirs(args.out_pc_root, exist_ok=True)

    if args.pc_root:
        name = os.listdir(args.pc_root)[0].split(".ply")[0]
        inference_single(model, os.path.join(args.pc_root, name + ".ply"), 
                         args.ima_root, name, args, os.path.join(args.out_pc_root, name + ".ply"))
    else:
        files = sorted([f for f in os.listdir(args.pc_root) if f.endswith('.ply')])
        for file in tqdm(files, desc="Inference"):
            name = os.path.splitext(file)[0]
            pc_file = os.path.join(args.pc_root, file)
            out_file = os.path.join(args.out_pc_root, name + ".ply")
            inference_single(model, pc_file, args.ima_root, name, args, out_file)
    end_time = time.time()
    print(f"[INFO] Model inference time is: {end_time - start_time}")


def visualize_point_clouds(pc_input, pc_output):
    """
    显示输入点云和预测点云
    :param pc_input: numpy array, shape (N,3)
    :param pc_output: numpy array, shape (M,3)
    """
    pcd_input = o3d.geometry.PointCloud()
    pcd_input.points = o3d.utility.Vector3dVector(pc_input)
    pcd_input.paint_uniform_color([0, 0, 1])  # 蓝色

    pcd_output = o3d.geometry.PointCloud()
    pcd_output.points = o3d.utility.Vector3dVector(pc_output)
    pcd_output.paint_uniform_color([1, 0, 0])  # 红色

    o3d.visualization.draw_geometries([pcd_input, pcd_output],
                                      window_name="Input vs Predicted Point Clouds",
                                      width=960, height=720)

def main():
    args = get_args()
    run_inference(args)


if __name__ == '__main__':
    main()