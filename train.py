import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import open_clip
if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Dataset and model
    # -------------------------------
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    model = create_model(opt)

    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)
    opt.visualizer = visualizer
    total_iters = 0
    optimize_time = 0.1
    times = []

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()


            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()

            model.set_input(data)
            model.optimize_parameters()
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()





# import time
# import torch
# import clip
# from options.train_options import TrainOptions
# from data import create_dataset
# from models import create_model
# from util.visualizer import Visualizer
# from tqdm import tqdm
# from PIL import Image
# import torch.nn.functional as F
# import os
# import random
# import torchvision.transforms.functional as TF
# # ------------------------------
# # 用于提取数据集平均向量
# # ------------------------------
# def compute_mean_vector(data_path, clip_model, preprocess, clip_device, target_device, target_dtype):
#     embs_list = []
#     # 将 CLIP 设为 eval 已在外面完成
#     for img_name in tqdm(os.listdir(data_path), desc=f"Processing {data_path}"):
#         img_path = os.path.join(data_path, img_name)
#         try:
#             img = Image.open(img_path).convert('RGB')
#         except Exception:
#             # 跳过无法打开的文件
#             continue
#         # preprocess 产生 float32 tensor，放到 clip_device
#         img_tensor = preprocess(img).unsqueeze(0).to(clip_device)  # [1,3,224,224]
#         with torch.no_grad():
#             f = clip_model.encode_image(img_tensor)   # 在 clip_device 上
#             # 在 clip_device 上归一化，避免 CPU 上对 half 的不支持
#             f = F.normalize(f, dim=-1)
#             embs_list.append(f)  # 保持在 clip_device，dtype 与 CLIP 输出一致（通常 float32）
#
#     if len(embs_list) == 0:
#         raise ValueError(f"No valid images in {data_path}")
#
#     # 在 clip_device 上拼接与求均值
#     embs = torch.cat(embs_list, dim=0)   # [N, D]，在 clip_device
#     mean_vec = embs.mean(dim=0)          # [D]，在 clip_device
#     mean_vec = F.normalize(mean_vec, dim=-1)  # 在 clip_device 上归一化
#
#     # 最后把 mean_vec 转到目标 device + dtype（通常 target_dtype=torch.float32）
#     mean_vec = mean_vec.to(device=target_device, dtype=target_dtype)
#
#     return mean_vec
#
# def compute_mean_vector(data_path, clip_model, preprocess,
#                         clip_device, target_device, target_dtype,
#                         crop_size=64, repeat_per_image=1):
#     """
#     计算数据集局部块的 CLIP 平均向量
#     - crop_size: 随机裁剪块大小 (默认 64x64)
#     - repeat_per_image: 每张图片取多少个随机块后取平均（>1 可减少方差）
#     """
#     embs_list = []
#
#     for img_name in tqdm(os.listdir(data_path), desc=f"Processing {data_path}"):
#         img_path = os.path.join(data_path, img_name)
#         try:
#             img = Image.open(img_path).convert('RGB')
#         except Exception:
#             continue
#
#         W, H = img.size
#         if W < crop_size or H < crop_size:
#             # 图片太小则直接 resize 全图到 224
#             for _ in range(repeat_per_image):
#                 img_tensor = preprocess(img).unsqueeze(0).to(clip_device)
#                 with torch.no_grad():
#                     f = clip_model.encode_image(img_tensor)
#                     f = F.normalize(f, dim=-1)
#                 embs_list.append(f)
#             continue
#
#         # 每张图随机取 repeat_per_image 个 64×64 块
#         for _ in range(repeat_per_image):
#             left = random.randint(0, W - crop_size)
#             top  = random.randint(0, H - crop_size)
#             patch = TF.crop(img, top, left, crop_size, crop_size)
#
#             # 依旧用 CLIP 的预处理(内部包含 resize->224 及标准化)
#             patch_tensor = preprocess(patch).unsqueeze(0).to(clip_device)
#
#             with torch.no_grad():
#                 f = clip_model.encode_image(patch_tensor)
#                 f = F.normalize(f, dim=-1)
#             embs_list.append(f)
#
#     if len(embs_list) == 0:
#         raise ValueError(f"No valid images in {data_path}")
#
#     embs = torch.cat(embs_list, dim=0)              # [N, D]
#     mean_vec = embs.mean(dim=0)                     # [D]
#     mean_vec = F.normalize(mean_vec, dim=-1)        # 单位化
#     mean_vec = mean_vec.to(device=target_device, dtype=target_dtype)
#     return mean_vec
#
#
# # ------------------------------
# # 辅助：把 CUTModel 内部子网络设为 float 并移动到目标 device
# # ------------------------------
# def force_model_subnets_float_and_device(model, target_device):
#     # 常见子网名：netG, netD, netF, netE 等。根据你的 CUTModel 实际属性做扩展。
#     subnet_names = ['netG', 'netD', 'netF', 'netE', 'netG_A', 'netG_B']  # 额外列举可能名称
#     for name in subnet_names:
#         if hasattr(model, name):
#             net = getattr(model, name)
#             if net is not None:
#                 try:
#                     net = net.float()                  # 确保 float32
#                     net = net.to(target_device)        # 移动到目标 device
#                     setattr(model, name, net)
#                 except Exception:
#                     # 某些 wrapper 可能不支持直接替换，这里尝试对 module 内部所有参数转换
#                     for p in net.parameters():
#                         p.data = p.data.float().to(target_device)
#                         if p.grad is not None:
#                             p.grad.data = p.grad.data.float().to(target_device)
#
#
# # ------------------------------
# # 主训练脚本
# # ------------------------------
# if __name__ == '__main__':
#     opt = TrainOptions().parse()   # get training options
#     dataset = create_dataset(opt)  # create dataset
#     dataset_size = len(dataset)
#     print('The number of training images = %d' % dataset_size)
#
#     # 创建模型（注意：此刻 model 可能还未 setup / parallelize）
#     model = create_model(opt)
#
#     # 创建可视化工具
#     visualizer = Visualizer(opt)
#     opt.visualizer = visualizer
#     total_iters = 0
#     optimize_time = 0.1
#
#     # ------------------------------
#     # 加载 CLIP，用于计算平均向量
#     # ------------------------------
#     clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     clip_model, preprocess = clip.load("ViT-B/32", device=clip_device)
#     clip_model.eval()
#     for p in clip_model.parameters():
#         p.requires_grad = False
#
#     # ------------------------------
#     # 计算晴朗域和恶劣天气域平均向量
#     # ------------------------------
#     print("Computing mean vectors for domains...")
#     dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # 恶劣天气域
#     dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # 晴朗天气域
#
#     # 获取 CUTModel 的主网络 device（优先用 netG）
#     if hasattr(model, 'netG') and model.netG is not None:
#         # 若 netG 尚未被 .to(...)，其 parameters 依然有 device（可能 cpu），我们取其 device
#         try:
#             model_device = next(model.netG.parameters()).device
#         except StopIteration:
#             # 如果 netG 没有参数（不太可能），退回到默认 GPU/CPU
#             model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         # 如果没有 netG，使用全局设备（尽量用 cuda）
#         model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 强制使用 float32（避免 half 引发的问题）
#     target_dtype = torch.float32
#
#     # 确保 CLIP 在 clip_device，mean 向量最终放到 model_device 且为 float32
#     mean_B = compute_mean_vector(
#         dir_B, clip_model, preprocess,
#         clip_device,
#         target_device=model_device,
#         target_dtype=target_dtype
#     )
#     mean_A = compute_mean_vector(
#         dir_A, clip_model, preprocess,
#         clip_device,
#         target_device=model_device,
#         target_dtype=target_dtype
#     )
#
#     # 强制把 model 内部子网络转换为 float 并移动到 model_device
#     force_model_subnets_float_and_device(model, model_device)
#
#     # 把 mean 向量赋给 model（放在 model 被移动到 device 之后）
#     model.mean_B = mean_B
#     model.mean_A = mean_A
#
#     print("Mean vectors computed and aligned to model device/dtype.")
#
#     # ------------------------------
#     # 训练循环
#     # ------------------------------
#     for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
#         epoch_start_time = time.time()
#         iter_data_time = time.time()
#         epoch_iter = 0
#         visualizer.reset()
#
#         # 如果 dataset 支持 set_epoch（某些 DistributedSampler 需要）
#         if hasattr(dataset, 'set_epoch'):
#             dataset.set_epoch(epoch)
#
#         for i, data in enumerate(dataset):
#             iter_start_time = time.time()
#             if total_iters % opt.print_freq == 0:
#                 t_data = iter_start_time - iter_data_time
#
#             # 确保输入 data 的张量在正确 device 且为 float32（model.set_input 通常会执行 .to(device)）
#             # 这里不主动转换为 half，全部用 float32
#             batch_size = data["A"].size(0)
#             total_iters += batch_size
#             epoch_iter += batch_size
#             if len(opt.gpu_ids) > 0:
#                 torch.cuda.synchronize()
#             optimize_start_time = time.time()
#             if epoch == opt.epoch_count and i == 0:
#                 # 这里在第一次 iter 时调用 data_dependent_initialize
#                 # 先确保 model 的子网为 float 并在正确 device
#                 force_model_subnets_float_and_device(model, model_device)
#                 model.data_dependent_initialize(data)
#                 model.setup(opt)
#                 model.parallelize()
#             model.set_input(data)
#             model.optimize_parameters()
#             if len(opt.gpu_ids) > 0:
#                 torch.cuda.synchronize()
#             optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time
#
#             if total_iters % opt.display_freq == 0:
#                 save_result = total_iters % opt.update_html_freq == 0
#                 model.compute_visuals()
#                 visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
#
#             if total_iters % opt.print_freq == 0:
#                 losses = model.get_current_losses()
#                 visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
#                 if opt.display_id is None or opt.display_id > 0:
#                     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
#
#             if total_iters % opt.save_latest_freq == 0:
#                 print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
#                 save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
#                 model.save_networks(save_suffix)
#
#             iter_data_time = time.time()
#
#         if epoch % opt.save_epoch_freq == 0:
#             print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
#             model.save_networks('latest')
#             model.save_networks(epoch)
#
#         print('End of epoch %d / %d \t Time Taken: %d sec' %
#               (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
#         model.update_learning_rate()

