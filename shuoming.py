# nce是权重10变为1 且数据集用的trainGH
#vgg是加上感知损失，且权重为1 nce权重为5 数据集用的trainGH vgg2数据集用的trainAB(前面全用的源域计算vgg)     vgg3用train且nce权重调为8 vgg权重为5(GH)    vgg4用train且nce权重调为1 vgg权重为1(GH)
#size用的trainAB size变为512
#判别器保证增强后的图像和晴朗域靠近，对比学习保证增强后的图像和增强前的图像的纹理相似
#clip 0GAN 1 clip 10DCE 0 vgg (AB)
#clip2 0GAN 2 clip 10DCE 0 vgg (AB)  引入多提示词，匹配出最像的那个
#clip2plus 0GAN 2 clip 3 DCE 0 vgg (AB)  引入多提示词，匹配出最像的那个
#clip3 0GAN 2 clip 10DCE 0 vgg (AB)  只最大化正向提示词
#clip4 0GAN 1 clip 0。3 DCE 1 vgg (AB) 只利用clip进行图像之间语义信息的对齐
#clip5 0GAN 1 clip 1 DCE 0 vgg (AB) 在4的基础上增加了多尺度
#clip510 0GAN 10clip 1 DCE 0 vgg (AB) 在4的基础上增加了多尺度

#clip510 1GAN  2 DCE 1 vgg (AB)  good
# === Average Metrics per Weather ===
# fog: SSIM=0.4105, PSNR=12.0776, LPIPS=0.5074, NIQE=4.0984
# night: SSIM=0.3495, PSNR=10.1423, LPIPS=0.5254, NIQE=19.2561
# rain: SSIM=0.3690, PSNR=12.8224, LPIPS=0.4342, NIQE=7.9381
# snow: SSIM=0.3298, PSNR=11.9458, LPIPS=0.5145, NIQE=8.6178
#
# Overall Average | SSIM: 0.3647, PSNR: 11.7470, LPIPS: 0.4954, NIQE=9.9776




# python train.py --dataroot /data/workspace/tyy/demo/AllWeatherNet-main/datasets/ACDC --name grumpycat_FastCUTwavekw --CUT_mode FastCUT
#
# python test.py --dataroot /data/workspace/tyy/demo/AllWeatherNet-main/datasets/ACDC  --name grumpycat_FastCUTclip514 --CUT_mode FastCUT --phase train --preprocess resize

import re
import matplotlib.pyplot as plt

def smooth_curve(values, window_size=10):
    """滑动平均平滑曲线"""
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        smoothed.append(sum(values[start:i+1]) / (i - start + 1))
    return smoothed

log_file = '/data/workspace/tyy/demo/contrastive-unpaired-translation-master/checkpoints/grumpycat_FastCUTclipyuyi/loss_log.txt'

iters = []
G_list = []

# 支持提取 epoch、iters、G loss
pattern = re.compile(r'\(epoch: (\d+), iters: (\d+).*?G: ([\d.]+)')

with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            if epoch > 200:
                break  # 只保留前200轮
            iteration = int(match.group(2))
            G_loss = float(match.group(3))
            iters.append((epoch, iteration))
            G_list.append(G_loss)

# 可选平滑
G_smooth = smooth_curve(G_list, window_size=10)

# 生成横坐标为 epoch.iteration，如 1.100, 1.200...
x = [e + i / 10000 for e, i in iters]

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(x, G_smooth, color='blue', label='G loss (smoothed)')
plt.xlabel('Epoch.Iteration')
plt.ylabel('G Loss')
plt.title('Generator Loss Curve (Epochs 1–200)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve2.png")
plt.show()


loadsize 384
=== FID Score: 0.3681 ===

=== Average Metrics per Weather ===
fog: SSIM=0.4182, PSNR=12.9781, LPIPS=0.5140, DISTS=0.2305, NIQE=19.9301
night: SSIM=0.3961, PSNR=10.1832, LPIPS=0.4811, DISTS=0.2897, NIQE=22.8644
rain: SSIM=0.4045, PSNR=13.0805, LPIPS=0.4076, DISTS=0.1981, NIQE=8.5236
snow: SSIM=0.3358, PSNR=11.6485, LPIPS=0.5174, DISTS=0.2357, NIQE=20.1847

=== Overall Average Metrics (Mean of 4 Weathers) ===
SSIM=0.3886, PSNR=11.9726, LPIPS=0.4800, DISTS=0.2385, NIQE=17.8757




#
#
# torch>=1.4.0
# torchvision>=0.5.0
# dominate>=2.4.0
# visdom>=0.1.8.8
# packaging
# GPUtil>=1.4.0


# 513-200
# === Average Metrics per Weather ===
# fog: SSIM=0.4011, PSNR=12.4711, LPIPS=0.5217, NIQE=20.5047
# night: SSIM=0.3680, PSNR=9.1974, LPIPS=0.4658, NIQE=26.1746
# rain: SSIM=0.3875, PSNR=13.0047, LPIPS=0.4049, NIQE=9.7539
# snow: SSIM=0.3347, PSNR=11.7359, LPIPS=0.5102, NIQE=16.9669
#
# Overall Average | SSIM: 0.3728, PSNR: 11.6023, LPIPS: 0.4756, NIQE=18.3500
#
# 514-200 trainGH
# === Average Metrics per Weather ===
# fog: SSIM=0.4368, PSNR=13.2868, LPIPS=0.4912, NIQE=26.8505
# night: SSIM=0.3708, PSNR=9.3545, LPIPS=0.4691, NIQE=23.0362
# rain: SSIM=0.3909, PSNR=12.9785, LPIPS=0.4034, NIQE=9.9508
# snow: SSIM=0.3256, PSNR=11.7929, LPIPS=0.5135, NIQE=20.7956
#
# Overall Average | SSIM: 0.3810, PSNR: 11.8532, LPIPS: 0.4693, NIQE=20.1583

# 514-200 trainAB
# === Average Metrics per Weather ===
# fog: SSIM=0.4718, PSNR=13.4823, LPIPS=0.4843, NIQE=20.6787
# night: SSIM=0.4018, PSNR=9.6568, LPIPS=0.4606, NIQE=27.7973
# rain: SSIM=0.4058, PSNR=13.0728, LPIPS=0.3966, NIQE=5.5993
# snow: SSIM=0.3629, PSNR=12.3211, LPIPS=0.4963, NIQE=10.0813
#
# Overall Average | SSIM: 0.4106, PSNR: 12.1333, LPIPS: 0.4595, NIQE=16.0392

# 515-200 trainAB
# === Average Metrics per Weather ===
# fog: SSIM=0.4694, PSNR=13.5602, LPIPS=0.4881, NIQE=23.7161
# night: SSIM=0.3980, PSNR=9.4029, LPIPS=0.4550, NIQE=26.8665
# rain: SSIM=0.4024, PSNR=13.2093, LPIPS=0.3977, NIQE=9.2285
# snow: SSIM=0.3607, PSNR=12.2810, LPIPS=0.4960, NIQE=12.4884
#
# Overall Average | SSIM: 0.4076, PSNR: 12.1134, LPIPS: 0.4592, NIQE=18.0749

# wavekw-200 trainAB
# === Average Metrics per Weather ===
# fog: SSIM=0.4440, PSNR=13.0125, LPIPS=0.5014, NIQE=16.7723
# night: SSIM=0.3327, PSNR=9.6954, LPIPS=0.5271, NIQE=17.4695
# rain: SSIM=0.3923, PSNR=12.8878, LPIPS=0.4146, NIQE=1.4479
# snow: SSIM=0.3506, PSNR=11.8718, LPIPS=0.5123, NIQE=9.5161
#
# Overall Average | SSIM: 0.3799, PSNR: 11.8669, LPIPS: 0.4889, NIQE=11.3015



# #wavemergeH-200 trainAB
# === Average Metrics per Weather ===
# fog: SSIM=0.4342, PSNR=12.9695, LPIPS=0.4955, NIQE=0.2575
# night: SSIM=0.3319, PSNR=9.6161, LPIPS=0.5360, NIQE=10.5770
# rain: SSIM=0.3864, PSNR=12.9297, LPIPS=0.4060, NIQE=0.1604
# snow: SSIM=0.3408, PSNR=11.7964, LPIPS=0.5111, NIQE=0.6346
#
# Overall Average | SSIM: 0.3733, PSNR: 11.8279, LPIPS: 0.4871, NIQE=2.9074


#wave2-200-14 trainAB
# === Average Metrics per Weather ===
# fog: SSIM=0.4533, PSNR=12.9728, LPIPS=0.4873, NIQE=20.1905
# night: SSIM=0.3756, PSNR=9.2432, LPIPS=0.4590, NIQE=21.0709
# rain: SSIM=0.3951, PSNR=13.0171, LPIPS=0.4009, NIQE=7.8751
# snow: SSIM=0.3405, PSNR=11.9706, LPIPS=0.5050, NIQE=16.2230
#
# Overall Average | SSIM: 0.3911, PSNR: 11.8009, LPIPS: 0.4630, NIQE=16.3399

# wave2-200-15 trainAB 辨别器第一层用小波
# === Average Metrics per Weather ===
# fog: SSIM=0.4715, PSNR=13.4353, LPIPS=0.4862, NIQE=25.0843
# night: SSIM=0.4032, PSNR=9.5698, LPIPS=0.4545, NIQE=22.6385
# rain: SSIM=0.4076, PSNR=13.2950, LPIPS=0.3941, NIQE=11.0772
# snow: SSIM=0.3608, PSNR=12.2319, LPIPS=0.4960, NIQE=13.0942
#
# Overall Average | SSIM: 0.4108, PSNR: 12.1330, LPIPS: 0.4577, NIQE=17.9736

# clipyuyi-对输出直接在clip空间对齐  15
# === Average Metrics per Weather ===
# fog: SSIM=0.4704, PSNR=13.3617, LPIPS=0.4998, NIQE=22.4638
# night: SSIM=0.4009, PSNR=9.4656, LPIPS=0.4574, NIQE=20.3518
# rain: SSIM=0.4060, PSNR=13.0662, LPIPS=0.3976, NIQE=6.4616
# snow: SSIM=0.3597, PSNR=12.0213, LPIPS=0.5015, NIQE=13.4819
#
# Overall Average | SSIM: 0.4093, PSNR: 11.9787, LPIPS: 0.4641, NIQE=15.6897
#
# waveg512-对输出直接在小波变换LH HL HH三个子带的对齐  14
# === Average Metrics per Weather ===
# fog: SSIM=0.4575, PSNR=13.0577, LPIPS=0.4895, NIQE=23.5713
# night: SSIM=0.3708, PSNR=9.1524, LPIPS=0.4618, NIQE=25.4932
# rain: SSIM=0.3984, PSNR=12.9958, LPIPS=0.3993, NIQE=7.9408
# snow: SSIM=0.3489, PSNR=12.1183, LPIPS=0.5019, NIQE=16.1886
#
# Overall Average | SSIM: 0.3939, PSNR: 11.8311, LPIPS: 0.4631, NIQE=18.2985

# waveg512-对输出直接在小波变换LH HL HH三个子带的对齐  15
# === Average Metrics per Weather ===
# fog: SSIM=0.4707, PSNR=13.7151, LPIPS=0.4936, NIQE=26.0304
# night: SSIM=0.4229, PSNR=9.8841, LPIPS=0.4527, NIQE=19.8368
# rain: SSIM=0.4040, PSNR=13.3960, LPIPS=0.3978, NIQE=10.3466
# snow: SSIM=0.3598, PSNR=12.3514, LPIPS=0.4983, NIQE=14.8202
#
# Overall Average | SSIM: 0.4143, PSNR: 12.3367, LPIPS: 0.4606, NIQE=17.7585

# === Average Metrics per Weather ===
# fog: SSIM=0.4736, PSNR=13.4439, LPIPS=0.4812, NIQE=25.3114
# night: SSIM=0.4232, PSNR=10.0348, LPIPS=0.4526, NIQE=25.3498
# rain: SSIM=0.4075, PSNR=13.4712, LPIPS=0.3942, NIQE=10.3619
# snow: SSIM=0.3584, PSNR=12.2001, LPIPS=0.4967, NIQE=14.3304
#
# Overall Average | SSIM: 0.4157, PSNR: 12.2875, LPIPS: 0.4562, NIQE=18.8384

clippara03 Overall Average | SSIM: 0.3920, PSNR: 11.8000, LPIPS: 0.4679, NIQE=24.7964
clippara01        rall Average | SSIM: 0.3820, PSNR: 12.1616, LPIPS: 0.4948, NIQE=28.1279

=== FID Score: 0.2250 ===

=== Average Metrics per Weather ===
fog: SSIM=0.4395, PSNR=13.0910, LPIPS=0.4974, DISTS=0.2148, NIQE=6.1047
night: SSIM=0.3368, PSNR=9.7222, LPIPS=0.5141, DISTS=0.2981, NIQE=21.6302
rain: SSIM=0.4060, PSNR=13.0861, LPIPS=0.3956, DISTS=0.1963, NIQE=0.8461
snow: SSIM=0.3506, PSNR=12.2688, LPIPS=0.4978, DISTS=0.2261, NIQE=2.5497

=== Overall Average Metrics (Mean of 4 Weathers) ===
SSIM=0.3832, PSNR=12.0420, LPIPS=0.4762, DISTS=0.2338, NIQE=7.7827
2wavepatchplus 且为15