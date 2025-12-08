import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from torchvision import models
import torch.nn as nn
from torchvision.transforms import Resize, Normalize, Compose
import clip
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as TF  # ✅ 这行是关键
import torch.nn.functional as F
from models.networks import WFD
import pywt
import open_clip
from .networks import PatchSampleF2
import random

#固定核，可微
class DWTLayer(nn.Module):
    # 固定核，可微
    def __init__(self):
        super(DWTLayer, self).__init__()
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])

        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,2,2]
        self.register_buffer("filters", filters)  # 固定核，不更新参数

    def forward(self, x):
        B, C, H, W = x.shape
        # 复制核到每个通道
        filters = self.filters.to(x.device)  # 保证在相同设备
        x = x.view(B * C, 1, H, W)
        y = F.conv2d(x, filters, stride=2, padding=0)
        y = y.view(B, C, 4, y.shape[-2], y.shape[-1])
        LL = y[:, :, 0, :, :]
        LH = y[:, :, 1, :, :]
        HL = y[:, :, 2, :, :]
        HH = y[:, :, 3, :, :]
        return LL, LH, HL, HH





class WDRNet(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--mode', type=str, default="WDRNet")

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_CLIP', type=float, default=1, help='weight for CLIP loss: CLIP)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.mode.lower() == "WDRNet":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=2.0, lambda_CLIP=0.1, flip_equivariance=True, lambda_wavelet=1.0,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def clip_tokenize(self, prompt_list):
        import clip
        return clip.tokenize(prompt_list)

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.loss_names = ['G']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        # 初始化一个 WFD 模块（只保留低频）
        self.wfd = WFD(dim_in=3, dim=3, need=False).to(self.device)  # 假设图像是 RGB

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
            # self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netF2 = PatchSampleF2(
            use_mlp=True,
            nc=256,
            gpu_ids=self.gpu_ids,
            init_type='normal',
            init_gain=0.02
        )

        self.clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32',
                                                                        pretrained="/data/workspace/tyy/demo/contrastive-unpaired-translation-master/open_clip/daclip_ViT-B-32.pt")
        self.clip_model = self.clip_model.to(self.device)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD_P = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            vgg = models.vgg19(pretrained=True).features
            self.vgg_layers = vgg[:16].eval().to(self.device)  # 使用前16层（可调整）
            for param in self.vgg_layers.parameters():
                param.requires_grad = False  # 冻结参数
            self.criterionVGG = nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.patchD = opt.patchD
            self.patchSize = 64
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            # self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()  # 全局判别器损失
        self.loss_D.backward()
        self.optimizer_D.step()

        # 局部判别器优化
        if self.patchD:
            self.set_requires_grad(self.netD_P, True)
            self.optimizer_D_P.zero_grad()
            self.loss_D_P = self.compute_D_patch_loss(self.fake_B.detach(), self.real_B, self.patchSize)
            self.loss_D_P.backward()
            self.optimizer_D_P.step()

        # else:
        #     print("c")

        self.set_requires_grad(self.netD, False)
        if self.patchD:
            self.set_requires_grad(self.netD_P, False)

        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()

        self.loss_G = self.compute_G_loss()  # 包含生成器损失
        self.loss_G.backward()
        self.optimizer_G.step()

        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # 使用冻结的 DA-CLIP 编码图像上下文
        with torch.no_grad():  # 不需要 autocast，避免 half/float 冲突
            # 确保输入为 float32 且在同一设备
            img = self.real_A.to(self.device, dtype=torch.float32)
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            # 如果 clip_model 是 DataParallel/DDP，这里仍然有效
            image_context, degra_context = self.clip_model.encode_image(img, control=True)
            # 显式转为 float32（防止 encode_image 内部有 half precision）
            image_context = image_context.to(dtype=torch.float32)
            degra_context = degra_context.to(dtype=torch.float32)

        # 缓存到对象属性
        self.image_context = image_context
        self.degra_context = degra_context

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        # self.fake = self.netG(self.real, self.image_context)
        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_D_patch_loss(self, fake_B, real_B, patch_size=64):
        """Calculate GAN loss for the patch discriminator"""
        B, C, H, W = fake_B.size()
        h_offset = random.randint(0, max(0, H - patch_size - 1))
        w_offset = random.randint(0, max(0, W - patch_size - 1))
        self.fake_patch = fake_B[:, :, h_offset:h_offset + patch_size, w_offset:w_offset + patch_size]
        real_patch = real_B[:, :, h_offset:h_offset + patch_size, w_offset:w_offset + patch_size]

        # 判别器预测
        pred_fake_P = self.netD_P(self.fake_patch)
        pred_real_P = self.netD_P(real_patch)

        # GAN损失
        loss_fake_P = self.criterionGAN(pred_fake_P, False).mean()
        loss_real_P = self.criterionGAN(pred_real_P, True).mean()
        loss_D_P = 0.5 * (loss_fake_P + loss_real_P)
        return loss_D_P  # ✅ 保留 tensor


    def compute_G_loss(self):
        """Calculate GAN, NCE loss, and high-frequency wavelet loss for the generator"""
        fake = self.fake_B

        # -------------------------
        # 1. GAN loss
        # -------------------------
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            B, C, H, W = fake.size()
            patch_size = getattr(self.opt, 'patch_size', 64)
            h_offset = random.randint(0, max(0, H - patch_size - 1))
            w_offset = random.randint(0, max(0, W - patch_size - 1))
            fake_patch = fake[:, :, h_offset:h_offset + patch_size, w_offset:w_offset + patch_size]
            pred_fake_P = self.netD_P(fake_patch)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            if self.patchD:
                self.loss_G_GAN_P = self.criterionGAN(pred_fake_P, True).mean() * self.opt.lambda_GAN
            else:
                print("辨别器——P损失为0")
                self.loss_G_GAN_P = 0.0
        else:
            self.loss_G_GAN = 0.0

        # -------------------------
        # 2. NCE loss
        # -------------------------
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # -------------------------
        # 3. High-frequency wavelet loss
        # -------------------------
        if getattr(self.opt, 'lambda_wavelet', 0.0) > 0.0:
            def high_freq_l1(x, y):
                # x, y: [B, C, H, W]
                loss = 0.0
                for i in range(x.size(0)):
                    for c in range(x.size(1)):
                        coeffs_x = pywt.dwt2(x[i, c].cpu().detach().numpy(), 'haar')
                        coeffs_y = pywt.dwt2(y[i, c].cpu().detach().numpy(), 'haar')
                        # coeffs = (LL, (LH, HL, HH))
                        LH_x, HL_x, HH_x = coeffs_x[1]
                        LH_y, HL_y, HH_y = coeffs_y[1]
                        loss += (torch.tensor(LH_x - LH_y).abs().mean() +
                                 torch.tensor(HL_x - HL_y).abs().mean() +
                                 torch.tensor(HH_x - HH_y).abs().mean())
                return loss / (x.size(0) * x.size(1))

            self.loss_wavelet = high_freq_l1(self.real_A, self.fake_B) * self.opt.lambda_wavelet
        else:
            self.loss_wavelet = 0.0
            print("a")

        # -------------------------
        # 4. Total generator loss
        # -------------------------
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_P + loss_NCE_both + self.loss_wavelet
        return self.loss_G

    def calculate_layer_contrastive_loss(self, degraded_img, enhanced_img, tau=0.07, num_patches=64):
        # -----------------------
        # 1. 使用 PatchSampleF 采样patch并归一化
        #    注意：不使用多层特征 使用的最终结果图
        # -----------------------
        feat_pos, _ = self.netF2(enhanced_img, num_patches=num_patches)  # 返回 list
        feat_neg, _ = self.netF2(degraded_img, num_patches=num_patches)

        # list -> tensor
        fB = feat_pos[0]  # [N, D]
        fR = feat_neg[0]  # [N, D]
        NB, NR = fB.size(0), fR.size(0)

        # -----------------------
        # 2. 计算相似度矩阵
        # -----------------------
        sim_BB = torch.exp(torch.mm(fB, fB.t()) / tau)  # [NB, NB] 正样本相似度
        sim_RR = torch.exp(torch.mm(fR, fR.t()) / tau)  # [NR, NR] 正样本相似度
        sim_BR = torch.exp(torch.mm(fB, fR.t()) / tau)  # [NB, NR] 负样本相似度
        sim_RB = sim_BR.t()  # [NR, NB]

        # 避免自身与自身的对比
        eye_B = torch.eye(NB, device=fB.device)
        eye_R = torch.eye(NR, device=fR.device)

        # -----------------------
        # 3. 论文公式(5)
        # -----------------------
        term_B = torch.log((sim_BB * (1 - eye_B)).sum(dim=1) /
                           (sim_BR.sum(dim=1) + 1e-8)).mean()
        term_R = torch.log((sim_RR * (1 - eye_R)).sum(dim=1) /
                           (sim_RB.sum(dim=1) + 1e-8)).mean()

        loss = -0.5 * (term_B + term_R)
        return loss

    def calculate_NCE_loss(self, src, tgt): #原来版本
        n_layers = len(self.nce_layers)
        # feat_q = self.netG(tgt, layers=self.nce_layers, encode_only=True, image_context=self.image_context)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

