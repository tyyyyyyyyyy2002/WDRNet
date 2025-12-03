import re
import matplotlib.pyplot as plt

# 日志文件路径
log_file = "/data/workspace/tyy/demo/contrastive-unpaired-translation-master/checkpoints/grumpycat_FastCUT2waveweight/loss_log.txt"
save_path = "G_loss_curve.png"  # 保存图片路径

epochs = []
g_loss = []

# 匹配 epoch 和 G loss
pattern = re.compile(r"\(epoch: (\d+), iters: \d+, .*?\) G: ([\d\.]+)")

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            g_loss.append(float(match.group(2)))

# 绘制曲线
plt.figure(figsize=(10,6))
plt.plot(epochs, g_loss, label="G Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Generator Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(save_path)  # 保存图片
plt.show()

print(f"G loss 曲线已保存为 {save_path}")
