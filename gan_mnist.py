import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# ハイパーパラメータの設定
EPOCHS = 50
BATCH_SIZE = 64
LATENT_DIM = 100 # ノイズベクトルの次元
LR_G = 0.0002 # Generatorの学習率
LR_D = 0.0002 # Discriminatorの学習率

# 画像の保存先ディレクトリ
SAVE_DIR = 'generated_images'
os.makedirs(SAVE_DIR, exist_ok=True)

# データの準備
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # -1から1に正規化
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Generatorの定義
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 入力: latent_dim次元のノイズ
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 28 * 28), # MNIST画像のサイズ
            nn.Tanh() # 出力を-1から1にクリップ
        )

    def forward(self, input):
        return self.main(input).view(-1, 1, 28, 28) # 1チャンネルの28x28画像にreshape

# Discriminatorの定義
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 入力: 1チャンネルの28x28画像
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid() # 0から1の確率を出力
        )

    def forward(self, input):
        return self.main(input.view(-1, 28 * 28))

# モデルのインスタンス化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

# 損失関数とOptimizer
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer_g = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))

# 学習ループ
print("学習を開始します...")
for epoch in range(EPOCHS):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Discriminatorの学習
        discriminator.zero_grad()

        # 本物画像
        real_labels = torch.ones(batch_size, 1).to(device)
        output_real = discriminator(real_images)
        loss_d_real = criterion(output_real, real_labels)

        # 偽物画像
        noise = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_images = generator(noise)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        output_fake = discriminator(fake_images.detach()) # Generatorの勾配を伝播させない
        loss_d_fake = criterion(output_fake, fake_labels)

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()

        # Generatorの学習
        generator.zero_grad()
        output_g = discriminator(fake_images) # Discriminatorが本物と判断するように学習
        loss_g = criterion(output_g, real_labels) # Generatorは偽物画像を本物と誤認させたいので、本物ラベルと比較

        loss_g.backward()
        optimizer_g.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], D_loss: {loss_d.item():.4f}, G_loss: {loss_g.item():.4f}")

    # エポックごとに画像を生成して保存
    with torch.no_grad():
        fixed_noise = torch.randn(64, LATENT_DIM).to(device) # 一貫したノイズで生成
        generated_samples = generator(fixed_noise).cpu()
        save_image(generated_samples, os.path.join(SAVE_DIR, f'epoch_{epoch+1:03d}.png'), nrow=8, normalize=True)

print("学習が完了しました。")