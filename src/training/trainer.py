import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.training.losses import combined_loss

class Trainer:
    def __init__(self, model, config, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.config = config
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        self.model = model.to(device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"]
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["training"]["epochs"]
        )
        os.makedirs(config["paths"]["results_dir"], exist_ok=True)
        self.history = []

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = total_c = total_r = total_cos = 0.0
        for batch in dataloader:
            x = batch[0].to(self.device)
            m = self.model.module if hasattr(self.model, "module") else self.model
            z1, z2, x1, x2 = m(x)
            loss, c_loss, r_loss = combined_loss(
                z1, z2, m, x1, x,
                temperature=self.config["training"]["temperature"],
                recon_weight=self.config["training"]["reconstruction_weight"]
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            total_c    += c_loss.item()
            total_r    += r_loss.item()
            total_cos  += F.cosine_similarity(z1, z2).mean().item()
        n = len(dataloader)
        return {"loss": total_loss/n, "c_loss": total_c/n,
                "r_loss": total_r/n, "cos_sim": total_cos/n}

    def train(self, dataloader):
        epochs = self.config["training"]["epochs"]
        print(f"Training for {epochs} epochs on {self.device}...")
        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader)
            self.scheduler.step()
            self.history.append({"epoch": epoch+1, **metrics})
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Contrastive: {metrics['c_loss']:.4f} | "
                      f"Cos: {metrics['cos_sim']:.4f}")
        m = self.model.module if hasattr(self.model, "module") else self.model
        m.save(self.config["paths"]["checkpoint"])
        print("Training complete")
        return self.history
