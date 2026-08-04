import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from src.training.losses import combined_loss


class Trainer:
    """
    Clean training loop for the SSL framework.
    Handles: optimiser, scheduler, logging, checkpointing.
    """

    def __init__(self, model, config, device='cpu'):
        self.model  = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs']
        )

        os.makedirs(config['paths']['results_dir'], exist_ok=True)
        self.history = []

    def train_epoch(self, dataloader):
        self.model.train()

        total_loss = 0.0
        total_c    = 0.0
        total_r    = 0.0
        total_cos  = 0.0

        for batch in dataloader:
            x = batch[0].to(self.device)

            # forward — generates two augmented views
            z1, z2, x1, x2 = self.model(x)

            # combined loss — pass x as the reconstruction target
            loss, c_loss, r_loss = combined_loss(
                z1, z2, self.model, x1, x,   # ← x is the clean original
                temperature=self.config['training']['temperature'],
                recon_weight=self.config['training']['reconstruction_weight']
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_c    += c_loss.item()
            total_r    += r_loss.item()
            total_cos  += F.cosine_similarity(z1, z2).mean().item()

        n = len(dataloader)
        return {
            'loss':     total_loss / n,
            'c_loss':   total_c    / n,
            'r_loss':   total_r    / n,
            'cos_sim':  total_cos  / n,
        }

    def train(self, dataloader):
        epochs = self.config["training"]["epochs"]
        print(f"Training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader)
            self.scheduler.step()
            self.history.append({"epoch": epoch + 1, **metrics})

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Contrastive: {metrics['c_loss']:.4f} | "
                    f"Cos: {metrics['cos_sim']:.4f}"
                )
                # checkpoint cada 10 epochs — protege contra desconexión
                checkpoint_path = os.path.join(
                    self.config["paths"]["results_dir"],
                    f"checkpoint_epoch_{epoch+1}.pt"
                )
                m = self.model.module if hasattr(self.model, "module") else self.model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': m.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                }, checkpoint_path)

        m = self.model.module if hasattr(self.model, "module") else self.model
        m.save(self.config["paths"]["checkpoint"])
        print("Training complete")
        return self.history