"""Train the multi-satellite fusion network with multi-target heads."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.dataloader import create_exported_data_loader
from model.transformer import SatelliteMultiLSTMWithAttention
from model.utils import project_root, set_seed

ROOT = project_root()


class SatelliteSignalTrainer:
    """Trainer for the multi-satellite fusion network."""

    def __init__(self, config: Dict):
        if "seed" in config:
            set_seed(config["seed"])

        self.config = config
        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.model = self._create_model()
        self.model.to(self.device)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss()

        self.training_log = {
            "start_time": None,
            "end_time": None,
            "total_epochs": self.config["epochs"],
            "config": self.config,
            "epochs": [],
            "best_epoch": None,
            "final_metrics": {},
        }

        self.best_val_accuracy = 0.0
        self.best_val_loss = float("inf")

        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def _create_model(self):
        return SatelliteMultiLSTMWithAttention(
            input_dim=self.config["input_dim"],
            sequence_length=self.config["signal_length"],
            lstm_hidden_dim=self.config["hidden_dim"],
            attention_dim=self.config["attention_dim"],
            num_classes=self.config["num_classes"],
            use_position_encoding=self.config.get("use_position_encoding", True),
            attention_layers=self.config.get("attention_layers", 1),
        )

    def _create_optimizer(self) -> optim.Optimizer:
        """Build the Adam optimizer."""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

    def _create_scheduler(self):
        """Build the cosine learning-rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["epochs"],
            eta_min=self.config["learning_rate"] * 0.01,
        )

    def train_epoch(self, train_loader) -> Tuple[float, float, float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions_task1 = 0
        correct_predictions_task2 = 0
        total_predictions = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                satellite_data = batch["satellite_signals"].to(self.device)
                azimuths = batch["azimuths"].to(self.device)
                elevations = batch["elevations"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(satellite_data, azimuths, elevations, attention_mask)

                # CE(target1 activity) + CE(target2 activity) for multi-target sensing
                loss1 = self.criterion(output["predictions1"], labels[:, 0])
                loss2 = self.criterion(output["predictions2"], labels[:, 1])
                loss = loss1 + loss2

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected at batch {batch_idx}: {loss.item()}")
                    continue

                loss.backward()

                if self.config["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"])

                self.optimizer.step()

                _, predicted_task1 = torch.max(output["predictions1"], 1)
                _, predicted_task2 = torch.max(output["predictions2"], 1)
                correct_task1 = (predicted_task1 == labels[:, 0]).sum().item()
                correct_task2 = (predicted_task2 == labels[:, 1]).sum().item()

                total_loss += loss.item()
                correct_predictions_task1 += correct_task1
                correct_predictions_task2 += correct_task2
                total_predictions += labels.size(0)

                if batch_idx % self.config["log_interval"] == 0:
                    batch_acc_task1 = correct_task1 / labels.size(0)
                    batch_acc_task2 = correct_task2 / labels.size(0)
                    print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
                    print(f"  Target1 Acc: {batch_acc_task1:.4f}, Target2 Acc: {batch_acc_task2:.4f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Batch keys: {batch.keys()}")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"{key}: {value.shape}")
                raise e

        avg_loss = total_loss / len(train_loader)
        accuracy_task1 = correct_predictions_task1 / total_predictions
        accuracy_task2 = correct_predictions_task2 / total_predictions
        avg_accuracy = (accuracy_task1 + accuracy_task2) / 2

        return avg_loss, avg_accuracy, accuracy_task1, accuracy_task2

    def validate(self, val_loader) -> Tuple[float, float, float, float]:
        """Validate on the held-out split."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions_task1 = 0
        correct_predictions_task2 = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                satellite_data = batch["satellite_signals"].to(self.device)
                labels = batch["labels"].to(self.device)
                azimuths = batch["azimuths"].to(self.device)
                elevations = batch["elevations"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                output = self.model(satellite_data, azimuths, elevations, attention_mask)

                loss1 = self.criterion(output["predictions1"], labels[:, 0])
                loss2 = self.criterion(output["predictions2"], labels[:, 1])
                loss = loss1 + loss2

                _, predicted_task1 = torch.max(output["predictions1"], 1)
                _, predicted_task2 = torch.max(output["predictions2"], 1)

                correct_task1 = (predicted_task1 == labels[:, 0]).sum().item()
                correct_task2 = (predicted_task2 == labels[:, 1]).sum().item()

                total_loss += loss.item()
                correct_predictions_task1 += correct_task1
                correct_predictions_task2 += correct_task2
                total_predictions += labels.size(0)
        if len(val_loader) == 0:
            return 0, 0, 0, 0
        avg_loss = total_loss / len(val_loader)
        accuracy_task1 = correct_predictions_task1 / total_predictions
        accuracy_task2 = correct_predictions_task2 / total_predictions
        avg_accuracy = (accuracy_task1 + accuracy_task2) / 2

        return avg_loss, avg_accuracy, accuracy_task1, accuracy_task2

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Persist a training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "training_log": self.training_log,
            "timestamp": datetime.now().isoformat(),
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation accuracy: {self.best_val_accuracy:.4f}")

        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_accuracy = checkpoint["best_val_accuracy"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if "training_log" in checkpoint:
            self.training_log = checkpoint["training_log"]

        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint["epoch"]

    def _save_training_log(self):
        """Write the training log as JSON."""
        log_path = self.output_dir / "training_log.json"
        serializable_log = self._make_serializable(self.training_log)
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(serializable_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save training log: {e}")

    def _make_serializable(self, obj):
        """Convert nested objects to JSON-safe values."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return str(obj)
        return obj

    def train(self, train_loader, val_loader, resume_from: str = None):
        """Run the full training loop."""
        start_epoch = 0

        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {start_epoch}")
        else:
            self.training_log["start_time"] = datetime.now().isoformat()

        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        for epoch in range(start_epoch, self.config["epochs"]):
            epoch_start_time = time.time()
            epoch_start_datetime = datetime.now()

            train_loss, train_accuracy, train_acc_task1, train_acc_task2 = self.train_epoch(train_loader)
            val_loss, val_accuracy, val_acc_task1, val_acc_task2 = self.validate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()

            is_best = val_accuracy > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_accuracy
                self.best_val_loss = val_loss
                self.training_log["best_epoch"] = epoch + 1
                self.save_checkpoint(epoch + 1, is_best)

            epoch_time = time.time() - epoch_start_time

            epoch_log = {
                "epoch": epoch + 1,
                "start_time": epoch_start_datetime.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": epoch_time,
                "metrics": {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "train_task1_accuracy": train_acc_task1,
                    "train_task2_accuracy": train_acc_task2,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_task1_accuracy": val_acc_task1,
                    "val_task2_accuracy": val_acc_task2,
                    "learning_rate": current_lr,
                },
                "is_best": is_best,
                "best_val_accuracy_so_far": self.best_val_accuracy,
                "improvement": val_accuracy - self.best_val_accuracy if not is_best else 0.0,
            }

            self.training_log["epochs"].append(epoch_log)

            if (epoch + 1) % self.config["save_interval"] == 0:
                self.save_checkpoint(epoch + 1, is_best)

            print(f"Epoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.4f}")
            print(f"    Target1 Acc: {train_acc_task1:.4f}, Target2 Acc: {train_acc_task2:.4f}")
            print(f"  Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.4f}")
            print(f"    Target1 Acc: {val_acc_task1:.4f}, Target2 Acc: {val_acc_task2:.4f}")
            print(f"  Time: {epoch_time:.2f}s, LR: {current_lr:.6f}")
            if is_best:
                print("  *** New best model! ***")
            print("-" * 60)

            self._save_training_log()

        self.training_log["end_time"] = datetime.now().isoformat()
        total_training_time = time.time() - time.mktime(
            time.strptime(self.training_log["start_time"][:19], "%Y-%m-%dT%H:%M:%S")
        )

        self.training_log["final_metrics"] = {
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.training_log["best_epoch"],
            "total_training_time_seconds": total_training_time,
            "total_training_time_hours": total_training_time / 3600,
            "final_train_loss": train_loss,
            "final_train_accuracy": train_accuracy,
            "final_val_loss": val_loss,
            "final_val_accuracy": val_accuracy,
        }

        self.save_checkpoint(self.config["epochs"], val_accuracy >= self.best_val_accuracy)
        self._save_training_log()

        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"Total training time: {total_training_time/3600:.2f} hours")
        print(f"Results saved to: {self.output_dir}")


def main():
    config_path = ROOT / "model" / "config" / "gnss_transformer.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    set_seed(config["seed"])

    dataset_dir = ROOT / config.get("dataset_dir", "dataset/GNSS")
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    print("Creating data loaders...")
    print(f"  Train split: {train_dir}")
    print(f"  Validation split: {test_dir}")
    train_loader = create_exported_data_loader(
        split_dir=str(train_dir),
        batch_size=config["batch_size"],
        max_length=config["signal_length"],
        normalize_amplitude=config.get("normalize_amplitude", False),
        normalize_phase=config.get("normalize_phase", False),
        normalization_eps=config.get("normalization_eps", 1e-6),
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config.get("pin_memory", True),
    )

    val_loader = create_exported_data_loader(
        split_dir=str(test_dir),
        batch_size=config["batch_size"],
        max_length=config["signal_length"],
        normalize_amplitude=config.get("normalize_amplitude", False),
        normalize_phase=config.get("normalize_phase", False),
        normalization_eps=config.get("normalization_eps", 1e-6),
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config.get("pin_memory", True),
    )

    print("\nValidating data loaders...")
    try:
        sample_batch = next(iter(train_loader))
        print("Sample batch keys:", list(sample_batch.keys()))
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                if key == "labels":
                    print(f"  Labels range: [{value.min().item()}, {value.max().item()}]")
                    print(f"  Unique labels: {torch.unique(value).tolist()}")

        unique_labels = torch.unique(sample_batch["labels"])
        max_label = unique_labels.max().item()
        if max_label >= config["num_classes"]:
            print(f"WARNING: Max label ({max_label}) >= num_classes ({config['num_classes']})")
            print(f"You may need to adjust num_classes to {max_label + 1}")

    except Exception as e:
        print(f"Error loading sample batch: {e}")
        return

    config["output_dir"] = str(ROOT / config.get("output_dir", "outputs/gnss_transformer"))
    trainer = SatelliteSignalTrainer(config)

    trainer.train(train_loader, val_loader, resume_from=None)


if __name__ == "__main__":
    main()
