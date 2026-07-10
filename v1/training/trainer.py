import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


torch.backends.cudnn.benchmark = True


class GPTTrainer:
    """
    TinyMind GPT Trainer (Educational Version)

    Features
    --------
    ✓ CUDA Support
    ✓ CPU Support
    ✓ Windows Compatible
    ✓ Progress Bar
    ✓ Live Loss
    ✓ Learning Rate Display
    ✓ GPU Memory Display
    ✓ Gradient Clipping
    ✓ Cosine LR Scheduler
    ✓ Shape Debugging
    """

    def __init__(
        self,
        model,
        dataset,
        learning_rate=3e-4,
        batch_size=32,
        device=None,
    ):

        # ---------------------------------------------------
        # Device
        # ---------------------------------------------------

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.dataset = dataset

        # ---------------------------------------------------
        # Data Loader
        # ---------------------------------------------------

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if os.name == "nt" else 4,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )

        # ---------------------------------------------------
        # Loss
        # ---------------------------------------------------

        self.loss_fn = nn.CrossEntropyLoss()

        # ---------------------------------------------------
        # Optimizer
        # ---------------------------------------------------

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )

        # ---------------------------------------------------
        # Scheduler
        # ---------------------------------------------------

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
        )

        # ---------------------------------------------------
        # Parameter Count
        # ---------------------------------------------------

        self.total_parameters = sum(
            p.numel()
            for p in self.model.parameters()
        )

        self.trainable_parameters = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

    # =======================================================
    # Print Training Information
    # =======================================================

    def print_training_info(self, epochs):

        print("\n" + "=" * 65)
        print("               TinyMind v1.0 Training")
        print("=" * 65)

        print(f"Device            : {self.device}")

        if torch.cuda.is_available():
            print(f"GPU               : {torch.cuda.get_device_name(0)}")

            total_memory = (
                torch.cuda.get_device_properties(0).total_memory
                / 1024 ** 3
            )

            print(f"GPU Memory        : {total_memory:.2f} GB")

        print(f"PyTorch Version   : {torch.__version__}")

        print(f"Vocabulary Size   : {self.model.config.vocab_size}")

        print(f"Parameters        : {self.total_parameters:,}")

        print(f"Trainable Params  : {self.trainable_parameters:,}")

        print(f"Dataset Samples   : {len(self.dataset):,}")

        print(f"Batch Size        : {self.loader.batch_size}")

        print(f"Epochs            : {epochs}")

        print("=" * 65)

    # =======================================================
    # Train
    # =======================================================

    def train(self, epochs=10):

        self.print_training_info(epochs)

        self.model.train()

        first_batch = True

        total_training_start = time.time()

        for epoch in range(epochs):

            epoch_start = time.time()

            total_loss = 0.0

            progress = tqdm(
                self.loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                colour="green",
                leave=True,
            )

            for batch_idx, (inputs, targets) in enumerate(progress):

                inputs = inputs.to(
                    self.device,
                    non_blocking=True,
                )

                targets = targets.to(
                    self.device,
                    non_blocking=True,
                )

                logits = self.model(inputs)

                if first_batch:

                    print("\nTensor Shapes")
                    print("-" * 40)
                    print(f"Input   : {inputs.shape}")
                    print(f"Target  : {targets.shape}")
                    print(f"Logits  : {logits.shape}")
                    print("-" * 40)

                    first_batch = False

                loss = self.loss_fn(

                    logits.reshape(
                        -1,
                        logits.size(-1),
                    ),

                    targets.reshape(-1),

                )

                self.optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(

                    self.model.parameters(),

                    max_norm=1.0,

                )

                self.optimizer.step()

                total_loss += loss.item()

                gpu_memory = 0

                if torch.cuda.is_available():

                    gpu_memory = (

                        torch.cuda.memory_allocated()

                        / 1024 ** 3

                    )

                progress.set_postfix({

                    "Loss": f"{loss.item():.4f}",

                    "LR": f"{self.optimizer.param_groups[0]['lr']:.6f}",

                    "GPU": f"{gpu_memory:.2f} GB",

                })

            self.scheduler.step()

            avg_loss = total_loss / len(self.loader)

            epoch_time = time.time() - epoch_start

            print("\n" + "=" * 65)

            print(f"Epoch {epoch+1} Completed")

            print(f"Average Loss : {avg_loss:.4f}")

            print(f"Epoch Time   : {epoch_time:.2f} sec")

            print("=" * 65)

        total_time = time.time() - total_training_start

        print("\nTraining Finished Successfully!")

        print(f"Total Time : {total_time:.2f} sec")

        print("=" * 65)