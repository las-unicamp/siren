import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.checkpoint import load_checkpoint, save_checkpoint
from src.datasets import DerivativesDataset, DerivativesDatasetBatches
from src.early_stopping import EarlyStopping
from src.hyperparameters import args
from src.model import SIREN
from src.read_data import read_data_from_matfile
from src.running import (
    MixedPrecisionStrategy,
    Runner,
    StandardPrecisionStrategy,
    TrainingConfig,
    TrainingMetrics,
    run_epoch,
)
from src.tensorboard_tracker import TensorboardTracker
from src.timeit_decorator import timeit_decorator


def enforce_reproducibility():
    # Set random seed for CPU and GPU
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # For multi-GPU setups

    # Enable deterministic algorithms for operations, where available
    # torch.use_deterministic_algorithms(True)  # This might impact performance

    # Configure cuDNN for deterministic behavior in GPU training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Avoid auto-tuner for consistent results


@timeit_decorator
def main():
    enforce_reproducibility()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    training_data = read_data_from_matfile(args.input_filename)

    if args.batch_size > 1:
        print("Running with multiple batches")
        dataset = DerivativesDatasetBatches(training_data=training_data, device=device)
    else:
        print("Running with single batch")
        dataset = DerivativesDataset(training_data=training_data, device=device)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    model = SIREN(
        hidden_features=args.num_nodes_per_layer,
        hidden_layers=args.num_hidden_layers,
        first_omega=args.first_layer_omega,
        hidden_omega=args.hidden_layer_omega,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=5000)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1000, factor=0.5, min_lr=1e-7
    )

    if args.use_autocast:
        strategy = MixedPrecisionStrategy()
    else:
        strategy = StandardPrecisionStrategy()

    config: TrainingConfig
    config = {
        "device": device,
        "fit_option": args.fit,
        "optimizer": optimizer,
        "strategy": strategy,
    }

    runner = Runner(dataloader, model, config, TrainingMetrics())

    tracker = TensorboardTracker(
        log_dir=args.logging_root, dirname=args.experiment_name
    )

    print(f"Running on {device}")

    if args.load_checkpoint:
        (epoch_from_previous_run, prev_lr, *_) = load_checkpoint(
            model=model,
            optimizer=optimizer,
            device=device,
            filename=args.load_checkpoint,
        )
        runner.epoch = epoch_from_previous_run

    lowest_err = np.inf

    progress_bar = tqdm(
        total=(args.num_epochs - runner.epoch),
        leave=True,
        desc="Training Progress",
    )

    for epoch in range(runner.epoch, args.num_epochs):
        progress_bar.set_description(f"Epoch {epoch + 1}/{args.num_epochs}")

        epoch_loss, epoch_psnr, epoch_mae = run_epoch(runner=runner, tracker=tracker)

        scheduler.step(epoch_loss)

        current_lr = scheduler.get_last_lr()[0]
        if epoch > 0 and current_lr != prev_lr:
            print(f"Learning rate changed! New learning rate: {current_lr}")
        prev_lr = current_lr

        early_stopping(epoch_loss)
        if early_stopping.stop:
            print("Ealy stopping")
            break

        if tracker.should_save_intermediary_data():
            if is_iteration_to_save_data(runner.epoch):
                tracker.save_epoch_data("data_epoch", runner.epoch)

        # Flush tracker after every epoch for live updates
        tracker.flush()

        if should_save_model(runner.epoch, lowest_err, epoch_loss):
            lowest_err = epoch_loss
            filename = args.checkpoint_file_name_on_save
            save_checkpoint(
                runner.model,
                optimizer,
                runner.epoch,
                current_lr,
                epoch_loss,
                filename,
            )
            print(f"Best psnr: {epoch_psnr} \t \t Best loss: {epoch_loss}")

        progress_bar.update(1)
        progress_bar.set_postfix(
            loss=f"{epoch_loss:.5f}",
            psnr=f"{epoch_psnr:.5f}",
            mae=f"{epoch_mae:.5f}",
        )

    progress_bar.close()


def should_save_model(epoch: int, lowest_err: float, actual_err: float) -> bool:
    if (epoch % args.epochs_until_checkpoint == 0) and (lowest_err > actual_err):
        return True
    return False


def is_iteration_to_save_data(epoch: int):
    return epoch % args.epochs_until_checkpoint == 0


if __name__ == "__main__":
    main()
