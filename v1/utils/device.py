import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def print_device_info():

    print("=" * 60)

    print("TinyMind Device Information")

    print("=" * 60)

    print("PyTorch :", torch.__version__)

    if torch.cuda.is_available():

        print("CUDA    : Enabled")

        print("GPU     :", torch.cuda.get_device_name(0))

        print(
            "VRAM    :",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024 ** 3,
                2
            ),
            "GB"
        )

    else:

        print("CUDA    : Disabled")

        print("Running on CPU")

    print("=" * 60)