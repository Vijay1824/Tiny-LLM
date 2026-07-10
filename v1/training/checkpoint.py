import torch

def save_checkpoint(
    model,
    optimizer,
    epoch,
    filename
):

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        filename
    )


def load_checkpoint(
    model,
    optimizer,
    filename
):

    checkpoint = torch.load(filename)

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    return checkpoint["epoch"]
