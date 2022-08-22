import imp
from time import perf_counter

import pandas as pd
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_encoder

from main import DeepLabV3Plus, MobileOne, MobileOneSMPAdapter
from mobileone import PARAMS, reparameterize_model


def deeplabv3plus_mobileones1():
    mobileone_s1 = MobileOne(**PARAMS["s1"])
    # create the adapter, I was lazy `mobileone_s1` will be modified in-place
    encoder = MobileOneSMPAdapter(mobileone_s1, PARAMS["s1"]["width_multipliers"])
    # have no idea what this thing does
    encoder.make_dilated(output_stride=16)
    # create our deep lab v3 +
    model = DeepLabV3Plus(encoder=encoder, in_channels=3, classes=1000).float().eval()
    # reparameterize
    model.encoder.model = reparameterize_model(model.encoder.model)

    return model


def deeplabv3plus_mobilenetv2():
    return (
        smp.DeepLabV3Plus(
            "mobilenet_v2", encoder_weights=None, in_channels=3, classes=1000
        )
        .float()
        .eval()
    )


models = {
    "deeplabv3plus_mobileones1": deeplabv3plus_mobileones1,
    "deeplabv3plus_mobilenetv2": deeplabv3plus_mobilenetv2,
}


def benchmark(model_name, bach_size: int = 4, device: str = "cuda", n_times: int = 32):
    torch_device = torch.device(device)
    x = torch.randn((bach_size, 3, 512, 512), device=torch_device)

    with torch.no_grad():
        model = models[model_name]().eval().to(torch_device)
        if device == "cuda":
            torch.cuda.synchronize()
        # warmup
        for _ in range(8):
            model(x)

        start = perf_counter()
        for _ in range(n_times):
            model(x)

        return perf_counter() - start


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="deeplabv3plus_mobileones1")
    parser.add_argument("--n_times", type=int, default=32)

    args = parser.parse_args()

    model_name, batch_size, device, n_times = (
        args.model_name,
        args.batch_size,
        args.device,
        args.n_times,
    )
    elapsed = benchmark(model_name, batch_size, device, n_times)

    file_path = Path("./benchmark.csv")

    df = pd.DataFrame.from_dict(
        {
            "model_name": [model_name],
            "batch_size": [batch_size],
            "device": [device],
            "n_times": [n_times],
            "elapsed": [elapsed],
        }
    )

    if file_path.exists():
        old_df = pd.read_csv(file_path)
        df = pd.concat([old_df, df])

    df.to_csv(file_path, index=None)
