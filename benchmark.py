import imp
from pathlib import Path
from time import perf_counter

import pandas as pd
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_encoder
from torchinfo import summary

from main import DeepLabV3Plus, MobileOne, MobileOneSMPAdapter
from mobileone import PARAMS, reparameterize_model
from tqdm import tqdm

def deeplabv3plus_mobileones1():
    mobileone_s1 = MobileOne(**PARAMS["s1"])
    # create the adapter, I was lazy `mobileone_s1` will be modified in-place
    encoder = MobileOneSMPAdapter(mobileone_s1, PARAMS["s1"]["width_multipliers"])
    # have no idea what this thing does
    encoder.make_dilated(output_stride=16)
    # create our deep lab v3 +
    model = DeepLabV3Plus(encoder=encoder, in_channels=3, classes=100).float().eval()
    # reparameterize
    model.encoder.model = reparameterize_model(model.encoder.model)

    return model


def deeplabv3plus_mobilenetv2():
    return (
        smp.DeepLabV3Plus(
            "mobilenet_v2", encoder_weights=None, in_channels=3, classes=100
        )
        .float()
        .eval()
    )


def mobileones1():
    mobileone_s1 = MobileOne(**PARAMS["s1"])
    return reparameterize_model(mobileone_s1)


def mobilenetv2():
    return get_encoder("mobilenet_v2")


models = {
    "deeplabv3plus_mobileones1": deeplabv3plus_mobileones1,
    "deeplabv3plus_mobilenetv2": deeplabv3plus_mobilenetv2,
    "mobilenetv2": mobilenetv2,
    "mobileones1": mobileones1,
}


def benchmark(model, bach_size: int = 4, device: str = "cuda", n_times: int = 8):
    torch_device = torch.device(device)
    x = torch.randn((bach_size, 3, 512, 512), device=torch_device)

    with torch.no_grad():
        model = model.eval().to(torch_device)
        if device == "cuda":
            torch.cuda.synchronize()
        # warmup
        for _ in range(4):
            model(x)
        times = []
        for _ in range(n_times):
            print('.', end='')
            start = perf_counter()
            model(x)
            times.append(perf_counter() - start)
        times_t = torch.as_tensor(times)

        return times_t.mean().item(), times_t.std().item()


def export_to_onnx(model, onnx_filename):
    x = torch.randn((1, 3, 512, 512))
    scripted_model = torch.jit.trace(model,  x)
    torch.onnx.export(
        scripted_model,
        x,
        onnx_filename,
        do_constant_folding=True,
        # opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def benchmark_onnx(
    onnx_filename, bach_size: int = 4, device: str = "cuda", n_times: int = 8
):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    # sess_options.enable_profiling = True
    # ort.set_default_logger_severity(0)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]

    session = ort.InferenceSession(
        onnx_filename, sess_options=sess_options, providers=providers
    )
    torch.cuda.synchronize()
    x = torch.randn((bach_size, 3, 512, 512)).numpy()
    # see https://onnxruntime.ai/docs/api/python/api_summary.html#data-on-device
    x = ort.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
    output = ort.OrtValue.ortvalue_from_shape_and_type([bach_size, 100, 512, 512], x.dtype, 'cuda', 0)  
    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input('input', x)
    io_binding.bind_ortvalue_output('output', output)
    # warmup
    for _ in range(4):
       session.run_with_iobinding(io_binding)
    times = []
    for _ in range(n_times):
        print('.', end='')
        start = perf_counter()
        session.run_with_iobinding(io_binding)
        times.append(perf_counter() - start)
    times_t = torch.as_tensor(times)

    return times_t.mean().item(), times_t.std().item()


def export_and_benchmark(model, model_name, *args, **kwargs):
    onnx_filename = f"{model_name}.onnx"
    if not Path(f"./{onnx_filename}").exists():
        print("[INFO] exporting onnx model ...")
        export_to_onnx(model, onnx_filename)
        print("[INFO] done!")

    return benchmark_onnx(onnx_filename, *args, **kwargs)


runtimes = {"torch": benchmark, "onnx": export_and_benchmark}

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="deeplabv3plus_mobileones1")
    parser.add_argument("--n_times", type=int, default=8)
    parser.add_argument("--runtime", type=str, default="torch")

    args = parser.parse_args()

    model_name, batch_size, device, n_times, runtime = (
        args.model_name,
        args.batch_size,
        args.device,
        args.n_times,
        args.runtime,
    )

    model = models[model_name]()
    summary(model)

    if runtime == "onnx":
        mean, std = export_and_benchmark(model, model_name, batch_size, device, n_times)
    else:
        mean, std = benchmark(model, batch_size, device, n_times)

    file_path = Path("./benchmark.csv")

    df = pd.DataFrame.from_dict(
        {
            "model_name": [model_name],
            "batch_size": [batch_size],
            "device": [device],
            "n_times": [n_times],
            "runtime": [runtime],
            "mean": [mean],
            "std": [std],
        }
    )

    print(df)

    if file_path.exists():
        old_df = pd.read_csv(file_path)
        df = pd.concat([old_df, df])

    print(df)
    df.to_csv(file_path, index=None)
