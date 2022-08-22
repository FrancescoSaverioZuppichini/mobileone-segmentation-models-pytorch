from typing import List, Optional

from segmentation_models_pytorch.base import (ClassificationHead,
                                              SegmentationHead,
                                              SegmentationModel)
from segmentation_models_pytorch.decoders.deeplabv3.decoder import \
    DeepLabV3PlusDecoder
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torch import nn

from mobileone import MobileOne


class MobileOneSMPAdapter(nn.Module, EncoderMixin):
    def __init__(self, model: MobileOne, width_multipliers):
        super().__init__()
        self.model = model
        self._depth = 5
        self._out_channels = [
            3,
            *[int(c * width_multipliers[i]) for i, c in enumerate([64, 128, 256, 512])],
        ]
        self._in_channels = 3

        del model.gap
        del model.linear

    def get_stages(self):
        return [
            nn.Identity(),
            self.model.stage0,
            self.model.stage1,
            self.model.stage2,
            self.model.stage3,
            self.model.stage4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("linear.bias", None)
        state_dict.pop("linear.weight", None)
        super().load_state_dict(state_dict, **kwargs)


# smp sucks


class DeepLabV3Plus(SegmentationModel):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_depth: int = 5,
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "Encoder output stride should be 8 or 16, got {}".format(
                    encoder_output_stride
                )
            )

        self.encoder = encoder

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None


if __name__ == "__main__":
    # get weights from here, unfused version
    # https://github.com/apple/ml-mobileone
    import torch

    from mobileone import PARAMS, reparameterize_model

    mobileone_s1 = MobileOne(**PARAMS["s1"])
    # create the adapter, I was lazy `mobileone_s1` will be modified in-place
    encoder = MobileOneSMPAdapter(mobileone_s1, PARAMS["s1"]["width_multipliers"])
    # have no idea what this thing does
    encoder.make_dilated(output_stride=16)
    # create our deep lab v3 +
    model = DeepLabV3Plus(encoder=encoder, in_channels=3, classes=10).float().eval()
    # train model, then
    reparameterize_model(model.encoder.model)

    with torch.no_grad():
        x = torch.randn((1, 3, 224, 224))
        print(model(x).shape)
