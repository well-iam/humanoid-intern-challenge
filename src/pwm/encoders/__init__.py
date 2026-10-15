"""Encoder adapters: frames (T, H, W, 3) uint8 -> frame embeddings (T, patches, dims).

An encoder has a `name` and an `encode(frames)` method. Frames come already cropped to
the square of table the encoder should see; the encoder only resizes them. Heavy encoders
are imported only when asked for.
"""


def _toy():
    from pwm.encoders.toy import ToyEncoder

    return ToyEncoder()


def _dinov2():
    from pwm.encoders.vit import DinoV2Small

    return DinoV2Small()


def _vjepa2():
    from pwm.encoders.vit import VJepa2Large

    return VJepa2Large()


ENCODERS = {"toy": _toy, "dinov2": _dinov2, "vjepa2": _vjepa2}


def get_encoder(name: str):
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder: {name}")
    return ENCODERS[name]()
