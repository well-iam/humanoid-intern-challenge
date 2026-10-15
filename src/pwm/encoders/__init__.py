"""Encoder adapters: frames (T, H, W, 3) uint8 -> frame embeddings (T, patches, dims).

An encoder has a `name` and an `encode(frames)` method. Heavy encoders are imported
only when asked for.
"""


def get_encoder(name: str):
    if name == "toy":
        from pwm.encoders.toy import ToyEncoder

        return ToyEncoder()
    raise ValueError(f"unknown encoder: {name}")
