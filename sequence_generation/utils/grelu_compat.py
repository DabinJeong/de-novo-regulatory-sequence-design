"""
grelu LightningModel checkpoint-format compatibility.

DRAKES distributes enhancer regressors saved by an older grelu that stored
``data_params`` nested inside ``checkpoint['hyper_parameters']``. Newer grelu
releases read it as a top-level ``checkpoint['data_params']`` key via
``LightningModel.on_load_checkpoint``, so ``load_from_checkpoint`` dies with
``KeyError: 'data_params'`` on DRAKES' ``reward_oracle_eval.ckpt`` etc.

``patch_grelu_lightning_compat()`` wraps ``on_load_checkpoint`` so that, when
the top-level key is missing, it is copied from hyper_parameters before the
original method runs. Idempotent.
"""


def patch_grelu_lightning_compat() -> None:
    from grelu.lightning import LightningModel

    if getattr(LightningModel, "_grelu_compat_patched", False):
        return

    original = LightningModel.on_load_checkpoint

    def patched(self, checkpoint):
        if "data_params" not in checkpoint:
            hp = checkpoint.get("hyper_parameters", {})
            if isinstance(hp, dict) and "data_params" in hp:
                checkpoint["data_params"] = hp["data_params"]
        return original(self, checkpoint)

    LightningModel.on_load_checkpoint = patched
    LightningModel._grelu_compat_patched = True
