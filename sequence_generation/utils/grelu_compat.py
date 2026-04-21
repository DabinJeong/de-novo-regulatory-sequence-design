"""
grelu LightningModel checkpoint-format compatibility.

DRAKES distributes enhancer regressors saved by an older grelu that stored
model/train/data params nested inside ``checkpoint['hyper_parameters']``.
Newer grelu releases read several pieces (``data_params``, ``performance``, ...)
as top-level checkpoint keys via ``LightningModel.on_load_checkpoint``, so
``load_from_checkpoint`` dies with ``KeyError: 'data_params'`` (or
``'performance'``) on DRAKES' ``reward_oracle_eval.ckpt`` etc.

``patch_grelu_lightning_compat()`` replaces ``on_load_checkpoint`` with a
forgiving version: any expected top-level key that is missing is filled from
``hyper_parameters`` when possible, otherwise set to an empty default. This
keeps the state_dict load path (handled separately by Lightning) untouched.
Idempotent.
"""


def patch_grelu_lightning_compat() -> None:
    from grelu.lightning import LightningModel

    if getattr(LightningModel, "_grelu_compat_patched", False):
        return

    def patched(self, checkpoint):
        hp = checkpoint.get("hyper_parameters", {}) or {}

        # Fields grelu's on_load_checkpoint has been observed to require but
        # that DRAKES-era checkpoints don't store at the top level.
        fallback_from_hp = ("data_params", "model_params", "train_params")
        for key in fallback_from_hp:
            if key not in checkpoint and isinstance(hp, dict) and key in hp:
                checkpoint[key] = hp[key]

        # `performance` is a training-time metric dict; harmless to stub out.
        empty_defaults = {"performance": {}}
        for key, default in empty_defaults.items():
            checkpoint.setdefault(key, default)

        # Mirror the effect of grelu's native on_load_checkpoint: copy the
        # checkpoint keys onto the module so downstream code can read them.
        for key in list(fallback_from_hp) + list(empty_defaults):
            if key in checkpoint:
                setattr(self, key, checkpoint[key])

    LightningModel.on_load_checkpoint = patched
    LightningModel._grelu_compat_patched = True
