from app.backend import prepare_backend


def test_wan_backend_imports():
    prepare_backend()
    from engine.animate.config import AnimateConfig

    config = AnimateConfig.animate_14b()
    assert config.num_layers == 40
    assert config.vae_stride == (4, 8, 8)
