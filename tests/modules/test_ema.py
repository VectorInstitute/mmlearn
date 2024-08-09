"""Test functionality of the ExponentialMovingAverage ."""

import pytest
import torch
import torchvision

from mmlearn.modules.ema import ExponentialMovingAverage


torch.manual_seed(42)


@pytest.mark.integration_test()
def test_ema() -> None:
    """Test ExponentialMovingAverage."""
    model = torchvision.models.resnet18()
    ema = ExponentialMovingAverage(
        model=model,
        ema_decay=0.9998,
        ema_end_decay=0.9999,
        ema_anneal_end_step=300000,
    )
    ema.model = ema.model.cpu()  # for testing purposes

    # test output between model and ema model
    model_input = torch.rand(1, 3, 224, 224)
    output = model(model_input)
    output_ema = ema.model(model_input)
    torch.testing.assert_close(output, output_ema)

    # change the model parameters to simulate training
    model_updated = torchvision.models.resnet18()

    # test ema step
    ema.step(model_updated)

    # Validate that the EMA step has updated the model parameters
    ema_state_dict = ema.model.state_dict()
    model_updated_state_dict = model_updated.state_dict()
    for param_name, param in model.state_dict().items():
        ema_param = ema_state_dict[param_name].float()
        model_updated_param = model_updated_state_dict[param_name].float()

        if param_name in ema.skip_keys or not param.requires_grad:
            assert torch.allclose(
                ema_param,
                model_updated_param,
            ), f"Unexpected EMA parameter value for {param_name}"
        elif len(param.shape) > 0:  # check only if it is model parameter
            expected_ema_param = param.mul(ema.decay).add(
                model_updated_param.to(dtype=ema_param.dtype),
                alpha=1 - ema.decay,
            )

            assert torch.allclose(
                ema_param,
                expected_ema_param,
            ), f"Unexpected EMA parameter value for {param_name}"
