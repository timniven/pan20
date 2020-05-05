from torch import autograd, nn


class RevGradFunction(autograd.Function):
    """Gradient reversal function.

    https://github.com/janfreyberg/pytorch-revgrad/blob/master/src/pytorch_revgrad/functional.py
    """

    @staticmethod
    def forward(ctx, inputs, lambda_grad):
        ctx.save_for_backward(inputs)
        ctx.lambda_grad = lambda_grad
        return inputs

    @staticmethod
    def backward(ctx, grad_wrt_output):
        grad_wrt_input = None
        if ctx.needs_input_grad[0]:
            grad_wrt_input = -grad_wrt_output * ctx.lambda_grad
        return grad_wrt_input, None


class GradientReversal(nn.Module):
    """Gradient reversal layer.

    Identity function on forward pass, reverse gradients on backward.

    https://github.com/janfreyberg/pytorch-revgrad/blob/master/src/pytorch_revgrad/module.py
    """

    def __init__(self, lambda_grad):
        """Create a new gradient reversal layer.

        Args:
          lambda_grad: Float, multiplies the gradient on the backward pass. This
            should be a positive number.
        """
        super().__init__()
        self.lambda_grad = lambda_grad

    def forward(self, inputs):
        return RevGradFunction.apply(inputs, self.lambda_grad)
