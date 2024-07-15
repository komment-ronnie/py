from itertools import islice
import math
from pathlib import Path
from typing import *
import einops
import torch
from torch import nn, Tensor
import tqdm
try:
    import poptorch

    poptorch_available = True
except ModuleNotFoundError:
    poptorch_available = False

class Config(dict):
    """
    Is a simple implementation of a dictionary-like object with additional methods
    for initialization and attribute access.

    Attributes:
        __dict__ (dict): Used to store the instance's attributes, which are added
            during initialization using the `super()` method.

    """
    def __init__(self, *args: Any, **kwargs: Any):
        """
        Initializes instances by calling the parent class's `__init__` and setting
        attributes to their default values before overwriting them with user-provided
        arguments.

        Args:
            *args (Any): List of positional arguments
            **kwargs (Any): Dictionary of keyword arguments

        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self

CONFIG = Config(
    sequence_length=256,
    batch_size=16,
    hidden_size=256,
    head_size=64,
    depth=4,
    fully_scaled_attention=False,
    lr=2**-10,
    steps=5000,
)


# https://www.gutenberg.org/cache/epub/100/pg100.txt
DATA = torch.tensor(list(Path("shakespeare.txt").read_bytes()))


def batches() -> Iterable[Tensor]:
    """
    Iterates over a dataset `DATA` and generates batches of fixed size by randomly
    selecting indices within a range, ensuring each batch contains at least one
    element from every sequence in the dataset.

    Yields:
        Tensor: An iterable containing multiple tensors, where each tensor represents
        a batch of data from the original dataset.

    """
    while True:
        offsets = torch.randint(
            len(DATA) - CONFIG.sequence_length - 1, (CONFIG.batch_size,)
        )
        yield torch.stack([DATA[i : i + CONFIG.sequence_length + 1] for i in offsets])


class Attention(nn.Module):


    """
    Defines a self-attention mechanism that computes attention weights and projects
    output to the final space. It takes a input tensor, applies linear transformations
    to compute q, k, v, pre-attention, and attention weights, and then projects
    the output to the final space using another linear transformation.

    Attributes:
        head_size (float|int): Used to determine the size of each attention head.
            It controls the number of linear layers within each attention head,
            which in turn affects the computational complexity of self-attention.
        n_heads (int): 7.
        qkv (nnLinear): 3D tensor with dimensions (hidden size x number of heads
            x head size). It computes the query, key, and value vectors used in
            the attention mechanism.
        proj (nnLinear): Used to transform the output of the attention mechanism
            into the original input space.
        out_scale (float): 1.0 for fully scaled attention or a scaling factor
            computed as (sequence length / math.e)**0.5 otherwise.

    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Performs self-attention on input tensor `x` by rearranging its dimensions,
        scaling the query and key vectors, computing the attention weights, and
        applying softmax to obtain a weighted sum of the value vector. The output
        is then passed through a projection layer to transform it back to the
        original shape.

        Args:
            x (Tensor): Passed through an einops.rearrange operation to manipulate
                its shape before being processed by the QKV layer.

        Returns:
            Tensor: A transformed version of the input tensor `x`.

        """
        s = x.shape[1]
        q, k, v = einops.rearrange(
            self.qkv(x), "b s (M n d) -> M b n s d", M=3, n=self.n_heads
        )
        qk_scale = torch.tensor(self.head_size**-0.5, dtype=x.dtype, device=x.device)
        pre_a = torch.einsum("bnsd, bntd -> bnst", q, k) * qk_scale
        pre_a = pre_a + torch.triu(
            torch.full((s, s), -1e4, device=x.device), diagonal=1
        )
        a = torch.softmax(pre_a, -1)
        out = torch.einsum("bnst, bntd -> bnsd", a, v) * self.out_scale
        return self.proj(einops.rearrange(out, "b n s d -> b s (n d)"))

    def __init__(self):
        """
        Sets up the necessary components for attention mechanism: head size, number
        of heads, linear layers for projection and scaling.

        """
        super().__init__()
        self.head_size = CONFIG.head_size
        self.n_heads = CONFIG.hidden_size // CONFIG.head_size
        self.qkv = nn.Linear(CONFIG.hidden_size, 3 * self.n_heads * self.head_size)
        self.proj = nn.Linear(self.n_heads * self.head_size, CONFIG.hidden_size)
        # Put the scale in a non-trainable parameter, to avoid recompilation
        self.out_scale = nn.Parameter(
            torch.tensor(
                (CONFIG.sequence_length / math.e) ** 0.5
                if CONFIG.fully_scaled_attention
                else 1.0
            ),
            requires_grad=False,
        )
class FFN(nn.Module):
    """
    Is a neural network layer that consists of two sub-layers: an upsampling (or
    "up") layer and a downsampling layer. The up layer takes the input and maps
    it to a higher dimensional space using a linear transformation, while the down
    layer takes the output of the up layer and maps it back to the original space
    using another linear transformation.

    Attributes:
        up (nnLinear): Responsible for mapping the input to a higher dimensional
            space before passing it through another linear layer.
        down (nnLinear): Used to transform the output of the upward pass through
            the network, which consists of a matrix multiplication between the
            input and weights of the layer.

    """
    def __init__(self):
        """
        Defines two linear layers: `up` and `down`. The `up` layer maps the input
        to a larger vector space, while the `down` layer maps the output of the
        `up` layer back to the original input space.

        """
        super().__init__()
        self.up = nn.Linear(CONFIG.hidden_size, 4 * CONFIG.hidden_size)
        self.down = nn.Linear(self.up.out_features, self.up.in_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(torch.nn.functional.gelu(self.up(x)))


class PreNormResidual(nn.Module):
    """
    Takes a neural network module `body` and applies it to the input `x` after
    normalizing the input using layer normalization. The output is the sum of the
    original input and the modified output from the body module.

    Attributes:
        norm (nnLayerNorm): Used to apply layer normalization to the input tensor
            before passing it through the body module.
        body (nnModule): A module that takes the output of the layer norm and adds
            it to the input of the original module.

    """
    def __init__(self, body: nn.Module):
        """
        Initializes an instance of the class by defining a norm layer and assigning
        it to the `self.norm` attribute, followed by the assignment of a module
        (represented by the `body` parameter) to the `self.body` attribute.

        Args:
            body (nn.Module): Passed to the constructor as the second argument.

        """
        super().__init__()
        self.norm = nn.LayerNorm([CONFIG.hidden_size])
        self.body = body

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(self.norm(x))


class AbsolutePositionalEncoding(nn.Module):
    """
    Defines a neural network layer that adds an absolute positional encoding to
    the input sequence, using a learned weight vector.

    Attributes:
        weight (Parameter|torchTensor): Initialized with a random tensor of size
            (sequence length, hidden size).

    """
    def __init__(self):
        """
        Initializes an instance by setting its weight parameter to a random value
        drawn from a standard normal distribution with expected value 0 and standard
        deviation 1, which enables the model to learn positional information in
        the input sequence.

        """
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(CONFIG.sequence_length, CONFIG.hidden_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.weight


class Model(nn.Module):
    """
    Defines a neural network architecture consisting of an embedding layer, absolute
    positional encoding, layer normalization, and multiple residual blocks with
    attention and feedforward networks.

    Attributes:
        model (nnSequential): Defined as a chain of nn layers, including embedding,
            absolute positional encoding, layer normalization, residual blocks
            with attention and FFN, and linear layer.

    """
    def __init__(self):
        """
        Initializes the model's architecture, consisting of an embedding layer,
        absolute positional encoding, layer normalization, multiple residual blocks
        with attention and feedforward networks, and finally a linear layer to
        produce the output.

        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(256, CONFIG.hidden_size),
            AbsolutePositionalEncoding(),
            nn.LayerNorm([CONFIG.hidden_size]),
            *(
                nn.Sequential(PreNormResidual(Attention()), PreNormResidual(FFN()))
                for _ in range(CONFIG.depth)
            ),
            nn.LayerNorm([CONFIG.hidden_size]),
            nn.Linear(CONFIG.hidden_size, 256),
        )

    def forward(self, indices: Tensor) -> Tensor:
        """
        Computes cross-entropy loss between the flattened output of the model for
        the input indices up to the second-to-last dimension and the remaining
        one-dimensional indices.

        Args:
            indices (Tensor): 1D, representing a tensor of shape `(N, 2)`, where
                N is the batch size, containing the indices of the input samples
                to be classified.

        Returns:
            Tensor: The output of the cross-entropy loss calculation between the
            model's prediction and the true values.

        """
        return nn.functional.cross_entropy(
            self.model(indices[:, :-1]).flatten(0, -2), indices[:, 1:].flatten()
        )

# Extending the Model class
class ExtendedModel(Model):
    """
    Extends a base model by adding new methods `new_method()` and `forward()`,
    which multiply input values by 2, respectively, before passing them to the
    parent model's `forward()` method.

    """
    def __init__(self):
        super().__init__()

    def new_method(self, x: Tensor) -> Tensor:
        # New functionality
        return x * 2  # Example operation

    def forward(self, indices: Tensor) -> Tensor:
        # Optionally override the forward method
        return super().forward(indices) * 2  # Example modification

def train_cpu() -> Tensor:
    """
    Trains a machine learning model using the Adam optimizer and tensorboard to
    track loss values.

    Returns:
        Tensor: 4-dimensional, containing the average loss across all training batches.

    """
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG.lr)
    losses = []
    for batch in tqdm.tqdm(islice(batches(), CONFIG.steps), total=CONFIG.steps):
        opt.zero_grad()
        loss = model(batch)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return torch.tensor(losses)


def train_ipu() -> Tensor:
    """
    Trains a PyTorch model using Adam optimizer and tqdm for progress bar. It
    returns a tensor containing the trained model's output after processing a
    sequence of input batches.

    Returns:
        Tensor: A batch of the model's output for each training example in the
        given batches, evaluated using the Adam optimizer and the specified learning
        rate.

    """
    model = Model()
    options = poptorch.Options()
    options.showCompilationProgressBar(False)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG.lr)
    session = poptorch.trainingModel(model, options, opt)
    try:
        return torch.tensor(
            [
                float(session(batch.int()))
                for batch in tqdm.tqdm(
                    islice(batches(), CONFIG.steps), total=CONFIG.steps
                )
            ]
        )
    finally:
        session.destroy()


def train() -> Tensor:
    return train_ipu() if poptorch_available else train_cpu()
