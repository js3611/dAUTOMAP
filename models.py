import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_fourier_(tensor, norm='ortho'):
    """Initialise convolution weight with Inverse Fourier Transform"""
    with torch.no_grad():
        # tensor should have shape: (nc_out, nc_in, kx, ky)=(2*N, 2, N, kernel_size)
        nc_out, nc_in, N, kernel_size = tensor.shape

        for k in range(N):
            for n in range(N):
                tensor.data[k, 0, n, kernel_size // 2] = np.cos(2 * np.pi * n * k / N)
                tensor.data[k, 1, n, kernel_size // 2] = -np.sin(2 * np.pi * n * k / N)
                tensor.data[k + N, 0, n, kernel_size // 2] = np.sin(2 * np.pi * n * k / N)
                tensor.data[k + N, 1, n, kernel_size // 2] = np.cos(2 * np.pi * n * k / N)

        if norm == 'ortho':
            tensor.data[...] = tensor.data[...] / np.sqrt(N)

        return tensor


def init_fourier_2d(N, M, inverse=True, norm='ortho', out_tensor=None,
                    complex_type=np.complex64):
    """Initialise fully connected layer as 2D Fourier transform

    Parameters
    ----------

    N, M: a number of rows and columns

    inverse: bool (default: True) - if True, initialise with the weights for
    inverse fourier transform

    norm: 'ortho' or None (default: 'ortho')

    out_tensor: torch.Tensor (default: None) - if given, copies the values to
    out_tensor

    """
    dft1mat_m = np.zeros((M, M), dtype=complex_type)
    dft1mat_n = np.zeros((N, N), dtype=complex_type)
    sign = 1 if inverse else -1

    for (l, m) in itertools.product(range(M), range(M)):
        dft1mat_m[l,m] = np.exp(sign * 2 * np.pi * 1j * (m * l / M))

    for (k, n) in itertools.product(range(N), range(N)):
        dft1mat_n[k,n] = np.exp(sign * 2 * np.pi * 1j * (n * k / N))

    # kronecker product
    mat_kron = np.kron(dft1mat_n, dft1mat_m)

    # split complex channels into two real channels
    mat_split = np.block([[np.real(mat_kron), -np.imag(mat_kron)],
                          [np.imag(mat_kron), np.real(mat_kron)]])

    if norm == 'ortho':
        mat_split /= np.sqrt(N * M)
    elif inverse:
        mat_split /= (N * M)

    if out_tensor is not None:
        out_tensor.data[...] = torch.Tensor(mat_split)
    else:
        out_tensor = mat_split
    return out_tensor


def init_noise_(tensor, init):
    with torch.no_grad():
        return getattr(torch.nn.init, init)(tensor) if init else tensor.zero_()


class GeneralisedIFT2Layer(nn.Module):

    def __init__(self, nrow, ncol,
                 nch_in, nch_int=None, nch_out=None,
                 kernel_size=1, nl=None,
                 init_fourier=True, init=None, bias=False, batch_norm=False,
                 share_tfxs=False, learnable=True):
        """Generalised domain transform layer

        The layer can be initialised as Fourier transform if nch_in == nch_int
        == nch_out == 2 and if init_fourier == True.

        It can also be initialised
        as Fourier transform plus noise by setting init_fourier == True and
        init == 'kaiming', for example.

        If nonlinearity nl is used, it is recommended to set bias = True

        One can use this layer as 2D Fourier transform by setting nch_in == nch_int
        == nch_out == 2 and learnable == False


        Parameters
        ----------
        nrow: int - the number of columns of input

        ncol: int - the number of rows of input

        nch_in: int - the number of input channels. One can put real & complex
        here, or put temporal coil channels, temporal frames, multiple
        z-slices, etc..

        nch_int: int - the number of intermediate channel after the transformation
        has been applied for each row. By default, this is the same as the input channel

        nch_out: int - the number of output channels. By default, this is the same as the input channel

        kernel_size: int - kernel size for second axis of 1d transforms

        init_fourier: bool - initialise generalised kernel with inverse fourier transform

        init_noise: str - initialise generalised kernel with standard initialisation. Option: ['kaiming', 'normal']

        nl: ('tanh', 'sigmoid', 'relu', 'lrelu') - add nonlinearity between two transformations. Currently only supports tanh

        bias: bool - add bias for each kernels

        share_tfxs: bool - whether to share two transformations

        learnable: bool

        """
        super(GeneralisedIFT2Layer, self).__init__()
        self.nrow = nrow
        self.ncol = ncol
        self.nch_in = nch_in
        self.nch_int = nch_int
        self.nch_out = nch_out
        self.kernel_size = kernel_size
        self.init_fourier = init_fourier
        self.init = init
        self.nl = nl

        if not self.nch_int:
            self.nch_int = self.nch_in

        if not self.nch_out:
            self.nch_out = self.nch_in

        # Initialise 1D kernels
        idft1 = torch.nn.Conv2d(self.nch_in, self.nch_int * self.nrow, (self.nrow, kernel_size),
                                padding=(0, kernel_size // 2), bias=bias)
        idft2 = torch.nn.Conv2d(self.nch_int, self.nch_out * self.ncol, (self.ncol, kernel_size),
                                padding=(0, kernel_size // 2), bias=bias)

        # initialise kernels
        init_noise_(idft1.weight, self.init)
        init_noise_(idft2.weight, self.init)

        if self.init_fourier:
            if not (self.nch_in == self.nch_int == self.nch_out == 2):
                raise ValueError

            if self.init:
                # scale the random weights to make it compatible with FFT basis
                idft1.weight.data = F.normalize(idft1.weight.data, dim=2)
                idft2.weight.data = F.normalize(idft2.weight.data, dim=2)

            init_fourier_(idft1.weight)
            init_fourier_(idft2.weight)

        self.idft1 = idft1
        self.idft2 = idft2

        # Allow sharing weights between two transforms if the input size are the same.
        if share_tfxs and nrow == ncol:
            self.idft2 = self.idft1

        self.learnable = learnable
        self.set_learnable(self.learnable)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(self.nch_int)
            self.bn2 = torch.nn.BatchNorm2d(self.nch_out)

    def forward(self, X):
        # input shape should be (batch_size, nc, nx, ny)
        batch_size = len(X)
        # first transform
        x_t = self.idft1(X)

        # reshape & transform
        x_t = x_t.reshape([batch_size, self.nch_int, self.nrow, self.ncol]).permute(0, 1, 3, 2)

        if self.batch_norm:
            x_t = self.bn1(x_t.contiguous())

        if self.nl:
            if self.nl == 'tanh':
                x_t = F.tanh(x_t)
            elif self.nl == 'relu':
                x_t = F.relu(x_t)
            elif self.nl == 'sigmoid':
                x_t = F.sigmoid(x_t)
            else:
                raise ValueError

        # second transform
        x_t = self.idft2(x_t)
        x_t = x_t.reshape([batch_size, self.nch_out, self.ncol, self.nrow]).permute(0, 1, 3, 2)

        if self.batch_norm:
            x_t = self.bn2(x_t.contiguous())


        return x_t

    def set_learnable(self, flag=True):
        self.learnable = flag
        self.idft1.weight.requires_grad = flag
        self.idft2.weight.requires_grad = flag


def get_refinement_block(model='automap_scae', in_channel=1, out_channel=1):
    if model == 'automap_scae':
        return nn.Sequential(nn.Conv2d(in_channel, 64, 5, 1, 2), nn.ReLU(True),
                             nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(True),
                             nn.ConvTranspose2d(64, out_channel, 7, 1, 3))
    elif model == 'simple':
        return nn.Sequential(nn.Conv2d(in_channel, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, out_channel, 3, 1, 1))
    else:
        raise NotImplementedError


class AUTOMAP(nn.Module):
    """
    Pytorch implementation of AUTOMAP [1].

    Reference:
    ----------
    [1] Zhu et al., AUTOMAP, Nature 2018. <url:https://www.nature.com/articles/nature25988.pdf>
    """

    def __init__(self, input_shape, output_shape,
                 init_fc2_fourier=False,
                 init_fc3_fourier=False):
        super(AUTOMAP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.ndim = input_shape[-1]

        # "Mapped to hidden layer of n^2, activated by tanh"
        self.input_reshape = int(np.prod(self.input_shape))
        self.output_reshape = int(np.prod(self.output_shape))

        self.domain_transform = nn.Linear(self.input_reshape, self.output_reshape)
        self.domain_transform2 = nn.Linear(self.output_reshape, self.output_reshape)

        if init_fc2_fourier or init_fc3_fourier:
            if input_shape != output_shape:
                raise ValueError('To initialise the kernels with Fourier transform,'
                                 'the input and output shapes must be the same')

        if init_fc2_fourier:
            init_fourier_2d(input_shape[-2], input_shape[-1], self.domain_transform.weight)

        if init_fc3_fourier:
            init_fourier_2d(input_shape[-2], input_shape[-1], self.domain_transform2.weight)


        # Sparse convolutional autoencoder for further finetuning
        # See AUTOMAP paper
        self.sparse_convolutional_autoencoder = get_refinement_block('automap_scae', output_shape[0], output_shape[0])

    def forward(self, x):
        """Expects input_shape (batch_size, 2, ndim, ndim)"""
        batch_size = len(x)
        x = x.reshape(batch_size, int(np.prod(self.input_shape)))
        x = F.tanh(self.domain_transform(x))
        x = F.tanh(self.domain_transform2(x))
        x = x.reshape(-1, *self.output_shape)
        x = self.sparse_convolutional_autoencoder(x)
        return x


class dAUTOMAP(nn.Module):
    """
    Pytorch implementation of dAUTOMAP

    Decomposes the automap kernel into 2 Generalised "1D" transforms to make it scalable.
    """
    def __init__(self, input_shape, output_shape, tfx_params, tfx_params2):
        super(SMAP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.domain_transform = GeneralisedIFT2Layer(**tfx_params)
        self.domain_transform2 = GeneralisedIFT2Layer(**tfx_params2)
        self.processor = get_refinement_block('automap_scae', input_shape[0], output_shape[0])

    def forward(self, x):
        """Assumes input to be (batch_size, 2, nrow, ncol)"""
        x_mapped = self.domain_transform(x)
        x_mapped = F.tanh(x_mapped)
        x_mapped2 = self.domain_transform2(x_mapped)
        x_mapped2 = F.tanh(x_mapped2)
        out = self.processor(x_mapped2)
        return out


class dAUTOMAPExt(nn.Module):
    """
    Pytorch implementation of dAUTOMAP with adjustable depth and nonlinearity

    Decomposes the automap kernel into 2 Generalised "1D" transforms to make it scalable.

    Parameters
    ----------

    input_shape: tuple (n_channel, nx, ny)

    output_shape: tuple (n_channel, nx, ny)

    depth: int (default: 2)

    tfx_params: list of dict or dict. If list of dict, it must provide the parameter for each. If dict, then the same parameter config will be shared for all the layers.


    """
    def __init__(self, input_shape, output_shape, tfx_params=None, depth=2, nl='tanh'):
        super(SDAUTOMAP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.depth = depth
        self.nl = nl

        # copy tfx_parameters
        domain_transforms = []
        if isinstance(tfx_params, list):
            if self.depth and self.depth != len(tfx_params):
                raise ValueError('Depth and the length of tfx_params must be the same')
        else:
            tfx_params = [tfx_params] * self.depth

        # create domain transform layers
        for tfx_param in tfx_params:
            domain_transform = GeneralisedIFT2Layer(**tfx_param)
            domain_transforms.append(domain_transform)

        self.domain_transforms = nn.ModuleList(domain_transforms)
        self.refinement_block = get_refinement_block('automap_scae', input_shape[0], output_shape[0])

    def forward(self, x):
        """Assumes input to be (batch_size, 2, nrow, ncol)"""
        for i in range(self.depth):
            x = self.domain_transforms[i](x)
            x = getattr(F, self.nl)(x)

        out = self.refinement_block(x)
        return out
