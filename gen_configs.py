import theano
import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano_ops.Ops import dense, flatten
from theano_ops.activations import sigmoid, tanh
from theano_ops.base_model import TheanoModel
import config as CONF
from archs import dense_encoder, dense_decoder, conv_encoder, conv_decoder


class ICVAE(TheanoModel):

    def __init__(self, init_params=None,
                 input_shape=None,
                 target_shape=40,
                 encode_label=False,
                 nb_hidden=1024,
                 lmbd=1e-4,
                 nb_latent=200):

        self.encode_label = encode_label
        self.target_size = target_shape
        self.init_params = init_params
        self.input_size = input_shape
        self.lmbd = lmbd
        self.nb_latent = nb_latent
        self.encoded_label_size = 3 * self.nb_latent
        self.L = 4
        self.nb_hidden = nb_hidden
        self.class_label_size = self.target_size
        self.params = []
        self.to_regularize = []
        self._def_arch()
        self._def_cost_acc()
        self._generate = theano.function(
            [self.l_z, self.y], self.reconstructed)
        self._get_latent = theano.function([self.x], self.l_z)

    def generate(self, sample, label):
        return self._generate(sample, label)

    def _def_tensors(self):
        self.x = T.tensor4(dtype=theano.config.floatX, name='VAE_x')
        self.y = T.matrix(dtype='uint8', name='VAE_y')

    def _def_arch(self):
        self._def_tensors()
        self.mu, self.logsigma = self.encoder(self.x)
        self.l_z = self.sampler(self.mu, self.logsigma)
        self.reconstructed = self.decoder(self.l_z, self.y)
        self.outputs = self.reconstructed
        self.param_summary()

    def _def_cost_acc(self):
        # logpxzy = T.sum(
        #     -1. * T.nnet.binary_crossentropy(T.clip(self.outputs, 1e-7, 1 - 1e-7), self.x), axis=1)

        logpxzy = T.mean(
            T.sum((self.x - self.outputs) ** 2, axis=[1, 2, 3]))
        KLD = T.mean(
            -0.5 * T.sum(1 + self.logsigma - self.mu ** 2 - T.exp(self.logsigma), axis=1))
        logpx = logpxzy + 2.0 * KLD
        self.cost = T.mean(logpx) + self.lmbd * \
            np.array([T.sum(i ** 2) for i in self.to_regularize]).sum()
        # self.cost = - T.mean(logpx) + self.lmbd * np.array(
        #     [T.sum(i ** 2) for i in self.to_regularize]).sum()
        self.acc = T.mean(logpx)

    def label_encoder(self, x):
        code, _par = dense(x, self.class_label_size, self.encoded_label_size,
                           layer_name='omega', init_params=self.get_params('omega', self.init_params))
        code = sigmoid(code)
        self.params += _par
        self.to_regularize.append(_par[0])
        return code

    def encoder(self, x):
        # mu, logsigma, params, to_reg = dense_encoder(x=x,
        #                                              theano_model=self,
        #                                              input_size=self.input_size[
        #                                                  1],
        #                                              nb_hidden=self.nb_hidden,
        #                                              nb_latent=self.nb_latent,
        #                                              init_params=self.init_params)
        # self.params += params
        # self.to_regularize += to_reg

        mu, logsigma, params, to_reg = conv_encoder(x=x,
                                                    theano_model=self,
                                                    input_size=self.input_size,
                                                    latent_size=self.nb_latent,
                                                    init_params=self.init_params)
        self.params += params
        self.to_regularize += to_reg

        return mu, logsigma

    def sampler(self, mu, logsigma):
        l_z = self._sample_z(mu, logsigma).mean(axis=0)
        return l_z

    def _sample_z(self, mu, logsigma):
        srng = RandomStreams(seed=42)
        return mu + T.exp(0.5 * logsigma) * srng.normal((self.L, mu.shape[0], self.nb_latent))

    def decoder(self, l_z, y):
        y_hat = self.label_encoder(y) if self.encode_label else y
        label_size = self.encoded_label_size if self.encode_label else self.target_size
        l_z_class_conditional = T.concatenate([l_z, y_hat], axis=1)

        # outputs, params, to_reg = dense_decoder(x=l_z_class_conditional,
        #                                         theano_model=self,
        #                                         label_size=label_size,
        #                                         input_size=self.input_size[1],
        #                                         nb_hidden=self.nb_hidden,
        #                                         nb_latent=self.nb_latent,
        #                                         init_params=self.init_params)

        outputs, params, to_reg = conv_decoder(x=l_z_class_conditional,
                                               theano_model=self,
                                               input_size=self.input_size,
                                               latent_size=self.nb_latent +
                                               label_size,
                                               init_params=self.init_params)

        self.params += params
        self.to_regularize += to_reg
        return outputs
