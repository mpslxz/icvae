import theano
import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano_ops.Ops import dense
from theano_ops.activations import sigmoid, tanh
from theano_ops.base_model import TheanoModel
import config as CONF


class ICVAE(TheanoModel):

    def __init__(self, init_params=None,
                 input_shape=None,
                 encode_label=False,
                 nb_hidden=1024,
                 lmbd=1e-4,
                 nb_latent=10):

        self.encode_label = encode_label
        self.INPUT_SHAPE = input_shape
        self.init_params = init_params
        self.input_size = CONF.input_size[1] * CONF.input_size[2]
        self.lmbd = lmbd
        self.nb_latent = nb_latent
        self.encoded_label_size = 3 * self.nb_latent
        self.L = 4
        self.nb_hidden = nb_hidden
        self.class_label_size = self.input_size
        self.params = []
        self.to_regularize = []
        self._def_arch()
        self._def_cost_acc()
        self._generate = theano.function(
            [self.l_z, self.y], self.reconstructed)

    def generate(self, sample, label):
        return self._generate(sample, label)

    def _def_tensors(self):
        self.x = T.matrix(dtype=theano.config.floatX, name='VAE_x')
        self.y = T.matrix(dtype='uint8', name='VAE_y')

    def _def_arch(self):
        self._def_tensors()
        self.mu, self.logsigma = self.encoder(self.x)
        self.l_z = self.sampler(self.mu, self.logsigma)
        self.reconstructed = self.decoder(self.l_z, self.y)
        self.outputs = self.reconstructed

    def _def_cost_acc(self):
        logpxzy = T.sum(
            -1 * T.nnet.binary_crossentropy(self.outputs, self.x), axis=1)
        KLD = 0.5 * \
            T.sum(1 + self.logsigma - self.mu ** 2 -
                  T.exp(self.logsigma), axis=1)

        logpx = logpxzy + KLD
        self.cost = - T.mean(logpx) + self.lmbd * np.array(
            [T.sum(i ** 2) for i in self.to_regularize]).sum()
        self.acc = T.mean(logpxzy)

    def label_encoder(self, x):
        code, _par = dense(x, self.class_label_size, self.encoded_label_size,
                           layer_name='omega', init_params=self.get_params('omega', self.init_params))
        code = sigmoid(code)
        self.params += _par
        self.to_regularize.append(_par[0])
        return code

    def encoder(self, x):
        l_enc_hid, _par = dense(x, self.input_size, self.nb_hidden,
                                layer_name='encHid', init_params=self.get_params('encHid', self.init_params))
        l_enc_hid = tanh(l_enc_hid)
        self.params += _par
        self.to_regularize.append(_par[0])

        l_enc_hid_sec, _par = dense(
            l_enc_hid, self.nb_hidden, self.nb_hidden / 2,
            layer_name='encHidSec', init_params=self.get_params('encHidSec', self.init_params))
        l_enc_hid_sec = tanh(l_enc_hid_sec)
        self.params += _par
        self.to_regularize.append(_par[0])

        l_enc_mu, _par = dense(
            l_enc_hid_sec, self.nb_hidden / 2, self.nb_latent, layer_name='encMu', init_params=self.get_params('encMu', self.init_params))
        self.params += _par
        self.to_regularize.append(_par[0])

        l_enc_logsigma, _par = dense(
            l_enc_hid_sec, self.nb_hidden / 2, self.nb_latent, layer_name='encLogsigma', init_params=self.get_params('encLogsigma', self.init_params))
        self.params += _par
        self.to_regularize.append(_par[0])

        return l_enc_mu, l_enc_logsigma

    def sampler(self, mu, logsigma):
        l_z = self._sample_z(mu, logsigma).mean(axis=0)
        return l_z

    def _sample_z(self, mu, logsigma):
        srng = RandomStreams(seed=42)
        return mu + T.exp(0.5 * logsigma) * srng.normal((self.L, mu.shape[0], self.nb_latent))

    def decoder(self, l_z, y):
        y_hat = self.label_encoder(y) if self.encode_label else y
        l_z_class_conditional = T.concatenate([l_z, y_hat], axis=1)
        l_dec_hid_sec, _par = dense(
            l_z_class_conditional, self.nb_latent + self.encoded_label_size, self.nb_hidden / 2, layer_name='decHidSec', init_params=self.get_params('decHidSec', self.init_params))
        self.params += _par
        self.to_regularize.append(_par[0])
        l_dec_hid_sec = tanh(l_dec_hid_sec)

        l_dec_hid, _par = dense(
            l_dec_hid_sec, self.nb_hidden / 2, self.nb_hidden, layer_name='decHid', init_params=self.get_params('decHid', self.init_params))
        self.params += _par
        self.to_regularize.append(_par[0])
        l_dec_hid = tanh(l_dec_hid)

        l_rec_x, _par = dense(
            l_dec_hid, self.nb_hidden, self.input_size, layer_name='decMu', init_params=self.get_params('decMu', self.init_params))
        self.params += _par
        self.to_regularize.append(_par[0])
        l_rec_x = sigmoid(l_rec_x)
        outputs = l_rec_x
        return outputs
