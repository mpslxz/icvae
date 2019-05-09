import theano
import numpy as np
import theano.tensor as T
from theano_ops.Ops import dense
from theano_ops.activations import sigmoid, tanh


def dense_encoder(x, theano_model, input_size, nb_hidden, nb_latent, init_params):
    params = []
    to_regularize = []

    l_enc_hid, _par = dense(x, input_size, nb_hidden,
                            layer_name='encHid', init_params=theano_model.get_params('encHid', init_params))
    l_enc_hid = tanh(l_enc_hid)
    params += _par
    to_regularize.append(_par[0])

    l_enc_hid_sec, _par = dense(
        l_enc_hid, nb_hidden, nb_hidden / 2,
        layer_name='encHidSec', init_params=theano_model.get_params('encHidSec', init_params))
    l_enc_hid_sec = tanh(l_enc_hid_sec)
    params += _par
    to_regularize.append(_par[0])

    l_enc_mu, _par = dense(
        l_enc_hid_sec, nb_hidden / 2, nb_latent, layer_name='encMu', init_params=theano_model.get_params('encMu', init_params))
    params += _par
    to_regularize.append(_par[0])

    l_enc_logsigma, _par = dense(
        l_enc_hid_sec, nb_hidden / 2, nb_latent, layer_name='encLogsigma', init_params=theano_model.get_params('encLogsigma', init_params))
    params += _par
    to_regularize.append(_par[0])

    return l_enc_mu, l_enc_logsigma, params, to_regularize


def dense_decoder(x, theano_model, label_size, input_size, nb_hidden, nb_latent, init_params):
    params = []
    to_regularize = []

    l_dec_hid_sec, _par = dense(x, 
                                nb_latent + label_size, 
                                nb_hidden / 2, 
                                layer_name='decHidSec', 
                                init_params=theano_model.get_params('decHidSec', init_params))
    params += _par
    to_regularize.append(_par[0])

    l_dec_hid_sec = tanh(l_dec_hid_sec)

    l_dec_hid, _par = dense(l_dec_hid_sec, 
                            nb_hidden / 2, 
                            nb_hidden, 
                            layer_name='decHid', 
                            init_params=theano_model.get_params('decHid', init_params))
    params += _par
    to_regularize.append(_par[0])
    l_dec_hid = tanh(l_dec_hid)

    l_rec_x, _par = dense(l_dec_hid, 
                          nb_hidden, 
                          input_size, 
                          layer_name='decMu', 
                          init_params=theano_model.get_params('decMu', init_params))
    params += _par
    to_regularize.append(_par[0])
    l_rec_x = sigmoid(l_rec_x)
    outputs = l_rec_x

    return outputs, params, to_regularize
