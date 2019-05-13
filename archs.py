import theano
import numpy as np
import theano.tensor as T
from theano_ops.Ops import dense, conv_2d, bn, flatten
from theano_ops.activations import sigmoid, tanh, relu


def _conv_block(x, nb_filters, nb_channels, filter_size, block_idx, theano_model, double=True, init_params=None):
    params = []
    to_reg = []

    layer, pars = conv_2d(
        x, (nb_filters, nb_channels, filter_size,
            filter_size), layer_name=str(block_idx) + 'l1',
        init_params=theano_model.get_params(str(block_idx) + 'l1', param_list=init_params))
    params += pars
    to_reg.append(pars[0])

    layer, pars = bn(layer, trainable=True, layer_name=str(block_idx) + 'bn1',
                     init_params=theano_model.get_params(
                         str(block_idx) + 'bn1',
                     param_list=init_params))
    params += pars

    if double:
        layer, pars = conv_2d(
            layer, (2 * nb_filters, nb_filters, filter_size, filter_size), layer_name=str(block_idx) + 'l2',
            init_params=theano_model.get_params(str(block_idx) + 'l2', param_list=init_params))
        params += pars
        to_reg.append(pars[0])

    else:
        layer, pars = conv_2d(
            layer, (nb_filters, nb_filters, filter_size,
                    filter_size), layer_name=str(block_idx) + 'l2',
            init_params=theano_model.get_params(str(block_idx) + 'l2', param_list=init_params))
        params += pars
        to_reg.append(pars[0])
    return relu(layer), params, to_reg


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


def conv_encoder(x, theano_model, input_size, latent_size, init_params):
    params = []
    regs = []
    e1, pars = conv_2d(x, (8, 1, 5, 5),
                       layer_name='e1',
                       mode='half',
                       init_params=theano_model.get_params('e1', init_params))
    params += pars
    regs.append(pars[0])
    # e1 = bn(e1)
    e2, pars = conv_2d(tanh(e1),
                           (16, 8, 5, 5),
                       layer_name='e2',
                       mode='half',
                       init_params=theano_model.get_params('e2', init_params))
    params += pars
    regs.append(pars[0])
    # e2 = bn(e2)
    e3, pars = conv_2d(tanh(e2),
                           (32, 16, 5, 5),
                       layer_name='e3',
                       mode='half',
                       init_params=theano_model.get_params('e3', init_params))
    params += pars
    regs.append(pars[0])
    # e3 = bn(e3)
    mu, pars = dense(flatten(tanh(e3)),
                     input_size[2] * input_size[3] * 32,
                     latent_size,
                     layer_name='mu',
                     init_params=theano_model.get_params('mu', init_params))
    params += pars
    regs.append(pars[0])

    logsigma, pars = dense(flatten(tanh(e3)),
                           input_size[2] * input_size[3] * 32,
                           latent_size,
                           layer_name='logsigma',
                           init_params=theano_model.get_params('logsigma', init_params))
    params += pars
    regs.append(pars[0])

    return mu, logsigma, params, regs


def conv_decoder(x, theano_model, input_size, latent_size, init_params):
    params = []
    regs = []
    d4, pars = dense(x, latent_size, input_size[2] * input_size[3] * 32,
                     layer_name='d4ClassCond',
                     init_params=theano_model.get_params('d4ClassCond', init_params))
    params += pars
    regs.append(pars[0])
    d4 = T.reshape((d4),
                   (-1, 32, input_size[2], input_size[3]))
    # d4 = bn(d4)
    d3, pars = conv_2d(tanh(d4),
                           (16, 32, 5, 5),
                       layer_name='d3',
                       mode='half',
                       init_params=theano_model.get_params('d3', init_params))
    params += pars
    regs.append(pars[0])
    # d3 = bn(d3)
    d2, pars = conv_2d(tanh(d3),
                           (8, 16, 5, 5),
                       layer_name='d2',
                       mode='half',
                       init_params=theano_model.get_params('d2', init_params))
    params += pars
    regs.append(pars[0])
    # d2 = bn(d2)
    d1, pars = conv_2d(tanh(d2),
                           (1, 8, 5, 5),
                       layer_name='d1',
                       mode='half',
                       init_params=theano_model.get_params('d1', init_params))
    params += pars
    regs.append(pars[0])
    return sigmoid(d1), params, regs
