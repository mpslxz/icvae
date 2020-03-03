import theano
import theano.tensor as T
from theano_ops.base_model import TheanoModel

import gen_configs as CONF


class GeneratorModel(TheanoModel):

    def _def_tensors(self):
        self.x = T.tensor4(dtype=theano.config.floatX, name='G_x')
        self.y = T.tensor4(dtype='uint8', name='G_y')

    def _def_arch(self, init_params):
        self.model = CONF.ICVAE(init_params=init_params,
                                input_shape=self.INPUT_SHAPE)
        self.x = self.model.x
        self.y = self.model.y
        self.latent = self.model.l_z
        self.outputs = self.model.outputs
        self.params = self.model.params
        self.to_regularize = self.model.to_regularize

    def _def_cost_acc(self):
        self.acc = self.model.acc
        self.cost = self.model.cost

    def _def_functions(self):

        print("compiling model")
        self.batch_test_fcn = theano.function(
            [self.x, self.y], outputs=self.acc, mode=self.mode)
        self.batch_train_fcn = theano.function([self.x, self.y],
                                               outputs=self.output_metrics,
                                               updates=self.optimizer.updates(
                                               cost=self.cost, params=self.params),
                                               mode=self.mode)
        self.predict_fcn = theano.function(
            [self.x, self.y], outputs=self.outputs, mode=self.mode)
        self.get_latent = theano.function(
            [self.x], outputs=self.latent)

    def predict(self, x, y):
        return self.predict_fcn(x, y)

    def test(self, x, y):
        return self.batch_test_fcn(x, y)

    def get_latent(self, x):
        return self.model._get_latent(x)
