# Progressive model
# (i.e. unmixing happens 'progressively' in each layer)

'''
* Common abstract base class for all architectures
    This common class implements common things such as
    placeholder definition, step counter, acc, cost, 
    optimization, init, train_model, estimate_model, 
    save_model, load_model, close
* Each model inherits from common class
    Child classes implement specific architectures
'''

##### TODOS MAYBE ? #####
# For nonlinear recomposition: use more 'progressive' nonlinearities
# than relu/lrelu... For example ELU with some learnable scaling
# factors ... ??


##### TODOS LATER #######
# TODO: TRY tf.contrib.layers.variance_scaling_initializer
# TODO: Think whether batch normalization is adapted or not


import tensorflow as tf
import numpy as np

from synth_eeg.synth_eeg_graph import create_synth_data_graph
from synth_eeg import mix_mat

class ModelBase():
    # Abstract class for code and functionalities common to
    # all architectures and modes (ICA/classif)
    def __init__(self, 
                 in_size=512,
                 batch_size=128,
                 n_chan_in=19):

        self.in_size = in_size
        self.batch_size = batch_size
        self.n_chan_in = n_chan_in

        # placeholders
        # placeholders for in_X (and possibly target_Y)
        # will be defined in child classes
        self.lr = tf.placeholder(tf.float32, [])
        self.phase = tf.placeholder(tf.bool) # in case child class uses BN

        # step counter to give to saver, increment in optimizer
        self.global_step = tf.Variable(
            0, name='global_step', trainable=False)

    def init(self):
        self.session = tf.Session()
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)
    
    def save_model(self, checkpoint_path):
        print('Saving model...')
        self.saver.save(
            self.session, checkpoint_path)
        # let's use only for keeping the best model.
        # global_step saved as a variable, not passed
        # in .ckpt name

    def load_model(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #msg = ("total path is longer than 255 characters, "
        #      "not ok on NTFS volume")
        #assert len(ckpt.model_checkpoint_path) <= 255, msg
        print("loading model: ", ckpt.model_checkpoint_path)
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        
        print('Model loaded. ')

    def close(self):
        self.session.close()


class ICAModelBase(ModelBase):

    # common functions for training a decomposition
    # (cost etc.. different from classif model but 
    # architectures can be the same)

    # Measure of non-Gaussianity:
    # For a random variable y, an approximation of negentropy is:
    # J(y) = [E(G(y)) - E(G(g))]^2
    # where g is a a gaussian variable with zero mean and unit variance
    # and G is some non-quadratic function such as
    # G1(y) = 1/a * log cosh(ay), 1<=a<=2 
    # or G2(y) = -exp(-y^2 / 2)
    # REM: y is assumed zero mean and unit variance so if we 
    # do not have the guarantee that it is, let's center and scale it. 

    # let's use G2. Then the term E(G(g)) is -sqrt(1/2)
    
    def __init__(self, 
                 alpha_decorr,
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha_decorr = alpha_decorr

        ####### BUILD GRAPH ######
        # placeholder for input
        self.in_X = tf.placeholder(
            tf.float32, [self.batch_size, self.in_size, self.n_chan_in])

        transformed_X = self.get_transformed()
        # shape (batch_size, signal_len, n_chan)
        n_chan = transformed_X.get_shape()[2].value

        # center and normalize channels
        mean, var = tf.nn.moments(transformed_X, [1], keep_dims=True) #(batch, 1, n_chan)
        #in some cases n_chan can be different from self.n_chan_in
        self.transformed_X = (transformed_X - mean) / tf.sqrt(1e-8 + var)

        # We need not only the moments on transformed channels
        # (to center them) but also the covariances between channels
        # in order to force them to zero, break symmetry and avoid 
        # a solution with identical output channels
        # Let's calculate covariances
        covmats_tr_X = get_covmat(self.transformed_X, reduce=True)
        _, offdiag_covs = get_diag_outdiag(covmats_tr_X)

        # Main 'NON-gaussianity' term (we want to maximize it)
        G2 = -tf.exp(-tf.square(self.transformed_X) / 2.)
        E_G2 = tf.reduce_mean(G2, axis=1)
        J = tf.square(E_G2 + np.sqrt(0.5).astype(np.float32))
        # And decorrelation term: #REM: Use SQRT of sum or not ???
        C_decorr = tf.reduce_sum(
            tf.square(offdiag_covs), axis=1)

        ############## COST AND OPTIMIZATION ################
        self.COST_ICA = tf.reduce_mean(-J) # mean on batch and channels
        self.COST_DECORR = self.alpha_decorr * tf.reduce_mean(C_decorr)
        self.COST = self.COST_ICA + self.COST_DECORR
        # optim
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # tie BN statistics update to optimizer step
        # (in case BN is used in child class)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.OP = optimizer.minimize(
                self.COST, global_step=self.global_step)

        # saver
        self.saver = tf.train.Saver(max_to_keep=1)
        # (let's keep only the best model)

    def get_transformed(self):
        # to be implemented in child classes
        raise NotImplementedError

    def train_model(self, numeric_in):
        _, cost_value, cost_ica_value, cost_decorr_value = \
        self.session.run(
            [self.OP, self.COST, self.COST_ICA, self.COST_DECORR],
            feed_dict={
                self.in_X: numeric_in['in_X'],
                self.lr: numeric_in['lr'],
                self.phase: 1,
            })
        return cost_value, cost_ica_value, cost_decorr_value

    def estimate_model(self, numeric_in):
        cost_value, cost_ica_value, cost_decorr_value = \
        self.session.run(
            [self.COST, self.COST_ICA, self.COST_DECORR],
            feed_dict={
                self.in_X: numeric_in['in_X'],
                self.phase: 0,
            })
        return cost_value, cost_ica_value, cost_decorr_value

    def use_model(self, numeric_in):
        transformed_signals = self.session.run(
            self.transformed_X, 
            feed_dict={
                self.in_X: numeric_in['in_X'],
                self.phase: 0,
            })
        return transformed_signals


class ClassifModelBase(ModelBase):

    # common functions for training a classification model

    def __init__(self, 
                 n_totalsphere_sources,
                 r_sources,
                 r_zone,
                 radial_only,
                 fixed_ampl,
                 different_zone_distrib,
                 center_sources_idxs,
                 r_geom,
                 sigmas,
                 **kwargs):
        # For this classif model, data will be generated
        # in-graph on the fly, so class first has to create
        # data graph, and then network graph

        # center_sources_idxs: a list of int, 
        # the possible source indices of zone centers

        super().__init__(**kwargs)
        self.center_sources_idxs = center_sources_idxs

        ###### BUILD DATA GRAPH #######

        n_sources = len(mix_mat.get_spread_points_on_sphere(
            n=n_totalsphere_sources, r=r_sources))
        # These are the idxs of sources amongst which
        # 'zone center' sources can be drawn:
        self.possible_center_idxs = tf.placeholder_with_default(
            np.arange(n_sources).astype(np.int32), shape=[None])

        common_args = {
            'n_totalsphere_sources': n_totalsphere_sources,
            'r_sources': r_sources,
            'in_size': self.in_size, 
            'r_zone': r_zone,
            'radial_only': radial_only,
            'fixed_ampl': fixed_ampl,
            'different_zone_distrib': different_zone_distrib,
            'r_geom': r_geom,
            'sigmas': sigmas,
        }

        if (len(center_sources_idxs)>=2) \
        and (-1 not in center_sources_idxs):
            # Here we do k-class classification
            # BCI-like task, differentiate spacial patterns
            # For k=2 also compare with CSP
            print('Task: k-class classification')
            self.n_classes = len(center_sources_idxs)
            batch_size_ = self.batch_size
            _, act_elec_clz, csource_idxs_amgst_possible = \
                create_synth_data_graph(
                    possible_center_idxs=center_sources_idxs,
                    batch_size=batch_size_,
                    **common_args)
            self.in_X = act_elec_clz
            self.target_Y = tf.one_hot(
                csource_idxs_amgst_possible, depth=self.n_classes)
        else:
            if (len(center_sources_idxs)>=2) \
                and (-1 in center_sources_idxs):
                # Here we do 2-class classification
                # k-class + background
                # Task: background vs. rest
                print(('Task: 2-class classification: '
                       '\'background\' vs \'rest in k-classes\''))
            elif center_sources_idxs==[-1]:
                # Here also 2 class classification
                # background vs. rest, in rest any source
                # can serve as center. 
                print(('Task: 2-class classification: '
                       '\'background\' vs \'rest amongst all possible sources\''))
            else:
                raise ValueError(('Check center_sources_idxs ?'
                                  'This case was not envisioned'))
            # in both of these cases:    
            self.n_classes = 2
            if self.batch_size % 2 != 0:
                raise ValueError(
                    '2 must divide batch size. ')
            batch_size_ = self.batch_size // 2
            act_elec_clb, act_elec_clz, _ = \
            create_synth_data_graph(
                possible_center_idxs=center_sources_idxs,
                batch_size=batch_size_,
                **common_args)
            self.in_X = tf.concat(
                [act_elec_clb, act_elec_clz], axis=0)
            # let's use label 0 for background, 1 for zone
            self.target_Y = tf.concat(
                [tf.one_hot([0]*batch_size_, depth=self.n_classes), 
                 tf.one_hot([1]*batch_size_, depth=self.n_classes)], 
                axis=0)


        ####### BUILD NETWORK GRAPH #######
        # architecture
        out_vals = self.get_out_vals()
        # (batch_size, layer_sizes[-1])
        # a last (fully-connected) layer for classification:
        out_logits = tf.layers.dense(
            out_vals,
            units=self.n_classes,
            activation=None,
            name='dense_classif')

        ############ ACC, COST, OPTIMIZATION ################
        self.COST = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.target_Y,
                logits=out_logits))
        # for external eval
        self.preds = tf.nn.softmax(out_logits, dim=-1)
        self.preds_int = tf.argmax(self.preds, axis=1)
        self.preds_true = tf.equal(
            self.preds_int, tf.argmax(self.target_Y, axis=1))
        self.acc = tf.reduce_mean(
            tf.cast(self.preds_true, tf.float32))
        # optimization
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # tie BN statistics update to optimizer step
        # (in case BN is used in child class)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.OP = optimizer.minimize(
                self.COST, global_step=self.global_step)

        # saver
        self.saver = tf.train.Saver(
            tf.all_variables(), max_to_keep=1)
        # (let's keep only the best model)

    def get_out_vals(self):
        # to be implemented in child classes
        raise NotImplementedError

    def train_model(self, numeric_in):
        d = {self.lr: numeric_in['lr'], self.phase: 1,}
        if self.center_sources_idxs != [-1]:
            d_ = {self.possible_center_idxs: self.center_sources_idxs}
            d.update(d_)

        _, acc_value, cost_value = self.session.run(
            [self.OP, self.acc, self.COST],
            feed_dict=d)

        return acc_value, cost_value

    def estimate_model(self):
        d = {self.phase: 0,}
        if self.center_sources_idxs != [-1]:
            d_ = {self.possible_center_idxs: self.center_sources_idxs}
            d.update(d_)

        pred_values, acc_value, cost_value = self.session.run(
            [self.preds_int, self.acc, self.COST],
            feed_dict=d)
        
        return pred_values, acc_value, cost_value


class MLAdaptiveSF_ICA(ICAModelBase):
    # Multilayer Adaptive Spacial Filter for ICA comparison
    
    def __init__(self, 
                 layer_sizes,
                 activations_conv11,
                 activations_to_W, 
                 activations_to_out,
                 use_backconnects,
                 use_var_feat_only_net,
                 **kwargs):

        self.layer_sizes = layer_sizes
        self.activations_conv11 = activations_conv11
        self.activations_to_W = activations_to_W
        self.activations_to_out = activations_to_out
        self.use_backconnects = use_backconnects
        self.use_var_feat_only_net = use_var_feat_only_net
        super().__init__(**kwargs)

    def get_transformed(self):

        c = MultiLayerAdaptiveSpacialFilter(
            self.in_X,
            layer_sizes = self.layer_sizes,
            activations_conv11 = self.activations_conv11,
            activations_to_W = self.activations_to_W,
            activations_to_out = self.activations_to_out,
            use_backconnects = self.use_backconnects,
            use_var_feat_only = self.use_var_feat_only_net)
        return c


class MLSF_ICA(ICAModelBase):
    # Multilayer (non-adaptive) Spatial Filter for ICA

    def __init__(self, 
                 layer_sizes, 
                 activations,
                 **kwargs):

        self.layer_sizes = layer_sizes
        self.activations = activations
        super().__init__(**kwargs)

    def get_transformed(self):

        c = MultiLayerSpacialFilter(
            self.in_X,
            layer_sizes = self.layer_sizes,
            activations = self.activations)
        return c


class MLSF_classif(ClassifModelBase):
    # MultiLayer Spacial Filter, non adaptive, 
    # used for classication

    def __init__(self, 
                 layer_sizes,
                 activations,
                 use_var_feat_only_classif, 
                 **kwargs):

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.use_var_feat_only_classif = use_var_feat_only_classif
        super().__init__(**kwargs)

    def get_out_vals(self):

        c = MultiLayerSpacialFilter(
            self.in_X,
            layer_sizes = self.layer_sizes,
            activations = self.activations)
        # (batch_size, in_size, n_chan)

        # Actually, if using only an overall linear transform, 
        # it is probably not a good idea to use mean, var...
        mean, var = tf.nn.moments(c, [1]) # (batch_size, n_chan)
        if not self.use_var_feat_only_classif:
            spa_feats = tf.concat([mean, var], axis=1)
        else: spa_feats = var
        return spa_feats
        # TRY to use a non-gaussianity measure instead
        # (it simulates better what a temporal detector would do)
        # n_chan = c.get_shape()[2].value
        # # center and normalize channels (because measure of NG needs this)
        # mean, var = tf.nn.moments(c, [1], keep_dims=True)
        # c = (c - mean) / tf.sqrt(1e-8 + var)
        # # Main 'NON-gaussianity' term
        # G2 = -tf.exp(-tf.square(c) / 2.)
        # E_G2 = tf.reduce_mean(G2, axis=1)
        # J = tf.square(E_G2 + np.sqrt(0.5).astype(np.float32))
        # J = tf.multiply(J, 10000)
        # # (batch_size, n_chan)
        # return J




class MLAdaptiveSF_classif(ClassifModelBase):
    # MultiLayer Adaptive Spacial Filter used for classification

    def __init__(self, 
                 layer_sizes,
                 activations_conv11,
                 activations_to_W, 
                 activations_to_out,
                 use_backconnects,
                 use_var_feat_only_net,
                 use_var_feat_only_classif,
                 **kwargs):

        self.layer_sizes = layer_sizes
        self.activations_conv11 = activations_conv11
        self.activations_to_W = activations_to_W
        self.activations_to_out = activations_to_out
        self.use_backconnects = use_backconnects
        self.use_var_feat_only_net = use_var_feat_only_net
        self.use_var_feat_only_classif = use_var_feat_only_classif
        super().__init__(**kwargs)


    def get_out_vals(self):

        c = MultiLayerAdaptiveSpacialFilter(
            self.in_X,
            layer_sizes = self.layer_sizes,
            activations_conv11 = self.activations_conv11,
            activations_to_W = self.activations_to_W,
            activations_to_out = self.activations_to_out,
            use_backconnects = self.use_backconnects,
            use_var_feat_only = self.use_var_feat_only_net)
        
        # Actually, if using only an overall linear transform, 
        # it is probably not a good idea to use mean, var...
        mean, var = tf.nn.moments(c, [1]) # (batch_size, n_chan)
        if not self.use_var_feat_only_classif:
            spa_feats = tf.concat([mean, var], axis=1)
        else: spa_feats = var
        return spa_feats 

        # # TRY to use a non-gaussianity measure instead
        # # (it simulates better what a temporal detector would do)
        # n_chan = c.get_shape()[2].value
        # # center and normalize channels (because measure of NG needs this)
        # mean, var = tf.nn.moments(c, [1], keep_dims=True)
        # c = (c - mean) / tf.sqrt(1e-8 + var)
        # # Main 'NON-gaussianity' term
        # G2 = -tf.exp(-tf.square(c) / 2.)
        # E_G2 = tf.reduce_mean(G2, axis=1)
        # J = tf.square(E_G2 + np.sqrt(0.5).astype(np.float32))
        # # normalize along channels
        # Jv = tf.reduce_mean(J, axis=-1, keep_dims=True)
        # J = tf.divide(J, tf.sqrt(1e-8 + Jv)) # replace by BN ??
        # # (batch_size, n_chan)
        # #J = tf.multiply(J, 10000)
        # return J       


def lrelu(x, leak=0.1, name='lrelu'):
    return tf.maximum(x, leak*x, name)


def get_covmat(tsr, reduce=False):
    # tensor tsr is assumed to have shape (batch_size, in_size, n_chan)
    # covariances are calculated between the n_chan channels
    # and along the in_size dimension. 
    # other dimensions are kept. 

    # ALL CHANNELS ARE EXPECTED TO BE CENTERED !

    # broadcast channel dims before elmt-wise multiply
    # in order to calculate pairwise combinations
    batch_size = tsr.get_shape()[0].value
    in_size = tsr.get_shape()[1].value
    n_chan = tsr.get_shape()[2].value

    tsr_0 = tf.reshape(tsr, [batch_size, in_size, 1, n_chan])
    tsr_1 = tf.reshape(tsr, [batch_size, in_size, n_chan, 1])
    covs = tf.multiply(tsr_0, tsr_1)
    # shape (.., in_size, n_chan, n_chan)
    if reduce:
        covs = tf.reduce_mean(covs, axis=1)
    # shape (.., n_chan, n_chan)
    return covs


def get_diag_outdiag(covs):
    # covs: a covmat of shape (batch_size, n_chan, n_chan)
    # OR (batch_size, in_size, n_chan, n_chan)
    #
    # tensorflow does not have an option for indexing along 
    # a diagonal 
    # This function gets diagonal and off-diag terms from
    # a covmat
    # --------------------------------------------------------
    # EXAMPLE WITH DIAG TERMS: 
    # ------------------------
    # So let's do this by 'offsetting' the matrix, for example
    # (not showing batch dim)
    # covs = [[d, n, n, n],
    #         [n, d, n, n],
    #         [n, n, d, n],
    #         [n, n, n, d]]
    # tf.reshape(covs, [4*4])
    # covs = [d, n, n, n, n, d, n, n, n, n, d, n, n, n, n, d]
    # covs_allm1 = covs[0:(4*4-1)]
    # covs_last = covs[-1]
    # tf.reshape(cov_allm1, [3, 5])
    # [[d, n, n, n, n],
    #  [d, n, n, n, n],
    #  [d, n, n, n, n]]
    # now diagonal elemts are the first column + covs_last
    # and non-diagonal elements are the other columns

    n_dim = len(covs.get_shape())
    batch_size = covs.get_shape()[0].value
    n_chan = covs.get_shape()[-1].value
    if n_dim==3:
        covs = tf.reshape(covs, [batch_size, n_chan*n_chan])
        covs_allm1 = covs[:, 0:n_chan*n_chan-1]
        covs_last = tf.reshape(covs[:, -1],
            [batch_size, 1])
        covs_allm1 = tf.reshape(covs_allm1, 
            [batch_size, n_chan-1, n_chan+1])
        diag_terms = covs_allm1[:, :, 0]
        diag_terms = tf.concat([diag_terms, covs_last], axis=1)
        offdiag_terms = covs_allm1[:, :, 1:]
        offdiag_terms = tf.reshape(offdiag_terms, 
            [batch_size, n_chan*(n_chan-1)])
        # REM: we lose the order and we have duplicates (because
        # cov is symmetric... but let's not care for now. )
    elif n_dim==4:
        # is there a way to avoid this duplicate code
        # with tensorflow indexing ? Is .. supported in tf ?
        in_size = covs.get_shape()[1].value
        covs = tf.reshape(covs, [batch_size, in_size, n_chan*n_chan])
        covs_allm1 = covs[:, :, 0:n_chan*n_chan-1]
        covs_last = tf.reshape(covs[:, :, -1],
            [batch_size, in_size, 1])
        covs_allm1 = tf.reshape(covs_allm1,
            [batch_size, in_size, n_chan-1, n_chan+1])
        diag_terms = covs_allm1[:, :, :, 0]
        diag_terms = tf.concat([diag_terms, covs_lasxt], axis=2)
        offdiag_terms = covs_allm1[:, :, :, 1:]
        offdiag_terms = tf.reshape(offdiag_terms,
            [batch_size, in_size, n_chan*(n_chan-1)])

    return diag_terms, offdiag_terms


def MultiLayerSpacialFilter(in_X,
                            layer_sizes=[19], # or list for MultiLayer
                            activations=lrelu):

    # here we simply learn len(layer_sizes) spatial linear filters
    # and apply nonlinearities
    # sum along output channels are trained to be 
    # maximally discriminative for 

    in_size = in_X.get_shape()[1].value
    n_chan_in = in_X.get_shape()[2].value

    c = tf.reshape(
        in_X, [-1, 1, in_size, n_chan_in])

    for l in range(len(layer_sizes)):
        c = tf.layers.conv2d(
            c, 
            filters=layer_sizes[l],
            kernel_size=[1, 1],
            strides=1,
            activation=activations,
            name="spa_fil_%d" %l)

    c = tf.reshape(c, [-1, in_size, layer_sizes[-1]])
    return c


def AdaptiveSpacialLayer(in_X, 
                         layer_size,
                         activation_conv11,
                         activation_to_W,
                         activation_to_out,
                         use_backconnect=False,
                         prev_spa_feats=None,
                         use_var_feat_only=False,
                         normalize_Wb=True,
                         name=""):
    # adaptive spacial layer, to be used in MultiLayerAdaptiveSpacialFilter
    # expects and returns input of shape [batch_size, 1, in_size, n_chan]

    # if activation_to_out is not None, we also use a nonlinearity
    # with the main filter application

    batch_size = in_X.get_shape()[0].value
    in_size = in_X.get_shape()[2].value
    n_chan = in_X.get_shape()[3].value

    # make sure that if using backconnects, 
    # a prev_spa_feats is provided
    if use_backconnect and (prev_spa_feats is None):
        raise ValueError('If using a backconnect, please provide' + \
                         'a prev_spa_feats as input. ')

    c11 = tf.layers.conv2d(
        in_X, filters=layer_size,
        kernel_size=[1, 1], strides=1,
        activation=activation_conv11,
        name="c11_"+name)
    # optionally also add 3*1 filter here for spatio-
    # temporal filtering, and concatenate (later)
    mean, var = tf.nn.moments(c11, [1, 2]) # [batch_size, layer_size]
    spa_feats = var
    mulfac_spa_feats = 1
    if not use_var_feat_only:
        spa_feats = tf.concat([mean, var], axis=1)
        mulfac_spa_feats = 2

    if use_backconnect:
        # note that if using backconnects we do not necessarily need
        # that the current layer has the same number of units as
        # the previous layer, but it must have the SAME NUMBER
        # OF SPACIAL FEATURES

        # let's learn ONE alpha coeff per feature, so that
        # the network is able to 'decide to use different depths...'
        alpha = tf.get_variable(
            'alpha_'+name, 
            initializer=tf.ones([1, layer_size*mulfac_spa_feats]))
        spa_feats = tf.multiply(alpha, spa_feats) \
                    + tf.multiply(1. - alpha, prev_spa_feats)
    
    Wbdense_units = (n_chan + 1) * n_chan \
        if activation_to_out is not None else n_chan * n_chan

    if activation_to_out is None:
        Wb = tf.layers.dense(
            spa_feats, units=n_chan*n_chan,
            activation=activation_to_W,
            name="dense_"+name)
        W = Wb
        if normalize_Wb:
            Wmean = tf.reduce_mean(W, axis=-1, keep_dims=True)
            Wscale = tf.get_variable(
                "Wscale_"+name, initializer=0.01)
            W = tf.multiply(Wscale, tf.divide(W, Wmean))
    else:
        Wb = tf.layers.dense(
            spa_feats, units=n_chan*(n_chan+1),
            activation=activation_to_W,
            name="denseWb_"+name)
        W = Wb[:, :n_chan*n_chan]
        b = Wb[:, n_chan*n_chan:]
        if normalize_Wb:
            _, Wvar = tf.nn.moments(W, [-1], keep_dims=True)
            _, bvar = tf.nn.moments(b, [-1], keep_dims=True)
            Wstd = tf.sqrt(1e-8 + Wvar)
            bstd = tf.sqrt(1e-8 + bvar)
            Wscale = tf.get_variable(
                "Wscale_"+name, initializer=0.01)
            bscale = tf.get_variable(
                "bscale_"+name, initializer=0.01)
            W = tf.multiply(Wscale, tf.divide(W, Wstd))
            b = tf.multiply(bscale, tf.divide(b, bstd))
    I = tf.eye(n_chan)
    I = tf.reshape(I, [1, n_chan*n_chan])
    W = W + I # learn residual part
    # we will use this as weights of the input transformation layer

    # -------------------------
    # Now we apply the W (one DIFFERNET for EACH element of the batch)
    # For this we 'hack' a depthwise convolution:
    W = tf.reshape(W, [1, 1, batch_size*n_chan, n_chan])
    # in_X has shape (batch_size, 1, in_size, n_chan)
    in_X_r = tf.transpose(in_X, [1, 2, 0, 3])
    in_X_r = tf.reshape(in_X_r, [1, 1, in_size, batch_size*n_chan])

    out_filtered = tf.nn.depthwise_conv2d(
        in_X_r,
        filter=W,
        strides=[1, 1, 1, 1],
        padding='VALID') #REM: here we don't care about padding 
        # but still have to provide a value
    # out_filtered shape: (1, 1, in_size, batch_size*n_chan*n_chan)
    out_filtered = tf.reshape(
        out_filtered, [1, in_size, batch_size, n_chan, n_chan])
    out_filtered = tf.transpose(out_filtered, [2, 0, 1, 3, 4])
    # and finally sum on input chans
    out_filtered = tf.reduce_sum(out_filtered, axis=3)
    # shape (batch_size, 1, in_size, n_chan)
    if activation_to_out is not None:
        b = tf.reshape(b, [batch_size, 1, 1, n_chan])   
        out_filtered = out_filtered + b
        out_filtered = activation_to_out(out_filtered)

    return out_filtered if not use_backconnect \
        else spa_feats, out_filtered


def MultiLayerAdaptiveSpacialFilter(in_X, 
                                    layer_sizes=[19],
                                    activations_conv11=lrelu,
                                    activations_to_W=lrelu,
                                    activations_to_out=lrelu,
                                    use_backconnects=False,
                                    use_var_feat_only=False):
    
    # here at each layer we determine spacial features,
    # which we then use to determine (through an FC layer)
    # the spacial filter to be applied

    # an optional backconnect injects add features from
    # previous layers to current layer
    # if using backconnects, all layers must use the same
    # number of spatial features:
    if use_backconnects:
        s_ = [layer_sizes[0]]*len(layer_sizes)
        if not s_ == layer_sizes:
            raise ValueError('If using a backconnect, please use' + \
                             'layer sizes that are all equal')

    batch_size = in_X.get_shape()[0].value
    in_size = in_X.get_shape()[1].value
    n_chan = in_X.get_shape()[2].value

    c = tf.reshape(
        in_X, [batch_size, 1, in_size, n_chan])
    # if using backconnects, in any case for the first layer
    # provide zeros as spacial features
    mulfac_spa_feats = 2 if not use_var_feat_only else 1
    prev_spa_feats = None if not use_backconnects \
        else tf.zeros([batch_size, layer_sizes[0]*mulfac_spa_feats])

    for l in range(len(layer_sizes)):
        res = AdaptiveSpacialLayer(
            c, layer_size=layer_sizes[l],
            activation_conv11=activations_conv11,
            activation_to_W=activations_to_W,
            activation_to_out=activations_to_out,
            use_backconnect=use_backconnects, 
            prev_spa_feats=prev_spa_feats,
            use_var_feat_only=use_var_feat_only,
            name="adapt_%d" %l)
        if len(res) == 2:
            prev_spa_feats, c = res
        else:
            c = res

    c = tf.reshape(c, [-1, in_size, n_chan])
    return c




