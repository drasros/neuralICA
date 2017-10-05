# -----------------------------------
# POC (Proof of concept) on 'toy' ICA
# -----------------------------------

# train Multilayer Adaptive Spacial Filter ICA model 
# (with no nonlinearity on main filter)
# on simple 'ICA_poc_data'

import numpy as np
import tensorflow as tf

import argparse
import os
import sys

import ica_poc_data

from model import MLAdaptiveSF_ICA, MLSF_ICA, lrelu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#############################################
##########  CONFIG ##########################
results_dir = '/datadrive1/ica_demo/'
#results_dir = '/home/arnaud/data_these/multichan_cnn'

in_size = 512

sine_period = 16
sawtooth_period = 20

sine_period_valid = 20
sawtooth_period_valid = 30

# disable this or adapt with the api of your provider
send_txtmsg_when_done = True 
if send_txtmsg_when_done:
    import textmsg

#############################################

def write_to_comment_file(comment_file, text):
    with open(comment_file, "a") as f:
        f.write(text)

def get_exp_name(exp_type, exp_num, min_mix_coeff, min_Mlines_angle, 
                 batch_size, training_batches, learning_rates,
                 layer_sizes, alpha_decorr, use_backconnects):
    name_common = "_exp" + str(exp_num) \
                  + "b_" + str(*training_batches) \
                  + "_minmix" + str(min_mix_coeff) \
                  + "_minangle" + str(min_Mlines_angle) \
                  + '_lr' + str(*learning_rates) \
                  + "_lyr" + str(layer_sizes) \
                  + "_alph" + str(alpha_decorr)
    if exp_type == 'adaptive':
        name = "ica_adapt_exp" \
               + name_common \
               + "_bckcon" + str(use_backconnects)
        # VERY IMPORTANT: If your storage volume (unfortunately) uses
        # a NTFS filesystem, AVOID forbidden characters such as [, ] and ,
        # otherwise very weird things will happen: tensorflow save will 
        # work but restore not always, depending on the length of the path...
    elif exp_type == 'nonadaptive':
        name = "ica_nonadapt_exp" \
               + name_common
    else:
        raise ValueError('exp_type must be \'adaptive\' or \'nonadaptive\'. ')
    name = name.replace('[', '')
    name = name.replace(']', '')
    name = name.replace(', ', '_')
    return name

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_train_costs(save_dir):
    costs_train = np.load(os.path.join(save_dir, 'costs_train.npy'))
    costs_ica = np.load(os.path.join(save_dir, 'costs_ica_train.npy'))
    costs_decorr = np.load(os.path.join(save_dir, 'costs_decorr_train.npy'))
    r = range(len(costs_train[1000:]))
    plt.figure()
    plt.plot(r, costs_train[1000:], label='total')
    plt.plot(r, costs_ica[1000:], label='ica')
    plt.plot(r, costs_decorr[1000:], label='decorr')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'costs.png'))
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_type', type=str, default='adaptive')
    parser.add_argument('-exp_num', type=int, default=-1)
    parser.add_argument('-min_mix_coeff', type=float, default=0.1)
    parser.add_argument('-min_Mlines_angle', type=float, default=np.pi/8.)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-training_batches', nargs='+', type=int,
                        default=[20000])
    parser.add_argument('-learning_rates', nargs='+', type=float,
                        default=[5e-5])
    parser.add_argument('-layer_sizes', nargs='+', type=int,
                        default=[32, 32, 32, 32, 32, 32, 32, 32])
    parser.add_argument('-alpha_decorr', type=float, default=0.001)
    parser.add_argument('-use_backconnects', type=str2bool, 
                        default=str2bool('False'))
    args = parser.parse_args()
    train(args)

def train(args):
    print(args)

    ###### prepare experiment ######
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # NO: see note in exp_name definition. 
    exp_name = get_exp_name(
        exp_type=args.exp_type,
        exp_num=args.exp_num,
        min_mix_coeff=args.min_mix_coeff,
        min_Mlines_angle=args.min_Mlines_angle,
        batch_size=args.batch_size,
        training_batches=args.training_batches,
        learning_rates=args.learning_rates,
        layer_sizes=args.layer_sizes,
        alpha_decorr=args.alpha_decorr,
        use_backconnects=args.use_backconnects)

    exp_dir = os.path.join(results_dir, exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    comment_text = ("Mixing coeffs drawn from [-1., -min_coeff]U[min_coeff, 1]" + '\n' +
                    "Min mix coeff: " + str(args.min_mix_coeff) + '\n' +
                    "Min angle between the lines of M: " + '\n' +
                    "Training batches: " + str(args.training_batches) + '\n' + 
                    "Batch size: " + str(args.batch_size) + '\n' + 
                    "Learning rates: " + str(args.learning_rates) + '\n' + 
                    "Layer sizes: " + str(args.layer_sizes) + '\n' + 
                    "Alpha_decorr: " + str(args.alpha_decorr) + '\n')
    if args.exp_type == "adaptive":
        comment_text = (comment_text + 
                        "Use backconnects: " + str(args.use_backconnects) + '\n')

    write_to_comment_file(
        os.path.join(exp_dir, "comments.txt"), comment_text)

    ###### define model ######
    print('DEFINING MODEL...')

    if args.exp_type == "adaptive":
        modelICA = MLAdaptiveSF_ICA(
            in_size=in_size,
            batch_size=args.batch_size,
            n_chan_in=2,
            alpha_decorr=args.alpha_decorr,
            layer_sizes=args.layer_sizes,
            activations_conv11=lrelu,
            activations_to_W=lrelu,
            activations_to_out=None,
            use_backconnects=args.use_backconnects,
            use_var_feat_only_net=False)
    else:
        modelICA = MLSF_ICA(
            in_size=512,
            batch_size=args.batch_size,
            n_chan_in=2,
            alpha_decorr=args.alpha_decorr,
            layer_sizes=args.layer_sizes,
            activations=lrelu)

    modelICA.init()

    # load previously saved model if there is one
    if tf.train.get_checkpoint_state(exp_dir) is not None:
        modelICA.load_model(exp_dir)
        # also load sequence of previous costs
        costs_train = np.load(
            os.path.join(exp_dir, 'costs_train.npy')).tolist()
        costs_ica_train = np.load(
            os.path.join(exp_dir, 'costs_ica_train.npy')).tolist()
        costs_decorr_train = np.load(
            os.path.join(exp_dir, 'costs_decorr_train.npy')).tolist()
        #costs_valid = np.load(os.path.join(exp_dir, 'costs_valid.npy'))
        #costs_ica_valid
        #costs_decorr_valid
    else:
        costs_train = []
        costs_ica_train = []
        costs_decorr_train = []
        #costs_valid
        #costs_ica_valid
        #costs_decorr_valid

    ########## TRAIN #######################################
    print('TRAINING...')

    try:

        it_train = ica_poc_data.sine_sawtooth_iterator_fixedperiods(
            batch_size=args.batch_size,
            in_size=in_size,
            sine_period=sine_period,
            sawtooth_period=sawtooth_period,
            min_mix_coeff=args.min_mix_coeff,
            min_Mlines_angle=args.min_Mlines_angle)

        # it_valid = ica_poc_data.sine_sawtooth_iterator_fixedperiods(
        #     batch_size=args.batch_size,
        #     in_size=in_size,
        #     sine_period=sine_period_valid,
        #     sawtooth_period=sawtooth_period_valid,
        #     min_mix_coeff=args.min_mix_coeff,
        #     min_Mlines_angle=args.min_Mlines_angle)

        # Training loop
        for b in range(np.sum(args.training_batches)):
            if b > 0 and b % 1000 == 0:
                print(str(100*b/np.sum(args.training_batches)) + \
                      ' percent done...')
                print("average train cost_total over the last 1000 evals: ",
                      np.mean(costs_train[-1000:]))
                print("average train cost_ica over the last 1000 evals: ",
                      np.mean(costs_ica_train[-1000:]))
                print("average train cost_decorr over the last 1000 evals: ",
                      np.mean(costs_decorr_train[-1000:]))
                print("current learning rate: ", lr)
                sys.stdout.flush()

            if b > 0 and b % 10000 == 0:
                modelICA.save_model(
                    os.path.join(exp_dir, 'model'))

            # For variable learning rate:
            c01 = b > np.cumsum(args.training_batches)
            idx_lr = np.where(c01==1)[0]
            if len(idx_lr) == 0:
                idx_lr = 0
            else:
                idx_lr = idx_lr[-1] + 1
            lr = args.learning_rates[idx_lr]

            # train 
            examples = next(it_train)
            numeric_in = {
                'in_X': examples,
                'lr': lr,
            }
            cost_value, cost_ica_value, cost_decorr_value = \
            modelICA.train_model(numeric_in)

            costs_train += [cost_value]
            costs_ica_train += [cost_ica_value]
            costs_decorr_train += [cost_decorr_value]

            if b % 5000 == 0:
                np.save(
                    os.path.join(exp_dir, 'costs_train.npy'), 
                    np.array([*costs_train]))
                np.save(
                    os.path.join(exp_dir, 'costs_ica_train.npy'), 
                    np.array([*costs_ica_train]))
                np.save(
                    os.path.join(exp_dir, 'costs_decorr_train.npy'), 
                    np.array([*costs_decorr_train]))

        # also save after training
        np.save(os.path.join(exp_dir, 'costs_train.npy'), 
                np.array([*costs_train]))
        np.save(
            os.path.join(exp_dir, 'costs_ica_train.npy'), 
            np.array([*costs_ica_train]))
        np.save(
            os.path.join(exp_dir, 'costs_decorr_train.npy'), 
            np.array([*costs_decorr_train]))

        modelICA.save_model(
                    os.path.join(exp_dir, 'model'))

    except KeyboardInterrupt:
        print(' !!!!!!!! TRAINING INTERRUPTED !!!!!!!!')

    it_train = ica_poc_data.sine_sawtooth_iterator_fixedperiods(
        batch_size=args.batch_size,
        in_size=in_size,
        sine_period=sine_period,
        sawtooth_period=sawtooth_period,
        min_mix_coeff=args.min_mix_coeff,
        min_Mlines_angle=args.min_Mlines_angle)

    print('USING MODEL TO UNMIX SOME MIXED SAMPLES...')
    examples = next(it_train)
    numeric_in = {'in_X': examples}
    transformed_signals = modelICA.use_model(numeric_in)

    fig_save_dir = os.path.join(
        exp_dir, 'ica_demo_results', 'samples')
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    print('saving example unmixed signals to: ', fig_save_dir)
    for idx in range(10):
        ica_poc_data.plot_example_2ch(
            examples[idx], 
            os.path.join(fig_save_dir, 'mixed_example_%d.png' %idx))
        ica_poc_data.plot_example_2ch(
            transformed_signals[idx],
            os.path.join(fig_save_dir, 'unmixed_example_%d.png' %idx))

    # See values of alphas to see whether backconnects are 
    # effectively used or not:
    # (Put this in model we need to use it more often)
    if args.use_backconnects:
        alpha_vars = [v for v in tf.global_variables() 
                      if 'alpha' in v.name
                      and not 'Adam' in v.name]
        alpha_vars_names = [v.name for v in tf.global_variables()
                            if 'alpha' in v.name
                            and not 'Adam' in v.name]
        print(alpha_vars_names)
        alpha_vals = [modelICA.session.run(v) for v in alpha_vars]
        print(alpha_vals)

    modelICA.close()
    tf.reset_default_graph()

    plot_train_costs(save_dir=exp_dir)

    if send_txtmsg_when_done:
        textmsg.send_trainingdone_notif(exp_name)
    print('######## DONE. #########')

if __name__ == '__main__':
    main()






