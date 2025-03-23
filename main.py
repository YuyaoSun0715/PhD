import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

import argparse

from data_generator import DataGenerator
from maml import MAML

parser = argparse.ArgumentParser(description="MAML Training Configuration")

## Dataset/method options
parser.add_argument('--datasource', type=str, default='sinusoid', choices=['sinusoid', 'omniglot', 'miniimagenet'])
parser.add_argument('--num_classes', type=int, default=5, help='Number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--baseline', type=str, default=None, choices=['oracle', None])

## Training options
parser.add_argument('--pretrain_iterations', type=int, default=0)
parser.add_argument('--metatrain_iterations', type=int, default=15000)  # 15k for omniglot, 50k for sinusoid
parser.add_argument('--meta_batch_size', type=int, default=25)
parser.add_argument('--meta_lr', type=float, default=0.001)
parser.add_argument('--update_batch_size', type=int, default=5, help='Number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_lr', type=float, default=1e-3, help='Step size alpha for inner gradient update.')
parser.add_argument('--num_updates', type=int, default=1, help='Number of inner gradient updates during training.')

## Model options
parser.add_argument('--norm', type=str, default='batch_norm', choices=['batch_norm', 'layer_norm', None])
parser.add_argument('--num_filters', type=int, default=64, help='Number of filters for conv nets.')
parser.add_argument('--conv', action='store_true', default=True, help='Whether or not to use a convolutional network.')
parser.add_argument('--max_pool', action='store_true', default=False, help='Whether or not to use max pooling rather than strided convolutions.')
parser.add_argument('--stop_grad', action='store_true', default=False, help='If True, do not use second derivatives in meta-optimization (for speed).')

## Logging, saving, and testing options
parser.add_argument('--log', action='store_true', default=True, help='If false, do not log summaries, for debugging code.')
parser.add_argument('--logdir', type=str, default='/tmp/data', help='Directory for summaries and checkpoints.')
parser.add_argument('--resume', action='store_true', default=True, help='Resume training if there is a model available.')
parser.add_argument('--train', action='store_true', default=True, help='True to train, False to test.')
parser.add_argument('--test_iter', type=int, default=-1, help='Iteration to load model (-1 for latest model).')
parser.add_argument('--test_set', action='store_true', default=False, help='Set to true to test on the test set, False for the validation set.')
parser.add_argument('--train_update_batch_size', type=int, default=-1, help='Number of examples used for gradient update during training.')
parser.add_argument('--train_update_lr', type=float, default=-1, help='Value of inner gradient step during training.')

args = parser.parse_args()

def train(model, optimizer, exp_string, data_generator, resume_itr=0, config=None):
    if config is None:
        raise ValueError("Config dictionary is required.")

    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 1000 if config["datasource"] == "sinusoid" else 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

    # TensorBoard Logger
    logdir = config["logdir"]
    if config["log"]:
        writer = SummaryWriter(os.path.join(logdir, exp_string))

    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes  # for classification, 1 otherwise

    for itr in range(resume_itr, config["pretrain_iterations"] + config["metatrain_iterations"]):
        if hasattr(data_generator, "generate"):
            batch_x, batch_y, amp, phase = data_generator.generate()

            if config["baseline"] == "oracle":
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], axis=2)
                for i in range(config["meta_batch_size"]):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            inputa = torch.tensor(batch_x[:, :num_classes * config["update_batch_size"], :], dtype=torch.float32)
            labela = torch.tensor(batch_y[:, :num_classes * config["update_batch_size"], :], dtype=torch.float32)
            inputb = torch.tensor(batch_x[:, num_classes * config["update_batch_size"]:, :], dtype=torch.float32)  # Used for testing
            labelb = torch.tensor(batch_y[:, num_classes * config["update_batch_size"]:, :], dtype=torch.float32)

        # Forward pass
        if itr < config["pretrain_iterations"]:
            loss = model.pretrain_loss(inputa, labela)  # Assume pretrain_loss function exists
        else:
            loss = model.meta_train_loss(inputa, labela, inputb, labelb)  # Assume meta_train_loss exists

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging & printing losses
        if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
            prelosses.append(loss.item())

        if itr % SUMMARY_INTERVAL == 0:
            if config["log"]:
                writer.add_scalar("Loss/Train", loss.item(), itr)
            postlosses.append(loss.item())

        if itr != 0 and itr % PRINT_INTERVAL == 0:
            print_str = f"{'Pretrain Iteration' if itr < config['pretrain_iterations'] else 'Iteration'} {itr}: {np.mean(prelosses)}, {np.mean(postlosses)}"
            print(print_str)
            prelosses, postlosses = [], []

        # Save model
        if itr != 0 and itr % SAVE_INTERVAL == 0:
            model_save_path = os.path.join(logdir, exp_string, f"model_{itr}.pth")
            torch.save(model.state_dict(), model_save_path)

        # Validation step (only for non-sinusoid datasets)
        if itr != 0 and itr % TEST_PRINT_INTERVAL == 0 and config["datasource"] != "sinusoid":
            if hasattr(data_generator, "generate"):
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                inputa = torch.tensor(batch_x[:, :num_classes * config["update_batch_size"], :], dtype=torch.float32)
                inputb = torch.tensor(batch_x[:, num_classes * config["update_batch_size"]:, :], dtype=torch.float32)
                labela = torch.tensor(batch_y[:, :num_classes * config["update_batch_size"], :], dtype=torch.float32)
                labelb = torch.tensor(batch_y[:, num_classes * config["update_batch_size"]:, :], dtype=torch.float32)
            else:
                inputa, labela, inputb, labelb = None, None, None, None  # Handle non-data-generator case

            with torch.no_grad():
                if model.classification:
                    acc1, acc2 = model.evaluate(inputa, labela, inputb, labelb)  # Assume evaluate function exists
                    print(f"Validation results: {acc1}, {acc2}")
                else:
                    loss1, loss2 = model.evaluate(inputa, labela, inputb, labelb)
                    print(f"Validation results: {loss1}, {loss2}")

    # Final save
    final_model_path = os.path.join(logdir, exp_string, f"model_{itr}.pth")
    torch.save(model.state_dict(), final_model_path)

def test(model, exp_string, data_generator, test_num_updates=None, config=None):
    if config is None:
        raise ValueError("Config dictionary is required.")

    num_classes = data_generator.num_classes  # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    for _ in range(config["num_test_points"]):
        if hasattr(data_generator, "generate"):
            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            if config["baseline"] == "oracle":  # NOTE - this flag is specific to sinusoid
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], axis=2)
                batch_x[0, :, 1] = amp[0]
                batch_x[0, :, 2] = phase[0]

            inputa = torch.tensor(batch_x[:, :num_classes * config["update_batch_size"], :], dtype=torch.float32)
            inputb = torch.tensor(batch_x[:, num_classes * config["update_batch_size"]:, :], dtype=torch.float32)
            labela = torch.tensor(batch_y[:, :num_classes * config["update_batch_size"], :], dtype=torch.float32)
            labelb = torch.tensor(batch_y[:, num_classes * config["update_batch_size"]:, :], dtype=torch.float32)

        with torch.no_grad():
            if model.classification:
                result = model.evaluate(inputa, labela, inputb, labelb)  # Assume evaluate function exists
            else:  # This is for sinusoid regression
                result = model.evaluate(inputa, labela, inputb, labelb)

        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, axis=0)
    stds = np.std(metaval_accuracies, axis=0)
    ci95 = 1.96 * stds / np.sqrt(config["num_test_points"])

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    # Save results
    logdir = config["logdir"]
    test_prefix = f"test_ubs{config['update_batch_size']}_stepsize{config['update_lr']}"

    out_pkl = os.path.join(logdir, exp_string, f"{test_prefix}.pkl")
    out_csv = os.path.join(logdir, exp_string, f"{test_prefix}.csv")

    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)

    np.savetxt(out_csv, [means, stds, ci95], delimiter=",", header=",".join([f"update{i}" for i in range(len(means))]))



class FLAGS:
    datasource = 'miniimagenet'  # Example default
    train = True
    meta_batch_size = 4
    update_batch_size = 5
    metatrain_iterations = 10000
    baseline = None
    num_classes = 5
    num_updates = 5
    update_lr = 0.01
    num_filters = 64
    max_pool = False
    stop_grad = False
    norm = 'batch_norm'
    resume = False
    logdir = './logs'
    test_iter = 0


def main():
    test_num_updates = 5 if FLAGS.train else 10

    if FLAGS.datasource == 'miniimagenet':
        test_num_updates = 1 if FLAGS.train else 10

    if not FLAGS.train:
        orig_meta_batch_size = FLAGS.meta_batch_size
        FLAGS.meta_batch_size = 1  # Always use meta batch size of 1 when testing

    if FLAGS.datasource == 'sinusoid':
        data_generator = DataGenerator(FLAGS.update_batch_size * 2, FLAGS.meta_batch_size)
    else:
        if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(1, FLAGS.meta_batch_size)
        else:
            num_val_examples = 15 if FLAGS.train else FLAGS.update_batch_size * 2
            data_generator = DataGenerator(FLAGS.update_batch_size + num_val_examples, FLAGS.meta_batch_size)

    dim_output = data_generator.dim_output
    dim_input = 3 if FLAGS.baseline == 'oracle' else data_generator.dim_input

    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.update_lr)

    exp_string = f"cls_{FLAGS.num_classes}.mbs_{FLAGS.meta_batch_size}.ubs_{FLAGS.update_batch_size}.numstep{FLAGS.num_updates}.updatelr{FLAGS.update_lr}"
    if FLAGS.num_filters != 64:
        exp_string += f"hidden{FLAGS.num_filters}"
    if FLAGS.max_pool:
        exp_string += "maxpool"
    if FLAGS.stop_grad:
        exp_string += "stopgrad"
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm:
        exp_string += FLAGS.norm

    resume_itr = 0
    model_file = None

    if FLAGS.resume or not FLAGS.train:
        model_path = os.path.join(FLAGS.logdir, exp_string)
        if os.path.exists(model_path):
            model_file = sorted(os.listdir(model_path))[-1] if FLAGS.test_iter == 0 else f"model{FLAGS.test_iter}"
            model_file = os.path.join(model_path, model_file)
            checkpoint = torch.load(model_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resume_itr = checkpoint.get('iteration', 0)
            print(f"Restored model from {model_file}")

    if FLAGS.train:
        train(model, optimizer, exp_string, data_generator, resume_itr)
    else:
        test(model, exp_string, data_generator, test_num_updates)


if __name__ == "__main__":
    main()
