import csv
import itertools
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import Model

from config import Config
from data.dataset import Dataset
from metrics.base import EmptyMetric
from optimization.AdaBelief import AdaBeliefOptimizer
from registry.registry import ModelRegistry, DatasetRegistry
from utils.measure import Timer
from utils.parameters_log import HP_TRAINABLE_PARAMS, HP_TASK
from utils.sat import is_graph_sat, build_dimacs_file, walksat
from utils.visualization import create_cactus_data


def main():
    optimizer = AdaBeliefOptimizer(Config.learning_rate, beta_1=0.6, clip_gradients=True)

    model = ModelRegistry().resolve(Config.model)(optimizer=optimizer)
    dataset = DatasetRegistry().resolve(Config.task)(data_dir=Config.data_dir,
                                                     force_data_gen=Config.force_data_gen)

    ckpt, manager = prepare_checkpoints(model, optimizer)

    if Config.train:
        train(dataset, model, ckpt, manager)

    if Config.evaluate:
        test_metrics = evaluate_metrics(dataset, dataset.test_data(), model)
        for metric in test_metrics:
            metric.log_in_stdout()

    if Config.evaluate_round_gen:
        evaluate_round_generalization(dataset, optimizer)

    if Config.evaluate_variable_gen:
        evaluate_variable_generalization(model)

    if Config.make_cactus:
        make_cactus(model, dataset)

    if Config.make_scatter:
        make_scatter(model)


def make_scatter(model: Model):
    model_solved = []
    solver_solved = []
    model_time = []
    solver_time = []

    dataset = DatasetRegistry().resolve(Config.task)(data_dir=Config.data_dir,
                                                     force_data_gen=Config.force_data_gen,
                                                     batch_of_single=True)

    for step, step_data in enumerate(dataset.test_data()):
        model_input = dataset.filter_model_inputs(step_data)
        start = time.time()
        output = model.predict_step(**model_input)
        elapsed_time = time.time() - start

        if step >= 2:
            pred = tf.expand_dims(output["prediction"], axis=-1)
            is_sat = is_graph_sat(pred, step_data["adjacency_matrix"], step_data["clauses_graph_adj"]).numpy()
            is_sat = tf.squeeze(is_sat, axis=-1)
            solved_batch = [int(x) for x in is_sat]
            model_solved += solved_batch
            model_time += [elapsed_time / len(solved_batch)] * len(solved_batch)

            clauses = [x.numpy() for x in step_data["normal_clauses"]]
            vars_in_graph = step_data["variables_in_graph"].numpy()

            for iclauses, n_vars in zip(clauses, vars_in_graph):
                dimacs = build_dimacs_file(iclauses, n_vars)
                sat, solution, time_elapsed = walksat(dimacs)
                solver_solved.append(int(sat))
                solver_time.append(time_elapsed)

    rows = [[m_s, m_t, s_s, s_t] for m_s, m_t, s_s, s_t in zip(model_solved, model_time, solver_solved, solver_time)]

    model_name = model.__class__.__name__.lower()
    with open(model_name + "_vs_walksat_scatter.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def make_cactus(model: Model, dataset):
    solved = []
    var_count = []
    time_used = []

    for step, step_data in enumerate(dataset.test_data()):
        model_input = dataset.filter_model_inputs(step_data)
        start = time.time()
        output = model.predict_step(**model_input)
        elapsed_time = time.time() - start

        if step >= 10:
            pred = tf.expand_dims(output["prediction"], axis=-1)
            is_sat = is_graph_sat(pred, step_data["adjacency_matrix"], step_data["clauses_graph_adj"]).numpy()
            is_sat = tf.squeeze(is_sat, axis=-1)
            solved_batch = [int(x) for x in is_sat]
            solved += solved_batch
            var_count += step_data["variables_in_graph"].numpy().tolist()
            time_used += [elapsed_time / len(solved_batch)] * len(solved_batch)

    rows = create_cactus_data(solved, time_used, var_count)

    model_name = model.__class__.__name__.lower()
    with open(model_name + "_cactus.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def evaluate_variable_generalization(model):
    results_file = get_valid_file("gen_variables_size_result.txt")

    lower_limit = 10
    upper_limit = 100
    step = 10

    for var_count in range(lower_limit, upper_limit, step):
        print(f"Generating dataset with var_count={var_count}")
        dataset = DatasetRegistry().resolve(Config.task)(data_dir=Config.data_dir,
                                                         force_data_gen=Config.force_data_gen,
                                                         max_batch_size=20000,
                                                         min_vars=var_count,
                                                         max_vars=var_count)

        test_metrics = evaluate_metrics(dataset, dataset.test_data(), model)
        prepend_line = f"Results for dataset with var_count={var_count}:"
        for metric in test_metrics:
            metric.log_in_file(str(results_file), prepend_str=prepend_line)


def get_valid_file(file: str):
    train_dir = Path(Config.train_dir)
    results_file = train_dir / file
    if not train_dir.exists():
        train_dir.mkdir(parents=True)
    return results_file


def evaluate_round_generalization(dataset, optimizer):
    results_file = get_valid_file("gen_steps_result.txt")

    test_data = dataset.test_data()
    for test_rounds in [2 ** r for r in range(4, 13, 1)]:
        model = ModelRegistry().resolve(Config.model)(optimizer=optimizer, test_rounds=test_rounds)
        print(f"Evaluating model with test_rounds={test_rounds}")
        _ = prepare_checkpoints(model, optimizer)

        start_time = time.time()
        test_metrics = evaluate_metrics(dataset, test_data, model)
        elapsed_time = time.time() - start_time

        message = f"Results for model with test_rounds={test_rounds} and elapsed_time={elapsed_time / dataset.test_size :.3f}:"
        for metric in test_metrics:
            metric.log_in_file(str(results_file), prepend_str=message)


def train(dataset: Dataset, model: Model, ckpt, ckpt_manager):
    writer = tf.summary.create_file_writer(Config.train_dir)
    writer.set_as_default()

    mean_loss = tf.metrics.Mean()
    timer = Timer(start_now=True)
    validation_data = dataset.validation_data()
    train_data = dataset.train_data()

    for step_data in itertools.islice(train_data, Config.train_steps + 1 - int(ckpt.step)):
        tf.summary.experimental.set_step(ckpt.step)

        model_data = dataset.filter_model_inputs(step_data)

        model_output = model.train_step(**model_data)
        loss, gradients = model_output["loss"], model_output["gradients"]
        mean_loss.update_state(loss)

        if int(ckpt.step) % 100 == 0:
            loss_mean = mean_loss.result()
            with writer.as_default():
                tf.summary.scalar("loss", loss_mean, step=int(ckpt.step))

            print(f"{int(ckpt.step)}. step;\tloss: {loss_mean:.5f};\ttime: {timer.lap():.3f}s")
            mean_loss.reset_states()

            with tf.name_scope("variables"):
                with writer.as_default():
                    for var in model.trainable_variables:  # type: tf.Variable
                        tf.summary.histogram(var.name, var, step=int(ckpt.step))

        if int(ckpt.step) % 1000 == 0:
            n_eval_steps = 100
            if Config.task == 'euclidean_tsp' or Config.task == 'asymmetric_tsp':  # TODO: Make it similar to metrics
                n_eval_steps = 1
                iterator = itertools.islice(validation_data, 1)
                for visualization_step_data in iterator:
                    model_input = dataset.filter_model_inputs(visualization_step_data)
                    model.log_visualizations(**model_input)

            metrics = evaluate_metrics(dataset, validation_data, model, steps=n_eval_steps,
                                       initial=(int(ckpt.step) == 0))
            for metric in metrics:
                metric.log_in_tensorboard(reset_state=False, step=int(ckpt.step))
                metric.log_in_stdout(step=int(ckpt.step))

            hparams = model.get_config()
            hparams[HP_TASK] = dataset.__class__.__name__
            hparams[HP_TRAINABLE_PARAMS] = np.sum([np.prod(v.shape) for v in model.trainable_variables])
            hp.hparams(hparams)

        if int(ckpt.step) % 1000 == 0:
            save_path = ckpt_manager.save()
            print(f"Saved checkpoint for step {int(ckpt.step)}: {save_path}")

        if int(ckpt.step) % 100 == 0:
            writer.flush()

        ckpt.step.assign_add(1)


def prepare_checkpoints(model, optimizer):
    ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, Config.train_dir, max_to_keep=Config.ckpt_count)

    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print(f"Model restored from {manager.latest_checkpoint}!")
    else:
        print("Initializing new model!")

    return ckpt, manager


def evaluate_metrics(dataset: Dataset, data: tf.data.Dataset, model: Model, steps: int = None, initial=False) -> list:
    metrics = dataset.metrics(initial)
    iterator = itertools.islice(data, steps) if steps else data

    empty = True
    counter = 0
    for step_data in iterator:
        counter += 1
        model_input = dataset.filter_model_inputs(step_data)
        output = model.predict_step(**model_input)
        for metric in metrics:
            metric.update_state(output, step_data)
        empty = False

    return metrics if not empty else [EmptyMetric()]


if __name__ == '__main__':
    config = Config.parse_config()
    tf.config.run_functions_eagerly(Config.eager)

    if Config.restore:
        print(f"Restoring model from last checkpoint in '{Config.restore}'!")
        Config.train_dir = Config.restore
    else:
        current_date = time.strftime("%y_%m_%d_%T", time.gmtime(time.time()))
        label = "_" + Config.label if Config.label else ""
        Config.train_dir = Config.train_dir + "/" + Config.task + "_" + current_date + label

    main()
