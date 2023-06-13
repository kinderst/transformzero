import unittest

import random
import torch
from torch.nn import functional as F

import gymnasium as gym

from datasets.float_dataset import FloatDataset
from datasets.lunar_dataset import LunarDataset
from datasets.lunar_generator import get_random_episode_transitions
from models.min_gpt import GPT
from utils.min_gpt_utils import set_seed, eval_split
from trainers.min_gpt_trainer import Trainer


class MinGPTTests(unittest.TestCase):
    def loop_random_single_example_eval(self, model, dataset, threshold, name_string) -> None:
        single_passed = False
        single_results = []
        for i in range(10):
            x, y = dataset[random.randrange(0, len(dataset) - 1)]
            single_output, _ = model(x.unsqueeze(0))
            error_diff = F.huber_loss(single_output[0], y).item()
            single_results.append(error_diff)
            if error_diff < threshold:
                single_passed = True
                break
        print(f"minGPT {name_string} single results: {single_results}")
        self.assertTrue(single_passed)

    def setUp(self) -> None:
        set_seed(3407)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_converge_dummy(self) -> None:
        # Ensure minGPT-TS converges on dummy timeseries/state-action data

        # Create data
        train_dataset = FloatDataset('train')
        val_dataset = FloatDataset('val')

        # Instantiate minGPT-TS model
        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        # model_config.vocab_size = train_dataset.get_vocab_size()
        model_config.in_dim = 3  # state_dim_1, state_dim_2, action_scalar
        model_config.out_dim = 2  # state_dim_1, state_dim_2
        model_config.block_size = train_dataset.get_block_size()
        model = GPT(model_config)

        # Instantiate minGPT trainer
        train_config = Trainer.get_default_config()
        train_config.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster
        train_config.max_iters = 5000
        train_config.num_workers = 0
        train_config.val_interval = 100
        train_config.patience = 20
        trainer = Trainer(train_config, model, train_dataset, val_dataset)

        trainer.run()

        # train eval, which is only ever done once
        with torch.no_grad():
            train_score = eval_split(model, self.device, train_dataset)
        print(f"minGPT dummy train results: {train_score}")
        self.assertLessEqual(train_score, 0.0005)

        val_passed = False
        val_results = []
        for i in range(3):
            with torch.no_grad():
                val_score = eval_split(model, self.device, val_dataset)
                val_results.append(val_score)
                if val_score < 0.0005:
                    val_passed = True
                    break
                else:
                    val_dataset = FloatDataset('val')  # do it down here, since already created one first time
        print(f"minGPT dummy val results: {val_results}")
        self.assertTrue(val_passed)

        test_passed = False
        test_results = []
        for i in range(3):
            test_dataset = FloatDataset('test')
            # test eval
            with torch.no_grad():
                test_score = eval_split(model, self.device, test_dataset)
                test_results.append(test_score)
                if test_score < 0.0005:
                    test_passed = True
                    break
        print(f"minGPT dummy test results: {test_results}")
        self.assertTrue(test_passed)

        # and to check on an individual example
        single_dataset = FloatDataset('test')
        self.loop_random_single_example_eval(model, single_dataset, 0.0005, 'dummy')

    def test_converge_lunar(self) -> None:
        # create lunar environment
        env = gym.make("LunarLander-v2")
        obs, info = env.reset()

        # generate train, val, and test data by just taking random steps in lunar lander v2
        train_lunar_data_arr = get_random_episode_transitions(env, 100, 10, self.device)
        val_lunar_data_arr = get_random_episode_transitions(env, 100, 10, self.device)

        # format them into "Dataset" objects like Torch likes
        train_lunar_dataset = LunarDataset(train_lunar_data_arr)
        val_lunar_dataset = LunarDataset(val_lunar_data_arr)

        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        # model_config.vocab_size = train_dataset.get_vocab_size()  # no vocab size, n_dim in and out...
        model_config.in_dim = env.action_space.n + len(obs)  # Lunar lander state space is 8, and action space is 4
        model_config.out_dim = len(obs)  # we are predicting next state conditioned on previous time steps with action
        model_config.block_size = train_lunar_dataset.get_block_size()
        model = GPT(model_config)

        train_config = Trainer.get_default_config()
        train_config.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster
        train_config.max_iters = 10000
        train_config.num_workers = 0
        train_config.val_interval = 100
        train_config.patience = 20
        trainer = Trainer(train_config, model, train_lunar_dataset, val_lunar_dataset)

        trainer.run()

        model.eval()

        # train eval, which is only ever done once
        with torch.no_grad():
            train_score = eval_split(model, self.device, train_lunar_dataset)
        print(f"minGPT lunar train results: {train_score}")
        self.assertLessEqual(train_score, 0.001)

        val_passed = False
        val_results = []
        for i in range(3):
            with torch.no_grad():
                val_score = eval_split(model, self.device, val_lunar_dataset)
                val_results.append(val_score)
                if val_score < 0.003:
                    val_passed = True
                    break
                else:
                    val_lunar_data_arr = get_random_episode_transitions(env, 100, 10, self.device)
                    val_lunar_dataset = LunarDataset(val_lunar_data_arr)
        print(f"minGPT lunar val results: {val_results}")
        self.assertTrue(val_passed)

        test_passed = False
        test_results = []
        for i in range(3):
            test_lunar_data_arr = get_random_episode_transitions(env, 100, 10, self.device)
            test_lunar_dataset = LunarDataset(test_lunar_data_arr)
            # test eval
            with torch.no_grad():
                test_score = eval_split(model, self.device, test_lunar_dataset)
                test_results.append(test_score)
                if test_score < 0.003:
                    test_passed = True
                    break
        print(f"minGPT lunar test results: {val_results}")
        self.assertTrue(test_passed)

        # and to check on an individual example
        single_lunar_data_arr = get_random_episode_transitions(env, 10, 10, self.device)
        single_lunar_dataset = LunarDataset(single_lunar_data_arr)
        self.loop_random_single_example_eval(model, single_lunar_dataset, 0.003, 'lunar')


if __name__ == "__main__":
    unittest.main()

