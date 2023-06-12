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


class RandomAgentTests(unittest.TestCase):
    def setUp(self):
        set_seed(3407)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_converge_dummy(self):
        # Ensure minGPT-TS converges on dummy timeseries/state-action data

        # Create data
        train_dataset = FloatDataset('train')
        val_dataset = FloatDataset('val')
        test_dataset = FloatDataset('test')

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
        train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
        train_config.max_iters = 5000
        train_config.num_workers = 0
        train_config.val_interval = 100
        train_config.patience = 20
        trainer = Trainer(train_config, model, train_dataset, val_dataset)

        # Run training on model with train and validation data
        trainer.run()

        model.eval()

        # run a lot of examples from both train and test through the model and verify the output correctness
        with torch.no_grad():
            train_score = eval_split(model, trainer, train_dataset)
            test_score = eval_split(model, trainer, test_dataset)

        print(f"dummy: train score: {train_score}, test score: {test_score}")
        self.assertTrue(train_score < 0.001)
        self.assertTrue(test_score < 0.001)

    def test_converge_lunar(self):
        # create lunar environment
        env = gym.make("LunarLander-v2")
        obs, info = env.reset()

        # generate train, val, and test data by just taking random steps in lunar lander v2
        train_lunar_data_arr = get_random_episode_transitions(env, 100, 10, self.device)
        val_lunar_data_arr = get_random_episode_transitions(env, 100, 10, self.device)
        test_lunar_data_arr = get_random_episode_transitions(env, 100, 10, self.device)
        # format them into "Dataset" objects like Torch likes
        train_lunar_dataset = LunarDataset(train_lunar_data_arr)
        val_lunar_dataset = LunarDataset(val_lunar_data_arr)
        test_lunar_dataset = LunarDataset(test_lunar_data_arr)

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

        # run a lot of examples from both train and test through the model and verify the output correctness
        with torch.no_grad():
            train_score = eval_split(model, trainer, train_lunar_dataset)
            val_score = eval_split(model, trainer, val_lunar_dataset)
            test_score = eval_split(model, trainer, test_lunar_dataset)

        print(f"lunar: train score: {train_score}, val score: {val_score}, test score: {test_score}")
        self.assertTrue(train_score < 0.001)
        self.assertTrue(val_score < 0.005)  # sometimes it's noisy, this is still very good
        self.assertTrue(test_score < 0.005)

        # and to check on an individual example
        x, y = test_lunar_dataset[random.randrange(0, len(test_lunar_dataset) - 1)]
        single_output, _ = model(x.unsqueeze(0))
        error_diff = F.huber_loss(single_output[0], y).item()
        print("error diff for single value for lunar: ", error_diff)
        self.assertTrue(error_diff < 0.0075)  # again, we might get a weird sample, find your threshold


if __name__ == "__main__":
    unittest.main()

