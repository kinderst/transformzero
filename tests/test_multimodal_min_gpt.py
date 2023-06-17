import unittest

import random
import torch
from torch.nn import functional as F

import gymnasium as gym

from datasets.float_dataset import FloatDataset
from datasets.lunar_dataset import LunarDataset
from datasets.lunar_generator import get_random_episode_transitions, get_random_episode_masked
# from models.min_gpt import GPT
from models.min_gpt import MultimodalGPT
from utils.min_gpt_utils import set_seed, eval_split, mask_loss_neg_inf
from trainers.min_gpt_trainer import Trainer


class MultimodalMinGPTTests(unittest.TestCase):
    def loop_random_single_example_eval(self, model, dataset, threshold, name_string) -> None:
        single_passed = False
        single_results = []
        for i in range(10):
            x, y = dataset[random.randrange(0, len(dataset) - 1)]
            single_output, _ = model(x.unsqueeze(0))
            single_output, y = mask_loss_neg_inf(single_output[0], y)
            error_diff = F.huber_loss(single_output, y).item()
            single_results.append(error_diff)
            if error_diff < threshold:
                single_passed = True
                break
        print(f"MultimodalMinGPT {name_string} single results: {single_results}, actual values x: {x}, y: {y}, single output: {single_output}")
        self.assertTrue(single_passed)

    def setUp(self) -> None:
        set_seed(3407)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_converge_dummy(self, mask_all_but_last=False) -> None:
        # Ensure minGPT-TS converges on dummy timeseries/state-action data

        if mask_all_but_last:
            train_converge_score = 0.0005
            val_converge_score = 0.0005
            test_converge_score = 0.0005
        else:
            train_converge_score = 0.02
            val_converge_score = 0.02
            test_converge_score = 0.02

        # Create data
        train_dataset = FloatDataset('train', mask_all_but_last=mask_all_but_last)
        val_dataset = FloatDataset('val', mask_all_but_last=mask_all_but_last)

        # Instantiate minGPT-TS model
        model_config = MultimodalGPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        # model_config.vocab_size = train_dataset.get_vocab_size()
        model_config.in_dim = 3  # state_dim_1, state_dim_2, action_scalar
        model_config.out_dim = 2  # state_dim_1, state_dim_2
        model_config.modality_shapes = {
            "flatstate": 2,  # 2 dim simple state
            "embedaction": 4  # we usually would do action_space.n or num_actions here...
        }
        model_config.block_size = train_dataset.get_block_size()
        model = MultimodalGPT(model_config)

        # Instantiate minGPT trainer
        train_config = Trainer.get_default_config()
        train_config.learning_rate = 1e-4  # the model we're using is so small that we can go a bit faster
        train_config.max_iters = 7500
        train_config.num_workers = 0
        train_config.val_interval = 100
        train_config.patience = 20
        trainer = Trainer(train_config, model, train_dataset, val_dataset)

        trainer.run()

        # train eval, which is only ever done once
        with torch.no_grad():
            train_score = eval_split(model, self.device, train_dataset, use_mask=mask_all_but_last)
        print(f"MultimodalMinGPT dummy masked: {mask_all_but_last} train results: {train_score}")

        self.assertLessEqual(train_score, train_converge_score)

        val_passed = False
        val_results = []
        for i in range(3):
            with torch.no_grad():
                val_score = eval_split(model, self.device, val_dataset, use_mask=mask_all_but_last)
                val_results.append(val_score)
                if val_score < val_converge_score:
                    val_passed = True
                    break
                else:
                    val_dataset = FloatDataset('val', mask_all_but_last=mask_all_but_last)
        print(f"MultimodalMinGPT dummy masked: {mask_all_but_last} val results: {val_results}")
        self.assertTrue(val_passed)

        test_passed = False
        test_results = []
        for i in range(3):
            test_dataset = FloatDataset('test', mask_all_but_last=mask_all_but_last)
            # test eval
            with torch.no_grad():
                test_score = eval_split(model, self.device, test_dataset, use_mask=mask_all_but_last)
                test_results.append(test_score)
                if test_score < test_converge_score:
                    test_passed = True
                    break
        print(f"MultimodalMinGPT dummy masked: {mask_all_but_last} test results: {test_results}")
        self.assertTrue(test_passed)

        # and to check on an individual example
        single_dataset = FloatDataset('test', mask_all_but_last=mask_all_but_last)
        self.loop_random_single_example_eval(model, single_dataset, test_converge_score, f'dummy masked: {mask_all_but_last}')

    def test_converge_dummy_mask(self):
        self.test_converge_dummy(mask_all_but_last=True)

    def test_converge_lunar(self, mask_all_but_last=False) -> None:
        """
        :param mask_all_but_last: mask all but last except first transitions where its last but len < seq_len
        """

        if mask_all_but_last:
            train_converge_score = 0.003
            val_converge_score = 0.005
            test_converge_score = 0.005
        else:
            train_converge_score = 0.001
            val_converge_score = 0.001
            test_converge_score = 0.001

        # create lunar environment
        env = gym.make("LunarLander-v2")
        obs, info = env.reset()

        # generate train, val, and test data by just taking random steps in lunar lander v2
        if mask_all_but_last:
            train_lunar_data_arr = get_random_episode_masked(env, 200, 10, self.device)
            val_lunar_data_arr = get_random_episode_masked(env, 200, 10, self.device)
        else:
            train_lunar_data_arr = get_random_episode_transitions(env, 200, 10, self.device, use_one_hot=False)
            val_lunar_data_arr = get_random_episode_transitions(env, 200, 10, self.device, use_one_hot=False)

        # format them into "Dataset" objects like Torch likes
        train_lunar_dataset = LunarDataset(train_lunar_data_arr)
        val_lunar_dataset = LunarDataset(val_lunar_data_arr)

        model_config = MultimodalGPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        # model_config.vocab_size = train_dataset.get_vocab_size()  # no vocab size, n_dim in and out...
        model_config.in_dim = env.action_space.n + len(obs)  # Lunar lander state space is 8, and action space is 4
        model_config.out_dim = len(obs)  # we are predicting next state conditioned on previous time steps with action
        model_config.modality_shapes = {
            "flatstate": len(obs),  # 2 dim simple state
            "embedaction": env.action_space.n
        }
        model_config.block_size = train_lunar_dataset.get_block_size()
        model = MultimodalGPT(model_config)

        train_config = Trainer.get_default_config()
        train_config.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster
        train_config.max_iters = 7500
        train_config.num_workers = 0
        train_config.val_interval = 100
        train_config.patience = 25
        trainer = Trainer(train_config, model, train_lunar_dataset, val_lunar_dataset)

        trainer.run()

        model.eval()

        # train eval, which is only ever done once
        with torch.no_grad():
            train_score = eval_split(model, self.device, train_lunar_dataset, use_mask=mask_all_but_last)
        print(f"MultimodalMinGPT lunar masked: {mask_all_but_last} train results: {train_score}")
        self.assertLessEqual(train_score, train_converge_score)

        val_passed = False
        val_results = []
        for i in range(3):
            with torch.no_grad():
                val_score = eval_split(model, self.device, val_lunar_dataset, use_mask=mask_all_but_last)
                val_results.append(val_score)
                if val_score < val_converge_score:
                    val_passed = True
                    break
                else:
                    if mask_all_but_last:
                        val_lunar_data_arr = get_random_episode_masked(env, 200, 10, self.device)
                    else:
                        val_lunar_data_arr = get_random_episode_transitions(env, 200, 10, self.device, use_one_hot=False)
                    val_lunar_dataset = LunarDataset(val_lunar_data_arr)
        print(f"MultimodalMinGPT lunar masked: {mask_all_but_last} val results: {val_results}")
        self.assertTrue(val_passed)

        test_passed = False
        test_results = []
        for i in range(3):
            if mask_all_but_last:
                test_lunar_data_arr = get_random_episode_masked(env, 200, 10, self.device)
            else:
                test_lunar_data_arr = get_random_episode_transitions(env, 200, 10, self.device, use_one_hot=False)
            test_lunar_dataset = LunarDataset(test_lunar_data_arr)
            # test eval
            with torch.no_grad():
                test_score = eval_split(model, self.device, test_lunar_dataset, use_mask=mask_all_but_last)
                test_results.append(test_score)
                if test_score < test_converge_score:
                    test_passed = True
                    break
        print(f"MultimodalMinGPT lunar test masked: {mask_all_but_last} results: {test_results}")
        self.assertTrue(test_passed)

        # and to check on an individual example
        if mask_all_but_last:
            single_lunar_data_arr = get_random_episode_masked(env, 2, 10, self.device)
        else:
            single_lunar_data_arr = get_random_episode_transitions(env, 2, 10, self.device, use_one_hot=False)
        single_lunar_dataset = LunarDataset(single_lunar_data_arr)
        self.loop_random_single_example_eval(model, single_lunar_dataset, test_converge_score, f'lunar masked: {mask_all_but_last}')

    def test_converge_lunar_mask(self):
        self.test_converge_lunar(mask_all_but_last=True)


if __name__ == "__main__":
    unittest.main()
