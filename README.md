# TransformZero

## Deep Reinforcement Learning Continuous Integration

A major component of this major is the continuous integration via GitHub actions which allow for
us to confirm algorithm convergence in known environments. This is important to confirm that the code
for training models in new environments does work in known environments, along with other tests
to ensure that it should run properly in new environments (given proper setup for the new environment).
It also demonstrates a proof of concept for Deep RL Algorithms and their models for MLOps.

## The Algorithm

Tree-based search, similar to MuZero with time series transformer as transition model

The general idea for the algorithm is as follows:

1. Train a DQN exactly the same way it is down now
2. Collect transition samples from the training process, including initial obs/state, action, reward, terminated, next obs/state
3. Fit a time series transformer, taking the initial obs/state and action, and outputting the next obs/state time step
4. Once done, fine-tune a reward model and a terminated model, which takes obs/state and action, and outputs reward/terminated on transition

Once you have your DQN, Transition, Reward, and Terminated models, the algorithm for action selection goes as follows:

1. Initialize n threads, desired depth for threads, etc.
2. The Q-Values determine the proportion of threads to send each action branch, with dirichlet noise for exploration
3. The rewards are accumulated for each trajectory (from Reward model), but also being multiplied by how likely the states before were terminal (from Terminal model)
4. Action is based on trajectory that had best reward

For training, Q-Value targets are found the same as always, and the rest of the models continue to add data to the buffer
to improve the transition/reward/terminated models, which should help improve action search as training progresses

This idea comes in part from the original Deep Q Learning paper, where the authors stacked 
the previous 4 frames (CHECK) to understand what direction things were moving. A time series 
transformer would also have access to that information, as well as how past transitions
may influence future ones, even though we expect the transformer to really pay the most
attention to the last few time steps.

Additionally, it seems like a good idea because for other models we may want to have such as one for 
a reward can be fine-tuned from the transition model, since the input is the same we are just trying to 
predict something different, hence we can simply just change the final linear output layer, and get the improvement
of having good initialized weights in the model. This should significantly reduce training time,
and improve results in general.