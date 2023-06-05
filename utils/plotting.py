import torch
import matplotlib
import matplotlib.pyplot as plt


def plot_rewards(epoch_rewards: list, show_avg_of_last: int = 100, show_result: bool = False) -> None:
    plt.figure(1)
    durations_t = torch.tensor(epoch_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 (show avg last) episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, show_avg_of_last, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(show_avg_of_last - 1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
