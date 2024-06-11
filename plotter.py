import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def read_csv_files(path_pattern):
    """
    Reads all CSV files matching the given path pattern.
    """
    all_files = glob.glob(path_pattern)
    df_list = [pd.read_csv(file) for file in all_files]
    return pd.concat(df_list, ignore_index=True)

def plot_vps_across_training_games(df, agent_types):
    """
    Plots the mean VPs across training games for each agent type.
    """
    plt.figure(figsize=(10, 6))

    for agent in agent_types:
        agent_df = df[df['agent type'] == agent]
        mean_vps = agent_df.groupby('train games')['mean VPs'].mean()
        std_vps = agent_df.groupby('train games')['std VPs'].mean()

        plt.plot(mean_vps.index, mean_vps.values, label=f'{agent} (mean)')
        plt.fill_between(mean_vps.index, 
                         mean_vps.values - std_vps.values, 
                         mean_vps.values + std_vps.values, 
                         alpha=0.2)

    plt.xlabel('Training Games')
    plt.ylabel('Mean VPs')
    plt.title('Mean VPs across Training Games')
    plt.legend()
    plt.grid(True)

    agents_str = "-".join(agent_types)
    save_path = os.path.join('results/plots', f'{agents_str}_VPs.png')
    plt.savefig(save_path)

def plot_plot_quests_across_training_games(df, agent_types):
    """
    Plots the mean VPs across training games for each agent type.
    """
    plt.figure(figsize=(10, 6))

    for agent in agent_types:
        agent_df = df[df['agent type'] == agent]
        mean_vps = agent_df.groupby('train games')['mean VPs'].mean()
        std_mean_vps = agent_df.groupby('train games')['mean VPs'].std()

        plt.plot(mean_vps.index, mean_vps.values, label=f'{agent} (mean)')
        plt.fill_between(mean_vps.index, 
                         mean_vps.values - std_mean_vps.values, 
                         mean_vps.values + std_mean_vps.values, 
                         alpha=0.2)

    plt.xlabel('Training Games')
    plt.ylabel('Mean VPs')
    plt.title('Mean VPs across Training Games')
    plt.legend()
    plt.grid(True)

    agents_str = "-".join(agent_types)
    save_path = os.path.join('plots', f'{agents_str}_VPs.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def plot_quest_completion(df, agent_types):
    """
    Plots the mean VPs across training games for each agent type.
    """

    print("hello")
    plt.figure(figsize=(10, 6))

    for agent in agent_types:
        agent_df = df[df['agent type'] == agent]
        agent_df['Fraction Quests Completed'] = agent_df['Quests Completed per Game'] / (agent_df['Quests Taken per Game'] + 2)

        mean_fraction_completed = agent_df.groupby('train games')['Fraction Quests Completed'].mean()
        std_fraction_completed = agent_df.groupby('train games')['Fraction Quests Completed'].std()

        plt.plot(mean_fraction_completed.index, mean_fraction_completed.values, label=f'{agent} (mean)')
        plt.fill_between(mean_fraction_completed.index, 
                         mean_fraction_completed.values - std_fraction_completed.values, 
                         mean_fraction_completed.values + std_fraction_completed.values, 
                         alpha=0.2)

    plt.xlabel('Training Games')
    plt.ylabel('Quest completion frac')
    plt.title('Completion frac')
    plt.legend()
    plt.grid(True)
    # plt.show()

    agents_str = "-".join(agent_types)
    save_path = os.path.join('results/plots', f'{agents_str}_completion.png')
    plt.savefig(save_path)
    


def plot_metric_across_training_games(df, agent_types, metric):
    """
    Plots the specified metric across training games for each agent type and saves the plot to a file.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    agent_types (list): List of agent types.
    metric (str): The metric to plot.
    expname (str): The name of the experiment.
    save_dir (str): The directory where the plot will be saved. Default is 'plots'.
    """
    plt.figure(figsize=(12, 6))

    for agent in agent_types:
        agent_df = df[df['agent type'] == agent]
        mean_metric = agent_df.groupby('train games')[metric].mean()
        std_metric = agent_df.groupby('train games')[metric].std()

        plt.plot(mean_metric.index, mean_metric.values, label=f'{agent} {metric} (mean)')
        plt.fill_between(mean_metric.index, 
                         mean_metric.values - std_metric.values, 
                         mean_metric.values + std_metric.values, 
                         alpha=0.2)

    plt.xlabel('Training Games')
    plt.ylabel(metric)
    plt.title(f'{metric} across Training Games')
    plt.legend()
    plt.grid(True)
    

    agents_str = "-".join(agent_types)
    save_path = os.path.join('results/plots', f'{agents_str}_{metric}.png')
    plt.savefig(save_path)



if __name__ == "__main__":
    path_pattern = 'results/training/PolicyAgent-Random*_*_seed*.csv'
    
    df = read_csv_files(path_pattern)
    
    agent_types = df['agent type'].unique()

    plot_vps_across_training_games(df, agent_types)
    print('here')
    plot_quest_completion(df, agent_types)

    plot_metric_across_training_games(df, agent_types, 'Buildings Purchased per Game')

    plot_metric_across_training_games(df, agent_types, 'win rate')