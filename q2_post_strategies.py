"""
Young You / Hao Le
CSE 163 AE

This module researches the second question of the project.
=> How is the win probability of the game changed in real-time
   based on post-strategies?

The post-strategy represents the states that can be only changed
by the action after the start of the game.

This question researches how much such states that are changed during
the game (e.g., item purchase and game style) influence the win percentage.

[Q2.1] How different are win rates according to the amount of item purchase?
       (Four types: Eco / Semi-Eco / Semi-Buy / Full-Buy)
[Q2-2] Questions related to kills
    Q2-2a. How does the win rate change according to the total number of
           the enemy kills?
    Q2-2b. How does the win rate change according to the total number of
           the first kills in the game?
    Q2-2c. How does the win rate change according to the percent of
           the first kills in the round?
    Q2-2d. How does the win rate change when the team lose each type of
           agent first?
[Q2-3] Questions related to victory strategies
    Q2-3a. How does the win rate change according to the percent of
           the attempt to plant a bomb in the round?
    Q2-3b. How does the win rate change according to the total number
           of defusing bombs?

When the analysis is complete, this file writes the corresponding results
to the /result folder.
"""

from data_depot import DataDepot
import pandas as pd
import q_helper_functions as q_helper
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def analyze_purchases(data_depot):
    """
    Analyze the second question related to item purchases. (#Q2-1)
    Q2.1. How different are win rates according to the amount of item purchase?
          (Four types: Eco / Semi-Eco / Semi-Buy / Full-Buy)

    @param data_depot  The DataDepot class (Data dispatcher)
    """
    df_rounds = data_depot.get_rounds()

    # Remove all the records for the 1st round and the 13rd round.
    # Because they all start with 'eco' without any choice at such rounds,
    # 'eco' at such rounds has a different meaning from 'eco' in other rounds.
    df_rounds = \
        df_rounds[(df_rounds['Team1Score'] + df_rounds['Team2Score'] != 1) &
                  (df_rounds['Team1Score'] + df_rounds['Team2Score'] != 13)]

    # Slice the dataframe to keep only necessary columns
    col_rounds = ['WinnerTeamIdx', 'Team1BuyType', 'Team2BuyType']
    df_rounds = df_rounds[col_rounds]

    # Calculate the result of each purchase of each team
    # Now, it has a string with the win/lose result of the purchase.
    # WinnerTeamIdx Team1BuyType Team2BuyType   Team1BuyWin  Team2BuyWin
    #             1     semi-buy          eco          True        False
    #             1     semi-buy     full-buy          True        False
    #             1     full-buy     semi-eco          True        False
    df_rounds['Team1BuyWin'] = (df_rounds['WinnerTeamIdx'] - 2).astype('bool')
    df_rounds['Team2BuyWin'] = (df_rounds['WinnerTeamIdx'] - 1).astype('bool')

    # Seperate the columns of two teams into each data frame and rename
    # columns so that it can be combined into single dataframe later.
    # Now, each data frame has information of each team purchase result.
    #       BuyType BuyResult  WinnerTeamIdx
    # 0         eco       win              1
    # 1    semi-buy       win              1
    # 2    semi-buy       win              1
    # .....
    df_team1 = df_rounds[['Team1BuyType', 'Team1BuyWin', 'WinnerTeamIdx']]
    df_team2 = df_rounds[['Team2BuyType', 'Team2BuyWin', 'WinnerTeamIdx']]
    df_team1.columns = ['BuyType', 'TeamWin', 'WinnerTeamIdx']
    df_team2.columns = ['BuyType', 'TeamWin', 'WinnerTeamIdx']

    # Concatenate two data field and group records by purchase type and result.
    # BuyType  BuyResult  Count
    # eco      lose          24
    #          win           27
    # full-buy lose         164
    #          win          238
    # .....
    df_purchase = pd.concat([df_team1, df_team2])
    df_purchase = df_purchase.groupby(['BuyType', 'TeamWin']).count()
    df_purchase = df_purchase.rename(columns={"WinnerTeamIdx": "Count"})

    # Calculate the percentage
    df_purchase['Percent'] = df_purchase['Count'] / \
        df_purchase.groupby('BuyType')['Count'].sum()

    # Write the result to the disk
    data_depot.to_csv(df_purchase, 'q2.1_purchases')

    # Plot the results
    plot_purchases(df_purchase)

    # Return results as tuple
    return (df_purchase)


def plot_purchases(df_purchase):
    """
    Plot charts with the data frame analyzed in analyze_purchases

    @param df_purchase  Result data frame. See q2.1_purchases.csv
    """

    # Drop index and adjust table to draw it.
    df_data = df_purchase
    df_data = df_data.reset_index()
    df_data['Percent'] *= 100

    df_data = pd.concat([
        df_data[df_data['BuyType'] == 'eco'],
        df_data[df_data['BuyType'] == 'semi-eco'],
        df_data[df_data['BuyType'] == 'semi-buy'],
        df_data[df_data['BuyType'] == 'full-buy'],
    ])

    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='BuyType', y='Percent', hue='TeamWin')

    # Set title, axis and adjust text size
    ax.set_title('Round Win/Loss Rate by Buy Type', fontsize=20)
    ax.set_xlabel('Buy Type', fontsize=15)
    ax.set_ylabel('Round Win/Loss Rate', fontsize=15)

    # Set Y-axis as a percent format
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Lose', 'Win'],
              title='Round Outcome',
              loc='upper right', fontsize=12)

    file_name = DataDepot.get_result_path() + 'q2.1_purchases.png'
    fig.savefig(file_name, bbox_inches='tight')

    plt.close(fig)


def analyze_kills(data_depot):
    """
    Analyze the second question related to kills. (#Q2-2)
    Q2-2a. How does the win rate change according to the total number of
           the enemy kills?
    Q2-2b. How does the win rate change according to the total number of
           the first kills in the game?
    Q2-2c. How does the win rate change according to the percent of
           the first kills in the round?
    Q2-2d. How does the win rate change when the team lose each type of
           agent first?

    @param data_depot  The DataDepot class (Data dispatcher)
    """
    df_scoreboard = data_depot.get_scoreboard()
    df_games = data_depot.get_games()
    df_teams = data_depot.get_teams()

    # Only take valid records
    df_scoreboard = df_scoreboard[(df_scoreboard['Kills'] > 0) |
                                  (df_scoreboard['Deaths'] > 0) |
                                  (df_scoreboard['Assists'] > 0)]

    # Slice the dataframe to keep only necessary columns
    col_scoreboard = ['GameID', 'Agent', 'TeamAbbr', 'Kills', 'FirstKills',
                      'FirstDeaths']
    df_scoreboard = df_scoreboard[col_scoreboard]

    # Assign team's id and win records into the scoreboard
    df_scoreboard = q_helper.scoreboard_team_record_assigner(
        df_scoreboard, df_games, df_teams)

    # Remove unnecessary columns used in intermediate calculations
    df_scoreboard = df_scoreboard.drop(
        columns=['TeamID', 'TeamAbbr', 'Team1ID', 'Team2ID', 'WinnerTeamIdx'])

    # Q2-2a. How does the win rate change according to the total number of
    #        the enemy kills?
    # Q2-2b. How does the win rate change according to the total number of
    #        the first kills in the game?
    # Q2-2c. How does the win rate change according to the percent of
    #        the first kills in the round?
    df_enemy_kills, df_enemy_first_kills, df_enemy_first_rate = \
        analyze_kills_enemy(df_scoreboard, df_games)
    plot_kills_enemy(df_enemy_kills, df_enemy_first_kills, df_enemy_first_rate)

    # Q2-2d. How does the win rate change when the team lose each type of
    #        agent first?
    df_agent = analyze_kills_agent(df_scoreboard, df_games)
    plot_kills_agent(df_agent)

    # Write the result to the disk
    data_depot.to_csv(df_enemy_kills, 'q2.2a_kills_enemy_total')
    data_depot.to_csv(df_enemy_first_kills, 'q2.2b_kills_enemy_first_kill')
    data_depot.to_csv(df_enemy_first_rate, 'q2.2c_kills_enemy_first_rate')
    data_depot.to_csv(df_agent, 'q2.2d_kills_agent')

    # Return results as tuple
    return (df_enemy_kills, df_enemy_first_kills, df_enemy_first_rate,
            df_agent)


def analyze_kills_enemy(df_scoreboard, df_games):
    """
    Analyze the following sub question.
    Q2-2a. How does the win rate change according to the total number of
           the enemy kills in each round?
    Q2-2b. How does the win rate change according to the total number of
           the first kills in the game?
    Q2-2c. How does the win rate change according to the percent of
           the first kills in the round?

    @param df_scoreboard  The data frame about score board
    @param df_games       The data frame about games
    @return               The result data frame
    """
    # Calculate the total round of the game
    df_games['TotalRound'] = (df_games['Team1_TotalRounds'] +
                              df_games['Team2_TotalRounds'])
    df_games = df_games[['TotalRound']]

    df_scoreboard = df_scoreboard.merge(df_games, left_on=['GameID'],
                                        right_on=['GameID'], how='left')

    # Calculate team result according to the number of kills per round and rate
    # FirstKillsRate means (0: 0%~9%, 1: 10%~19%, ...)
    # GameID  TeamWin  KillsPerRound  FirstKillsRate
    #  60879    False           3.20               2
    #  60880    False           2.80               3
    # ...//
    df_scoreboard['KillsPerRound'] = \
        df_scoreboard['Kills'] / df_scoreboard['TotalRound']
    df_scoreboard['FirstKillRate'] = \
        df_scoreboard['FirstKills'] / df_scoreboard['TotalRound'] * 10

    # Q2-2a. How does the win rate change according to the total number of
    #        the enemy kills in each round?
    df_kills = df_scoreboard[['GameID', 'TeamWin', 'KillsPerRound']]
    df_kills = df_kills.groupby(['GameID', 'TeamWin']).sum()

    # Round KillsPerRound and regroup it by KillsPerRound
    # KillsPerRound TeamWin  GameID
    # 0             False         8
    #               True          1
    # 1             False        85
    #               True         25
    # .....

    # Floor KillsPerRound by every 0.2 step
    df_kills = df_kills.dropna().reset_index()
    df_kills['KillsPerRound'] = (df_kills['KillsPerRound'] * 10) // 2 * 2 / 10
    df_kills = df_kills.groupby(['KillsPerRound', 'TeamWin']).count()
    df_kills = df_kills.rename(columns={"GameID": "Count"})

    # Calculate the percentage
    df_kills['Percent'] = df_kills['Count'] / \
        df_kills.groupby('KillsPerRound')['Count'].sum()

    # Q2-2b. How does the win rate change according to the total number of
    #        the first kills in the game?
    df_first_kills = df_scoreboard[['GameID', 'TeamWin', 'FirstKills']]
    df_first_kills = df_first_kills.groupby(['GameID', 'TeamWin']).sum()

    df_first_kills = df_first_kills.dropna().reset_index()
    df_first_kills = df_first_kills.groupby(['FirstKills', 'TeamWin']).count()
    df_first_kills = df_first_kills.rename(columns={"GameID": "Count"})

    # Calculate the percentage
    df_first_kills['Percent'] = df_first_kills['Count'] / \
        df_first_kills.groupby('FirstKills')['Count'].sum()

    # Q2-2c. How does the win rate change according to the percent of
    #        the first kills in the round?
    df_first_killrate = df_scoreboard[['GameID', 'TeamWin', 'FirstKillRate']]
    df_first_killrate = df_first_killrate.groupby(['GameID', 'TeamWin']).sum()

    df_first_killrate = df_first_killrate.dropna().reset_index()
    df_first_killrate['FirstKillRate'] = \
        (df_first_killrate['FirstKillRate'] + 0.5).astype(int)

    df_first_killrate = \
        df_first_killrate.groupby(['FirstKillRate', 'TeamWin']).count()
    df_first_killrate = df_first_killrate.rename(columns={"GameID": "Count"})

    # Calculate the percentage
    df_first_killrate['Percent'] = df_first_killrate['Count'] / \
        df_first_killrate.groupby('FirstKillRate')['Count'].sum()

    return (df_kills, df_first_kills, df_first_killrate)


def plot_kills_enemy(df_enemy_kills, df_enemy_first_kills,
                     df_enemy_first_rate):
    """
    Plot charts with the data frame analyzed in analyze_kills_enemy

    @param df_enemy_kills        Result. See q2.2a_kills_enemy_total.csv
    @param df_enemy_first_kills  Result. See q2.2b_kills_enemy_first_kill.csv
    @param df_enemy_first_rate   Result. See q2.2c_kills_enemy_first_rate.csv
    """

    # Plot enemy_kills
    df_data = df_enemy_kills
    df_data = df_data.reset_index()
    df_data['Percent'] *= 100

    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='KillsPerRound', y='Percent',
                hue='TeamWin')

    # Set title, axis and adjust text size
    ax.set_title('Round Win/Loss Rate by # of Enemies Killed per Round',
                 fontsize=20)
    ax.set_xlabel('# of Enemies Killed', fontsize=15)
    ax.set_ylabel('Round Win/Loss Rate', fontsize=15)

    # Set Y-axis as a percent format
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Lose', 'Win'],
              title='Round Outcome',
              loc='lower right', fontsize=12)

    file_name = DataDepot.get_result_path() + 'q2.2a_kills_enemy_total.png'
    fig.savefig(file_name, bbox_inches='tight')

    # Plot enemy_first_kills
    df_data = df_enemy_first_kills
    df_data = df_data.reset_index()
    df_data['Percent'] *= 100

    # Convert FirstKills to integer from float
    df_data['FirstKills'] = df_data['FirstKills'].astype(int)

    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='FirstKills', y='Percent',
                hue='TeamWin')

    # Set title, axis and adjust text size
    ax.set_title('Game Win/Loss Rate by # of FirstKills',
                 fontsize=20)
    ax.set_xlabel('# of FirstKills', fontsize=15)
    ax.set_ylabel('Game Win/Loss Rate', fontsize=15)

    # Set Y-axis as a percent format
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Lose', 'Win'],
              title='Game Outcome',
              loc='lower right', fontsize=12)

    file_name = \
        DataDepot.get_result_path() + 'q2.2b_kills_enemy_first_kill.png'
    fig.savefig(file_name, bbox_inches='tight')

    # Plot enemy_first_rate
    df_data = df_enemy_first_rate
    df_data = df_data.reset_index()
    df_data['Percent'] *= 100

    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='FirstKillRate', y='Percent',
                hue='TeamWin')

    # Set title, axis and adjust text size
    ax.set_title('Game Win/Loss Rate by First Kill Rate',
                 fontsize=20)
    ax.set_xlabel('First Kill Rate', fontsize=15)
    ax.set_ylabel('Game Win/Loss Rate', fontsize=15)

    # Set X-axis as a percent format
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=10))

    # Set Y-axis as a percent format
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Lose', 'Win'],
              title='Game Outcome',
              loc='lower right', fontsize=12)

    file_name = \
        DataDepot.get_result_path() + 'q2.2c_kills_enemy_first_rate.png'
    fig.savefig(file_name, bbox_inches='tight')

    plt.close(fig)


def analyze_kills_agent(df_scoreboard, df_games):
    """
    Analyze the following sub question.
    Q2-2d. How does the win rate change when the team lose each type of
           agent first?

    @param df_scoreboard  The data frame about score board
    @param df_games       The data frame about games
    @return               The result data frame
    """
    # Calculate the total round of the game
    df_games['TotalRound'] = (df_games['Team1_TotalRounds'] +
                              df_games['Team2_TotalRounds'])
    df_games = df_games[['TotalRound']]

    # Calculate team result according to the number of FirstDeathRate
    # FirstDeathRate means (0: 0%~9%, 1: 10%~19%, ...)
    # GameID  TeamWin   ...  FirstDeathRate
    #  60879    False   ...               2
    #  60880    False   ...               3
    # .....
    df_scoreboard = df_scoreboard.merge(df_games, left_on=['GameID'],
                                        right_on=['GameID'], how='left')
    df_scoreboard['FirstDeathRate'] = \
        df_scoreboard['FirstDeaths'] / df_scoreboard['TotalRound'] * 10

    # Calculate agent type
    # Now, it has a type for each agent
    # GameID    Agent  Kills  ...  TeamWin  FirstDeathRate    AgentType
    #  60894     Jett   24.0          True           2.000     Duelists
    #  60894  Chamber   16.0          True           0.500    Sentinels
    # .....
    df_scoreboard['AgentType'] = \
        df_scoreboard.apply(q_helper.agents_class_classifier, axis=1)

    # Slice the dataframe to keep only necessary columns
    df_agent = df_scoreboard[['TeamWin', 'FirstDeathRate', 'AgentType']]
    df_agent = df_agent.dropna().reset_index()
    df_agent['FirstDeathRate'] = df_agent['FirstDeathRate'].astype(int)
    df_agent = \
        df_agent.groupby(['AgentType', 'FirstDeathRate', 'TeamWin']).count()
    df_agent = df_agent.rename(columns={"index": "Count"})

    # Calculate the percentage
    df_agent['Percent'] = df_agent['Count'] / \
        df_agent.groupby(['AgentType', 'FirstDeathRate'])['Count'].sum()

    return df_agent


def plot_kills_agent(df_agent):
    """
    Plot charts with the data frame analyzed in analyze_kills_agent

    @param df_agent  Result data frame. See q2.2d_kills_agent.csv
    """

    df_data = df_agent
    df_data = df_data.reset_index()
    df_data['Percent'] *= 100

    # Make 2 x 2 plots
    row_max = 2
    col_max = 2

    fig, axes = plt.subplots(row_max, col_max, figsize=(20, 15))

    all_agent = df_data['AgentType'].drop_duplicates().to_list()
    index = 0

    # For fontsize for the legend.
    sns.set(font_scale=1.2)

    for agent_type in all_agent:
        ax = axes[index // col_max][index % col_max]
        index += 1

        df_subdata = df_data[df_data['AgentType'] == agent_type].sort_values(
                         by='Percent', ascending=False)

        sns.barplot(ax=ax, data=df_subdata, x='FirstDeathRate', y='Percent',
                    hue='TeamWin')

        ax.set_title(
            f"Game Win/Loss Rate when {agent_type}" + " Die First Every Round",
            fontsize=20)

        ax.set_xlabel('First Death Rate', fontsize=15)
        ax.set_ylabel('Percent', fontsize=15)

        # Make X-axis as a percent format
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=10))

        # Make Y-axis as a percent format
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    file_name = DataDepot.get_result_path() + 'q2.2d_kills_agent.png'
    fig.savefig(file_name, bbox_inches='tight')

    plt.close(fig)


def analyze_strategy(data_depot):
    """
    Analyze the second question related to victory strategies. (#Q2-3)
    Q2-3a. How does the win rate change according to the percent of
           the attempt to plant a bomb in the round?
    Q2-3b. How does the win rate change according to the total number
           of defusing bombs?

    @param data_depot  The DataDepot class (Data dispatcher)
    @return            The result data frame
    """
    df_scoreboard = data_depot.get_scoreboard()
    df_games = data_depot.get_games()
    df_teams = data_depot.get_teams()

    # Only take valid records
    df_scoreboard = df_scoreboard[(df_scoreboard['Plants'] > 0) |
                                  (df_scoreboard['Defuses'] > 0)]

    # Slice the dataframe to keep only necessary columns
    col_scoreboard = ['GameID', 'Agent', 'TeamAbbr', 'Plants', 'Defuses']
    df_scoreboard = df_scoreboard[col_scoreboard]

    # Assign team's id and win records into the scoreboard
    df_scoreboard = q_helper.scoreboard_team_record_assigner(
        df_scoreboard, df_games, df_teams)

    # Remove unnecessary columns used in intermediate calculations
    df_scoreboard = df_scoreboard.drop(
        columns=['TeamID', 'TeamAbbr', 'Team1ID', 'Team2ID', 'WinnerTeamIdx'])

    # Calculate the total round of the game
    df_games['TotalRound'] = (df_games['Team1_TotalRounds'] +
                              df_games['Team2_TotalRounds'])
    df_games = df_games[['TotalRound']]

    df_scoreboard = df_scoreboard.merge(df_games, left_on=['GameID'],
                                        right_on=['GameID'], how='left')

    # Calculate team result according to the number of plant/defuse per round.
    # PlantsPerRound means (0: 0%~9%, 1: 10%~19%, ...) / per single round
    # DefusesPerGame means total defuse count per all rounds
    df_scoreboard['PlantsPerRound'] = \
        df_scoreboard['Plants'] / df_scoreboard['TotalRound'] * 10
    df_scoreboard = df_scoreboard.rename(columns={"Defuses": "DefusesPerGame"})

    # Q2-3a. How does the win rate change according to the percent of
    #        the attempt to plant a bomb in the round?
    df_plants = df_scoreboard[['GameID', 'TeamWin', 'PlantsPerRound']]
    df_plants = df_plants.groupby(['GameID', 'TeamWin']).sum()

    # Floor PlantsPerRound and regroup it by PlantsPerRound
    df_plants = df_plants.dropna().reset_index()
    df_plants['PlantsPerRound'] = (df_plants['PlantsPerRound']).astype(int)
    df_plants = df_plants.groupby(['PlantsPerRound', 'TeamWin']).count()
    df_plants = df_plants.rename(columns={"GameID": "Count"})

    # Calculate the percentage
    df_plants['Percent'] = df_plants['Count'] / \
        df_plants.groupby('PlantsPerRound')['Count'].sum()

    # Q2-3b. How does the win rate change according to the total number
    #        of defusing bombs?
    df_defuses = df_scoreboard[['GameID', 'TeamWin', 'DefusesPerGame']]
    df_defuses = df_defuses.groupby(['GameID', 'TeamWin']).sum()

    # Floor DefusesPerGame and regroup it by DefusesPerGame
    df_defuses = df_defuses.dropna().reset_index()
    df_defuses['DefusesPerGame'] = (df_defuses['DefusesPerGame']).astype(int)
    df_defuses = df_defuses.groupby(['DefusesPerGame', 'TeamWin']).count()
    df_defuses = df_defuses.rename(columns={"GameID": "Count"})

    # Calculate the percentage
    df_defuses['Percent'] = df_defuses['Count'] / \
        df_defuses.groupby('DefusesPerGame')['Count'].sum()

    # Write the result to the disk
    data_depot.to_csv(df_plants, 'q2.3a_strategy_plants')
    data_depot.to_csv(df_defuses, 'q2.3b_strategy_defuses')

    # Plot the results
    plot_strategy(df_plants, df_defuses)

    # Return results as tuple
    return (df_plants, df_defuses)


def plot_strategy(df_plants, df_defuses):
    """
    Plot charts with the data frame analyzed in analyze_strategy

    @param df_plants   Result data frame. See q2.3a_strategy_plants.csv
    @param df_defuses  Result data frame. See q2.3b_strategy_defuses.csv
    """
    # Plot plants
    df_data = df_plants
    df_data = df_data.reset_index()
    df_data['Percent'] *= 100

    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='PlantsPerRound', y='Percent',
                hue='TeamWin')

    # Set title, axis and adjust text size
    ax.set_title('Game Win/Loss Rate by Spike Plant Rate',
                 fontsize=20)
    ax.set_xlabel('Spike Plant Rate', fontsize=15)
    ax.set_ylabel('Game Win/Loss Rate', fontsize=15)

    # Set X-axis as a percent format
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=10))

    # Set Y-axis as a percent format
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Lose', 'Win'],
              title='Game Outcome',
              loc='lower right', fontsize=12)

    file_name = DataDepot.get_result_path() + 'q2.3a_strategy_plants.png'
    fig.savefig(file_name, bbox_inches='tight')

    # Plot defuses
    df_data = df_defuses
    df_data = df_data.reset_index()
    df_data['Percent'] *= 100

    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='DefusesPerGame', y='Percent',
                hue='TeamWin')

    # Set title, axis and adjust text size
    ax.set_title('Game Win/Loss Rate by Spike Defusal Count',
                 fontsize=20)
    ax.set_xlabel('Spike Defusal Count', fontsize=15)
    ax.set_ylabel('Game Win/Loss Rate', fontsize=15)

    # Set Y-axis as a percent format
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Lose', 'Win'],
              title='Game Outcome',
              loc='lower right', fontsize=12)

    file_name = DataDepot.get_result_path() + 'q2.3b_strategy_defuses.png'
    fig.savefig(file_name, bbox_inches='tight')

    plt.close(fig)


def execute_module(data_depot):
    """
    This method executes only this module partly.
    If you want to execute the main project, please execute main.py
    If you want to test this module logic, please execute tester.py

    @param data_depot  The DataDepot class (Data dispatcher)
    """
    # Initialize seaborn
    sns.set()

    # Question set 2
    print("- Analyzing question set #2")

    print("- Analyzing question 2 - purchases")
    analyze_purchases(data_depot)

    print("- Analyzing question 2 - kills")
    analyze_kills(data_depot)

    print("- Analyzing question 2 - strategy")
    analyze_strategy(data_depot)


if __name__ == '__main__':
    print("Executing here only tests functions in this module.")
    print("To execute the main of the project, Please execute main.py.")

    print("- Initializing data depot")
    data_depot = DataDepot(test_mode=False)
    execute_module(data_depot)
