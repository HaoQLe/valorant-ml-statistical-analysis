"""
Young You / Hao Le
CSE 163 AE

This module researches the first question of the project.
=> What are the most effective pre-strategies?

The pre-strategy represents the states that can be only changed
by the decision before the start of the game.

This question researches how much such states that are decided before
the game start (e.g., game maps and agents) influence the win percentage.

[Q1.1] Questions related to maps
    Q1-1a. Is this map advantageous for the first attack or for defense?
    Q1-1b. Among four victory conditions (Elim/Boom/Defuse/Time),
        which winning strategy is advantageous on this map?
    Q1-1c. Which agent has the most advantage on this map?

[Q1-2] Questions related to agents
    Q1-2a. Win rate change rate when the same class of agent enters
        (Four classes : Duelists / Controllers / Initiators / Sentinels)
    Q1-2b. Which agent is more likely to kill the enemy first?
    Q1-2c. Which agent is more likely to be killed by the enemy first?
    Q1-2d. Which agent is more likely to be the last to survive on the team?
    Q1-2e. Which agent is more likely to kill the most enemies?
    Q1-2f. Which agent is most likely to be killed by the enemies?
    Q1-2g. Which agent is most likely to assist to kill enemies?
    Q1-2h. Which agent is most likely to plant a bomb?
    Q1-2i. Which agent is most likely to defuse a bomb?

When the analysis is complete, this file writes the corresponding results
to the /result folder.
"""

from data_depot import DataDepot
import q_helper_functions as q_helper
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def analyze_maps(data_depot):
    """
    Analyze the first question related to game maps. (#Q1-1)
    This function researches these three sub questions.

    Q1-1a. Is this map advantageous for the first attack or for defense?
    Q1-1b. Among four victory conditions (Elim/Boom/Defuse/Time),
           which winning strategy is advantageous on this map?
    Q1-1c. Which agent has the most advantage on this map?

    More advantageous means it shows higher win rate.

    @param data_depot  The DataDepot class (Data dispatcher)
    """
    df_games = data_depot.get_games()
    df_rounds = data_depot.get_rounds()
    df_teams = data_depot.get_teams()
    df_scoreboard = data_depot.get_scoreboard()

    # Slice the dataframe to keep only necessary columns
    col_games = ['Map', 'Team1ID', 'Team2ID', 'Team1_SideFirstHalf',
                 'Team2_SideFirstHalf', 'WinnerTeamIdx']
    df_games = df_games[col_games]

    col_rounds = ['GameID', 'WinType']
    df_rounds = df_rounds[col_rounds]

    col_scoreboard = ['GameID', 'Agent', 'TeamAbbr']
    df_scoreboard = df_scoreboard[col_scoreboard]

    # Drop all uncompleted fields
    df_games = df_games[df_games['Map'] != 'TBD']
    df_games = df_games.dropna()
    df_rounds = df_rounds.dropna()
    df_teams = df_teams.dropna()
    df_scoreboard = df_scoreboard.dropna()

    # Q1-1a. Is first attack advantageous or first defense advantageous?
    df_first_role = analyze_maps_first_role(df_games)
    plot_maps_first_role(df_first_role)

    # Q1-1b. Among three victory conditions (Elim/Boom/Defuse),
    #        which winning strategy is advantageous?
    df_winning_type = analyze_maps_winning_type(df_games, df_rounds)
    plot_maps_winning_type(df_winning_type)

    # Q1-1c. Which agent has the most advantage on this map?
    df_agents = analyze_maps_agents(df_games, df_scoreboard, df_teams)
    plot_maps_agents(df_agents)

    # Write the result to the disk
    data_depot.to_csv(df_first_role, 'q1.1a_maps_first_role')
    data_depot.to_csv(df_winning_type, 'q1.1b_maps_winning_type')
    data_depot.to_csv(df_agents, 'q1.1c_maps_agents')

    # Return results as tuple
    return (df_first_role, df_winning_type, df_agents)


def analyze_maps_first_role(df_games):
    """
    Analyze the following sub question.
    Q1-1a. Is this map advantageous for the first attack or for defense?

    @param df_games    The data frame about games
    @return            The result data frame
    """
    # Calculate winner's first half role
    df_games['WinnerFirstHalfRole'] = df_games.apply(
        lambda row:
            row['Team1_SideFirstHalf'] if (row['WinnerTeamIdx'] == 1)
            else row['Team2_SideFirstHalf'], axis=1
    )

    # Slice the dataframe to keep only necessary columns
    # It has information on the winning team according to the role on each map
    # GameID     Map WinnerFirstHalfRole  WinnerTeamIdx
    # 60894   Breeze              defend              1
    # 60895     Bind              attack              2
    # .....
    df_games = df_games[['Map', 'WinnerFirstHalfRole', 'WinnerTeamIdx']]

    # Group records by map and role
    # Now it has the total winning count of each map according to the role
    # Map    WinnerFirstHalfRole      Count
    # Ascent attack                   2
    #        defend                   1
    # .....
    df_result = df_games.groupby(['Map', 'WinnerFirstHalfRole']).count()
    df_result.columns = ['Count']

    # Calculate the percentage
    df_result['Percent'] = df_result['Count'] / \
        df_result.groupby('Map')['Count'].sum()

    return df_result


def plot_maps_first_role(df_first_role):
    """
    Plot charts with the data frame analyzed in analyze_maps_first_role

    @param df_first_role  Result data frame. See q1.1a_maps_first_role.csv
    """
    # Drop index and adjust table to draw it.
    df_data = df_first_role.reset_index()
    df_data['Percent'] *= 100

    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='Map', y='Percent',
                hue='WinnerFirstHalfRole')

    # Set title, axis and adjust text size
    ax.set_title('Win Rate by Map according to First Role', fontsize=20)
    ax.set_xlabel('Map Type', fontsize=15)
    ax.set_ylabel('Win Rate', fontsize=15)

    # Make Y-axis as a percent format
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Attack', 'Defence'],
              title='The First Role',
              loc='lower right', fontsize=12)

    file_name = DataDepot.get_result_path() + 'q1.1a_maps_first_role.png'
    fig.savefig(file_name, bbox_inches='tight')

    plt.close(fig)


def analyze_maps_winning_type(df_games, df_rounds):
    """
    Analyze the following sub question.
    Q1-1b. Among four victory conditions (Elim/Boom/Defuse/Time),
           which winning strategy is advantageous on this map?

    @param df_games    The data frame about games
    @param df_rounds   The data frame about rounds
    @return            The result data frame
    """
    # Slice the dataframe to keep only necessary columns
    # Now, it has map type of each map.
    # GameID     Map
    # 60894   Breeze
    # 60895     Bind
    # .....
    df_games = df_games[['Map']]

    # Join game and round data frames
    # Now it has winning type of each rounds of each game according to the map
    # GameID  WinType     Map
    #   60894    elim  Breeze
    #   60894    elim  Breeze
    # .....
    df_result = df_rounds.merge(df_games, left_on='GameID',
                                right_on='GameID', how='left')

    # Drop all mission data fields
    df_result = df_result.dropna()

    # Group records by map and win type
    # Now it has the total wins grouped by the map and winning type.
    # Map    WinType  Count
    # Ascent boom         2
    #        defuse      14
    #        elim        40
    # .....
    df_result = df_result.groupby(['Map', 'WinType']).count()
    df_result = df_result.rename(columns={"GameID": "Count"})

    # Calculate the percentage
    df_result['Percent'] = df_result['Count'] / \
        df_result.groupby('Map')['Count'].sum()

    return df_result


def plot_maps_winning_type(df_winning_type):
    """
    Plot charts with the data frame analyzed in analyze_maps_winning_type

    @param df_winning_type  Result data frame. See q1.1b_maps_winning_type.csv
    """
    df_data = df_winning_type.reset_index()
    df_data['Percent'] *= 100

    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(x='Map', y='Percent', hue='WinType', data=df_data, ax=ax)

    # Set title, axis and adjust text size
    ax.set_title('Win Rate by Map according to Victory Type', fontsize=20)
    ax.set_xlabel('Victory Type', fontsize=15)
    ax.set_ylabel('Win Rate', fontsize=15)

    # Make Y-axis as a percent format
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Boom', 'Defuse', 'Elimination', 'Timeout'],
              title='Victory Type',
              fontsize=12, loc='upper right')

    file_name = DataDepot.get_result_path() + 'q1.1b_maps_winning_type.png'
    fig.savefig(file_name, bbox_inches='tight')

    plt.close(fig)


def analyze_maps_agents(df_games, df_scoreboard, df_teams):
    """
    Analyze the following sub question.
    Q1-1c. Which agent has the most advantage on this map?

    @param df_games       The data frame about games
    @param df_scoreboard  The data frame about scoreboard
    @param df_teams       The data frame about teams
    @return               The result data frame
    """
    # Assign team's id and win records into the scoreboard
    df_scoreboard = q_helper.scoreboard_team_record_assigner(
        df_scoreboard, df_games, df_teams)

    # Calculate and make a winner team ID field to join.
    # Now, it has WinnerTeamID according to the WinnerTeamIdx pre-calculated
    # GameID     Map  Team1ID  Team2ID ...  WinnerTeamIdx   WinnerTeamID
    # 60894   Breeze     6903     6020 ...              1           6903
    # 60895     Bind     6903     6020 ...              2           6020
    # .....
    df_games['WinnerTeamID'] = df_games.apply(
        lambda row:
            row['Team1ID'] if (row['WinnerTeamIdx'] == 1)
            else row['Team2ID'], axis=1
    )

    # Slice the dataframe to keep only necessary columns
    # Now, df_games has WinnerTeamID on each map
    # GameID,   Map,    WinnerTeamID
    # 60894,    Breeze, 6903
    # 60895,    Bind,   6020
    # .....
    col_games = ['Map', 'WinnerTeamID']
    df_games = df_games[col_games]

    # Now, df_scoreboard has used agents of each team on each game
    # GameID,   Agent,   TeamID
    # 60894,    Jett,    6903
    # 60894,    Chamber, 6903
    # .....
    col_scoreboard = ['GameID', 'Agent', 'TeamID']
    df_scoreboard = df_scoreboard[col_scoreboard]

    # To get team ID, join df_scoreboard and df_teams data frames
    # Only records relevant to the winning team are required.
    # After joining, it has all agents of winning team on each map
    # GameID,   Map,    WinnerTeamID,   Agent,      TeamID
    # 60894,    Breeze, 6903,           Jett,       6903
    # 60894,    Breeze, 6903,           Chamber,    6903
    # .....
    df_games = df_games.merge(df_scoreboard,
                              left_on=['GameID', 'WinnerTeamID'],
                              right_on=['GameID', 'TeamID'], how='left')

    # Group records by map and agents type
    # Get the agent with the highest win rate for each map
    # Now it has the total wins grouped by the map and winning type.
    # Map     Agent    Count
    # Ascent  Astra        3
    #         Breach       1
    # .....
    df_games = df_games[['GameID', 'Map', 'Agent']]
    df_result = df_games.groupby(['Map', 'Agent']).count()
    df_result = df_result.rename(columns={"GameID": "Count"})

    # Calculate the percentage
    df_result['Percent'] = df_result['Count'] / \
        df_result.groupby('Map')['Count'].sum()

    return df_result


def plot_maps_agents(df_agents):
    """
    Plot charts with the data frame analyzed in analyze_maps_agents

    @param df_agents  Result data frame. See q1.1c_maps_agents.csv
    """
    df_data = df_agents.reset_index()
    df_data['Percent'] *= 100

    # Make 4 x 2 plots
    row_max = 4
    col_max = 2
    fig, axes = plt.subplots(row_max, col_max, figsize=(20, 30))

    # Since we use only 7 charts, remove the last chart
    fig.delaxes(axes[3][1])

    all_map = df_data['Map'].drop_duplicates().to_list()
    index = 0

    # For fontsize for the legend.
    sns.set(font_scale=1.2)

    for map_name in all_map:
        ax = axes[index // col_max][index % col_max]
        index += 1

        # Take only data for each map
        df_subdata = df_data[df_data['Map'] == map_name].sort_values(
                         by='Percent', ascending=False)
        df_subdata = df_subdata[:5]

        # Fill bar size maximum with dodge=False
        sns.barplot(ax=ax, data=df_subdata, x='Agent', y='Percent',
                    hue='Agent', dodge=False)

        ax.set_title(f"Top 5 Agents on {map_name}", fontsize=20)

        ax.set_xlabel('Agent', fontsize=15)
        ax.set_ylabel('Selection Ratio', fontsize=15)

        # Make Y-axis as a percent format
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    file_name = DataDepot.get_result_path() + 'q1.1c_maps_agents.png'
    fig.savefig(file_name, bbox_inches='tight')

    plt.close(fig)


def analyze_agents(data_depot):
    """
    Analyze the first question related to game agents. (#Q1-2)
    This function researches these nine sub questions.

    Q1-2a. Win rate change rate when the same class of agent enters
           (Four classes : Duelists / Controllers / Initiators / Sentinels)
    Q1-2b. Which agent is more likely to kill the enemy first?
    Q1-2c. Which agent is more likely to be killed by the enemy first?
    Q1-2d. Which agent is more likely to be the last to survive on the team?
    Q1-2e. Which agent is more likely to kill the most enemies?
    Q1-2f. Which agent is most likely to be killed by the enemies?
    Q1-2g. Which agent is most likely to assist to kill enemies?
    Q1-2h. Which agent is most likely to plant a bomb?
    Q1-2i. Which agent is most likely to defuse a bomb?

    @param data_depot  The DataDepot class (Data dispatcher)
    """
    df_scoreboard = data_depot.get_scoreboard()
    df_games = data_depot.get_games()
    df_teams = data_depot.get_teams()

    # Slice the dataframe to keep only necessary columns
    col_scoreboard = ['GameID', 'PlayerID', 'TeamAbbr', 'Agent', 'Kills',
                      'Deaths', 'Assists', 'OnevOne', 'OnevTwo', 'OnevThree',
                      'OnevFour', 'OnevFive', 'FirstKills', 'FirstDeaths',
                      'Plants', 'Defuses']
    df_scoreboard = df_scoreboard[col_scoreboard]

    # Slice the dataframe to keep only necessary columns
    col_games = ['Team1ID', 'Team2ID', 'Team1', 'Team2', 'WinnerTeamIdx',
                 'Team1_TotalRounds', 'Team2_TotalRounds']
    df_games = df_games[col_games]

    # Drop all uncompleted fields
    df_scoreboard = df_scoreboard.dropna()
    df_games = df_games.dropna()
    df_teams = df_teams.dropna()

    # Calculate agent type
    # Now, it has a type for each agent
    # GameID    Agent  Kills  Deaths  ...    AgentType
    #  60894     Jett   24.0    10.0  ...     Duelists
    #  60894  Chamber   16.0    10.0  ...    Sentinels
    # ...
    df_scoreboard['AgentType'] = \
        df_scoreboard.apply(q_helper.agents_class_classifier, axis=1)

    # Q1-2a. Win rate change rate when the same class of agent enters
    #        (Four classes : Duelists / Controllers / Initiators / Sentinels)
    df_class = analyze_agents_class(df_scoreboard, df_games, df_teams)
    plot_agents_class(df_class)

    # Q1-2b. Which agent is more likely to kill the enemy first?
    # Q1-2c. Which agent is more likely to be killed by the enemy first?
    # Q1-2d. Which agent is more likely to be the last to survive on the team?
    df_first_kill, df_first_dead, df_last_survive = \
        analyze_agents_first_kill_dead(df_scoreboard, df_games)
    plot_agents_first_kill_dead(df_first_kill, df_first_dead, df_last_survive)

    # Q1-2e. Which agent is more likely to kill the most enemies?
    # Q1-2f. Which agent is most likely to be killed by the enemies?
    # Q1-2g. Which agent is most likely to assist to kill enemies?
    # Q1-2h. Which agent is most likely to plant a bomb?
    # Q1-2i. Which agent is most likely to defuse a bomb?
    df_most_kill, df_most_dead, df_most_assist, \
        df_most_plant, df_most_defuse = \
        analyze_agents_most_kill_dead_assist(df_scoreboard, df_games)
    plot_agents_most_kill_dead_assist(df_most_kill, df_most_dead,
                                      df_most_assist, df_most_plant,
                                      df_most_defuse)

    # Write the result to the disk
    data_depot.to_csv(df_class, 'q1.2a_agents_class')
    data_depot.to_csv(df_first_kill, 'q1.2b_agents_first_kill')
    data_depot.to_csv(df_first_dead, 'q1.2c_agents_first_dead')
    data_depot.to_csv(df_last_survive, 'q1.2d_agents_last_survive')
    data_depot.to_csv(df_most_kill, 'q1.2e_agents_most_kill')
    data_depot.to_csv(df_most_dead, 'q1.2f_agents_most_dead')
    data_depot.to_csv(df_most_assist, 'q1.2g_agents_most_assist')
    data_depot.to_csv(df_most_plant, 'q1.2h_agents_most_plant')
    data_depot.to_csv(df_most_defuse, 'q1.2i_agents_most_defuse')

    # Return results as tuple
    return (df_class, df_first_kill, df_first_dead, df_last_survive,
            df_most_kill, df_most_dead, df_most_assist,
            df_most_plant, df_most_defuse)


def analyze_agents_class(df_scoreboard, df_games, df_teams):
    """
    Analyze the following sub question.
    Q1-2a. Win rate change rate when the same class of agent enters
           (Four classes : Duelists / Controllers / Initiators / Sentinels)

    @param df_scoreboard  The data frame about score board
    @param df_games       The data frame about games
    @param df_teams       The data frame about teams
    @return               The result data frame
    """
    # Group records by GameID, Team, and AgentType
    # Now it has the total number of the class types in each game for each team
    # GameID TeamAbbr AgentType    Count
    # 60879  T69      Controllers      1
    #                 Duelists         2
    # .....
    df_result = df_scoreboard[['GameID', 'TeamAbbr', 'AgentType', 'Agent']]
    df_result = df_result.groupby(['GameID', 'TeamAbbr', 'AgentType']).count()
    df_result = df_result.rename(columns={"Agent": "Count"})

    # Filter the game that only selected two or more the same class type
    df_result = df_result[df_result['Count'] >= 2]
    df_result = df_result.reset_index()

    # Assign team's id and win records into the result
    df_result = q_helper.scoreboard_team_record_assigner(
        df_result, df_games, df_teams)

    # Group by AgentType and TeamWin
    # AgentType   TeamWin  Count
    # Controllers False     1304
    #             True      1370
    # .....
    df_result = df_result[['AgentType', 'TeamWin', 'WinnerTeamIdx']]
    df_result = df_result.groupby(['AgentType', 'TeamWin']).count()
    df_result = df_result.rename(columns={"WinnerTeamIdx": "Count"})

    # Calculate the percentage
    df_result['Percent'] = df_result['Count'] / \
        df_result.groupby('AgentType')['Count'].sum()

    return df_result


def plot_agents_class(df_class):
    """
    Plot charts with the data frame analyzed in analyze_agents_class

    @param df_class  Result data frame. See q1.2a_agents_class.csv
    """
    df_data = df_class
    df_data = df_data.reset_index()
    df_data['Percent'] *= 100

    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='AgentType', y='Percent', hue='TeamWin')

    # Set title, axis and adjust text size
    ax.set_title('Round Win/Loss Rate by Agent Role', fontsize=20)
    ax.set_xlabel('Agent Role', fontsize=15)
    ax.set_ylabel('Round Win/Loss Rate', fontsize=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Lose', 'Win'],
              title='Game Outcome',
              fontsize=12, loc='lower right')

    file_name = DataDepot.get_result_path() + 'q1.2a_agents_class.png'
    fig.savefig(file_name, bbox_inches='tight')

    plt.close(fig)


def analyze_agents_first_kill_dead(df_scoreboard, df_games):
    """
    Analyze the following sub questions.
    Q1-2b. Which agent is more likely to kill the enemy first?
    Q1-2c. Which agent is more likely to be killed by the enemy first?
    Q1-2d. Which agent is more likely to be the last to survive on the team?

    @param df_scoreboard  The data frame about score board
    @param df_games       The data frame about games
    @return               The result data frame
    """
    # Slice the dataframe to keep only necessary columns
    df_result = df_scoreboard[['GameID', 'Agent', 'AgentType', 'FirstKills',
                               'FirstDeaths', 'OnevOne', 'OnevTwo',
                               'OnevThree', 'OnevFour', 'OnevFive']]

    # Calculates the probability of the first kill/death event based on the
    # total game rounds of each game to prevent the highly selected agent wins
    # Now, it has FirstKillsRate and FirstDeathsRate
    # GameID    Agent  AgentType  FirstKills ... FirstKillsRate FirstDeathsRate
    #  60894     Jett   Duelists         4.0 ...         0.2000          0.2000
    #  60894  Chamber  Sentinels         1.0 ...         0.0500          0.0500
    # .....
    df_games['TotalRound'] = (df_games['Team1_TotalRounds'] +
                              df_games['Team2_TotalRounds'])
    df_games = df_games[['TotalRound']]
    df_result = df_result.merge(df_games, left_on=['GameID'],
                                right_on=['GameID'], how='left')

    df_result['FirstKillsRate'] = df_result['FirstKills'] / \
        df_result['TotalRound']

    df_result['FirstDeathsRate'] = df_result['FirstDeaths'] / \
        df_result['TotalRound']

    df_result['LastSurviveRate'] = \
        (df_result['OnevOne'] + df_result['OnevTwo'] +
         df_result['OnevThree'] + df_result['OnevFour'] +
         df_result['OnevFive']) / df_result['TotalRound']

    # Group by agent and calculate the average of its percentage
    # Agent   AgentType    FirstKillsRate
    # Astra   Controllers        0.072573
    # Breach  Initiators         0.062302
    # .....
    df_kill = df_result[['Agent', 'AgentType', 'FirstKillsRate']]
    df_kill = df_kill.groupby(['Agent', 'AgentType']).mean()
    df_death = df_result[['Agent', 'AgentType', 'FirstDeathsRate']]
    df_death = df_death.groupby(['Agent', 'AgentType']).mean()
    df_survive = df_result[['Agent', 'AgentType', 'LastSurviveRate']]
    df_survive = df_survive.groupby(['Agent', 'AgentType']).mean()

    return (df_kill, df_death, df_survive)


def plot_agents_first_kill_dead(df_first_kill, df_first_dead, df_last_survive):
    """
    Plot charts with the data frame analyzed in analyze_agents_first_kill_dead

    @param df_first_kill    Result dataframe. See q1.2b_agents_first_kill.csv
    @param df_first_dead    Result dataframe. See q1.2c_agents_first_dead.csv
    @param df_last_survive  Result dataframe. See q1.2d_agents_last_survive.csv
    """

    # Plot graph for first kill
    df_data = df_first_kill
    df_data = df_data.sort_values('AgentType')
    df_data = df_data.reset_index()

    df_data['FirstKillsRate'] *= 100
    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='Agent', y='FirstKillsRate',
                hue='AgentType')

    # Set title, axis and adjust text size
    ax.set_title('First Kill Rate by Agent', fontsize=20)
    ax.set_xlabel('Agent', fontsize=15)
    ax.set_ylabel('First Kill Rate', fontsize=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Controller', 'Duelist', 'Initiator', 'Sentinel'],
              title='Agent Role',
              fontsize=12, loc='upper right')

    file_name = DataDepot.get_result_path() + 'q1.2b_agents_first_kill.png'
    fig.savefig(file_name, bbox_inches='tight')

    # Plot graph for first death
    df_data = df_first_dead
    df_data = df_data.sort_values('AgentType')
    df_data = df_data.reset_index()

    df_data['FirstDeathsRate'] *= 100
    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='Agent', y='FirstDeathsRate',
                hue='AgentType')

    # Set title, axis and adjust text size
    ax.set_title('First Death Rate by Agent', fontsize=20)
    ax.set_xlabel('Agent', fontsize=15)
    ax.set_ylabel('First Death Rate', fontsize=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    file_name = DataDepot.get_result_path() + 'q1.2c_agents_first_dead.png'
    fig.savefig(file_name, bbox_inches='tight')

    # Plot graph for last to survive
    df_data = df_last_survive
    df_data = df_data.sort_values('AgentType')
    df_data = df_data.reset_index()

    df_data['LastSurviveRate'] *= 100
    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='Agent', y='LastSurviveRate',
                hue='AgentType')

    # Set title, axis and adjust text size
    ax.set_title('Last to Survive Rate by Agent', fontsize=20)
    ax.set_xlabel('Agent', fontsize=15)
    ax.set_ylabel('Last Survive Rate', fontsize=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Controller', 'Duelist', 'Initiator', 'Sentinel'],
              title='Agent Role',
              fontsize=12, loc='lower right')

    file_name = DataDepot.get_result_path() + 'q1.2d_agents_last_survive.png'
    fig.savefig(file_name, bbox_inches='tight')

    plt.close(fig)


def analyze_agents_most_kill_dead_assist(df_scoreboard, df_games):
    """
    Analyze the following sub questions.
    Q1-2e. Which agent is more likely to kill the most enemies?
    Q1-2f. Which agent is most likely to be killed by the enemies?
    Q1-2g. Which agent is most likely to assist to kill enemies?
    Q1-2h. Which agent is most likely to plant a bomb?
    Q1-2i. Which agent is most likely to defuse a bomb?

    @param df_scoreboard  The data frame about score board
    @param df_games       The data frame about games
    @return               The result data frame
    """
    # Slice the dataframe to keep only necessary columns
    df_result = df_scoreboard[['GameID', 'Agent', 'AgentType', 'Kills',
                              'Deaths', 'Assists', 'Plants', 'Defuses']]

    # Calculates the probability of the kill/death/assist event based on the
    # total game rounds of each game to prevent the highly selected agent wins
    df_games['TotalRound'] = (df_games['Team1_TotalRounds'] +
                              df_games['Team2_TotalRounds'])
    df_games['TotatEnemy'] = df_games['TotalRound'] * 5
    df_games = df_games[['TotalRound', 'TotatEnemy']]
    df_result = df_result.merge(df_games, left_on=['GameID'],
                                right_on=['GameID'], how='left')

    df_result['KillsRate'] = df_result['Kills'] / df_result['TotatEnemy']
    df_result['DeathsRate'] = df_result['Deaths'] / df_result['TotatEnemy']
    df_result['AssistsRate'] = df_result['Assists'] / df_result['TotatEnemy']
    df_result['PlantsRate'] = df_result['Plants'] / df_result['TotalRound']
    df_result['DefusesRate'] = df_result['Defuses'] / df_result['TotalRound']

    # Group by agent and calculate the average of its percentage
    # Agent   AgentType    FirstKillsRate
    # Astra   Controllers        0.072573
    # Breach  Initiators         0.062302
    # .....
    df_kill = df_result[['Agent', 'AgentType', 'KillsRate']]
    df_kill = df_kill.groupby(['Agent', 'AgentType']).mean()
    df_death = df_result[['Agent', 'AgentType', 'DeathsRate']]
    df_death = df_death.groupby(['Agent', 'AgentType']).mean()
    df_assists = df_result[['Agent', 'AgentType', 'AssistsRate']]
    df_assists = df_assists.groupby(['Agent', 'AgentType']).mean()
    df_plants = df_result[['Agent', 'AgentType', 'PlantsRate']]
    df_plants = df_plants.groupby(['Agent', 'AgentType']).mean()
    df_defuses = df_result[['Agent', 'AgentType', 'DefusesRate']]
    df_defuses = df_defuses.groupby(['Agent', 'AgentType']).mean()

    return (df_kill, df_death, df_assists, df_plants, df_defuses)


def plot_agents_most_kill_dead_assist(df_most_kill, df_most_dead,
                                      df_most_assist, df_most_plant,
                                      df_most_defuse):
    """
    Plot charts with the data frame analyzed in
    analyze_agents_most_kill_dead_assist

    @param df_most_kill    Result data frame. See q1.2e_agents_most_kill.csv
    @param df_most_dead    Result data frame. See q1.2f_agents_most_dead.csv
    @param df_most_assist  Result data frame. See q1.2g_agents_most_assist.csv
    @param df_most_plant   Result data frame. See q1.2h_agents_most_plant.csv
    @param df_most_defuse  Result data frame. See q1.2i_agents_most_defuse.csv
    """

    # Plot most kills
    df_data = df_most_kill
    df_data = df_data.sort_values('AgentType')
    df_data = df_data.reset_index()

    df_data['KillsRate'] *= 100
    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='Agent', y='KillsRate',
                hue='AgentType')

    # Set title, axis and adjust text size
    ax.set_title('Most Kills Rate by Agent', fontsize=20)
    ax.set_xlabel('Agent', fontsize=15)
    ax.set_ylabel('Most Kills Rate', fontsize=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Controller', 'Duelist', 'Initiator', 'Sentinel'],
              title='Agent Role',
              fontsize=12, loc='lower right')

    file_name = DataDepot.get_result_path() + 'q1.2e_agents_most_kill.png'
    fig.savefig(file_name, bbox_inches='tight')

    # Plot most deaths
    df_data = df_most_dead
    df_data = df_data.sort_values('AgentType')
    df_data = df_data.reset_index()

    df_data['DeathsRate'] *= 100
    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='Agent', y='DeathsRate',
                hue='AgentType')

    # Set title, axis and adjust text size
    ax.set_title('Most Deaths Rate by Agent', fontsize=20)
    ax.set_xlabel('Agent', fontsize=15)
    ax.set_ylabel('Most Deaths Rate', fontsize=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Controller', 'Duelist', 'Initiator', 'Sentinel'],
              title='Agent Role',
              fontsize=12, loc='lower right')

    file_name = DataDepot.get_result_path() + 'q1.2f_agents_most_dead.png'
    fig.savefig(file_name, bbox_inches='tight')

    # Plot most assists
    df_data = df_most_assist
    df_data = df_data.sort_values('AgentType')
    df_data = df_data.reset_index()

    df_data['AssistsRate'] *= 100
    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='Agent', y='AssistsRate',
                hue='AgentType')

    # Set title, axis and adjust text size
    ax.set_title('Most Assists Rate by Agent', fontsize=20)
    ax.set_xlabel('Agent', fontsize=15)
    ax.set_ylabel('Most Assists Rate', fontsize=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Controller', 'Duelist', 'Initiator', 'Sentinel'],
              title='Agent Role',
              fontsize=12, loc='upper right')

    file_name = DataDepot.get_result_path() + 'q1.2g_agents_most_assist.png'
    fig.savefig(file_name, bbox_inches='tight')

    # Plot most spike plants
    df_data = df_most_plant
    df_data = df_data.sort_values('AgentType')
    df_data = df_data.reset_index()

    df_data['PlantsRate'] *= 100
    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='Agent', y='PlantsRate',
                hue='AgentType')

    # Set title, axis and adjust text size
    ax.set_title('Most Spike Plants Rate by Agent', fontsize=20)
    ax.set_xlabel('Agent', fontsize=15)
    ax.set_ylabel('Most Spike Plants Rate', fontsize=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Controller', 'Duelist', 'Initiator', 'Sentinel'],
              title='Agent Role',
              fontsize=12, loc='upper left')

    file_name = DataDepot.get_result_path() + 'q1.2h_agents_most_plant.png'
    fig.savefig(file_name, bbox_inches='tight')

    # Plot most spike defuses
    df_data = df_most_defuse
    df_data = df_data.sort_values('AgentType')
    df_data = df_data.reset_index()

    df_data['DefusesRate'] *= 100
    fig, ax = plt.subplots(1, figsize=(16, 6))

    sns.barplot(ax=ax, data=df_data, x='Agent', y='DefusesRate',
                hue='AgentType')

    # Set title, axis and adjust text size
    ax.set_title('Most Spike Defusals Rate by Agent', fontsize=20)
    ax.set_xlabel('Agent', fontsize=15)
    ax.set_ylabel('Most Spike Defusals Rate', fontsize=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set legend label and size again
    ax.legend(ax.get_legend_handles_labels()[0],
              ['Controller', 'Duelist', 'Initiator', 'Sentinel'],
              title='Agent Role',
              fontsize=12, loc='upper left')

    file_name = DataDepot.get_result_path() + 'q1.2i_agents_most_defuse.png'
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

    # Question set 1
    print("- Analyzing question set #1")

    print("- Analyzing question 1 - map")
    analyze_maps(data_depot)

    print("- Analyzing question 1 - agents")
    analyze_agents(data_depot)


if __name__ == '__main__':
    print("Executing here only tests functions in this module with TEST mode.")
    print("To execute the main of the project, Please execute main.py.")

    print("- Initializing data depot")
    data_depot = DataDepot(test_mode=False)
    execute_module(data_depot)
