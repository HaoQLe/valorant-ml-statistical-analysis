"""
Young You / Hao Le
CSE 163 AE

DataTester class reads data CSV from the test folder and conduct tests
analysis methods of each question

To remake a data set, please follow these steps to spread distribution
1) Copy games, matches, rounds, scordboard in /data folter to /data_test
2) Cut only recent 20 games in those data.
3) Change all team ID 4733 to 6436
4) Change the team name of player 'Coach' to 'BJOR'
5) Change the team name of player 'squishys' to 'SQUI'

Since the test set is still complicated for some tasks, we need to validate
the test value with the spreadsheet test.

This is the URL of the spreadsheet for the test set with a value filter.
=> docs.google.com/spreadsheets/d/136dxtK0nFMaxDbPqj14jztwnFK-LmqMQ5OWpbbE8H30
"""

from data_depot import DataDepot
from cse163_utils import assert_equals
import q1_pre_strategies as q1
import q2_post_strategies as q2
import q3_machine_learning as q3


def test_q1_analyze_maps(data_depot, print_result=False):
    """
    Test analyze_maps in q1_pre_strategies.py

    @param data_depot    The DataDepot class (Data dispatcher)
    @param print_result  If true, display the result on screen
    """
    print("Testing q1_pre_strategies::analyze_maps")
    df_first_role, df_winning_type, df_agents = q1.analyze_maps(data_depot)

    # Print all result tables
    if print_result:
        print('\n* RESULT : analyze_maps * \n\n',
              df_first_role, '\n\n', df_winning_type, '\n\n', df_agents)

    assert_equals(10, len(df_first_role))
    assert_equals(20, df_first_role['Count'].sum())
    assert_equals([2, 1, 3, 1, 1, 2, 2, 4, 1, 3],
                  list(df_first_role['Count']))

    # data_test/rounds.csv has 376 records
    # Since it has 5 map types, the sum of percent should 5.0
    assert_equals(376, df_winning_type['Count'].sum())
    assert_equals(5.0, df_winning_type['Percent'].sum())
    df_winning_type_ascent = df_winning_type[df_winning_type.index.
                                             get_level_values('Map').
                                             isin(['Ascent'])]
    assert_equals([2, 14, 40, 2], list(df_winning_type_ascent['Count']))
    assert_equals(1.0, df_winning_type_ascent['Percent'].sum())

    # Since it has 5 map types, the sum of percent should 5.0
    assert_equals(5.0, df_agents['Percent'].sum())

    # Verify specific type of record
    # To calculate it, use the Google Spreadtsheet testset the using filter.
    df_agents_ascent = df_agents[df_agents.index.
                                 get_level_values('Map').
                                 isin(['Ascent'])]
    df_agents_astra = df_agents[df_agents.index.
                                get_level_values('Agent').
                                isin(['Astra'])]
    assert_equals([3, 1, 3, 2, 3, 3], list(df_agents_ascent['Count']))
    assert_equals([3, 4, 1, 6], list(df_agents_astra['Count']))


def test_q1_analyze_agents(data_depot, print_result=False):
    """
    Test analyze_purchases in q1_pre_strategies.py

    @param data_depot    The DataDepot class (Data dispatcher)
    @param print_result  If true, display the result on screen
    """
    print("Testing q1_pre_strategies::analyze_agents")
    df_class, df_first_kill, df_first_dead, df_last_survive, \
        df_most_kill, df_most_dead, df_most_assist, \
        df_most_plant, df_most_defuse = q1.analyze_agents(data_depot)

    # Print all result tables
    if print_result:
        print('\n* RESULT : analyze_agents * \n\n',
              df_class, '\n\n', df_first_kill, '\n\n',
              df_first_dead, '\n\n', df_last_survive, '\n\n',
              df_most_kill, '\n\n', df_most_dead, '\n\n',
              df_most_assist, '\n\n', df_most_plant, '\n\n',
              df_most_defuse, '\n\n')

    # Since the testset 20 games, it should be more than 40 at least.
    assert_equals(True, df_class['Count'].sum() >= 40)

    # Since it has 4 Agent types, the sum of percent should 4.0
    assert_equals(4.0, df_class['Percent'].sum())

    # Verify specific type of record
    df_class_duelists = df_class[df_class.index.
                                 get_level_values('AgentType').
                                 isin(['Duelists'])]
    assert_equals([8, 5], list(df_class_duelists['Count']))

    # There are total 13 agents in the all test set
    assert_equals(13, len(df_first_kill))
    assert_equals(13, len(df_first_dead))
    assert_equals(13, len(df_last_survive))
    assert_equals(13, len(df_most_kill))
    assert_equals(13, len(df_most_dead))
    assert_equals(13, len(df_most_assist))
    assert_equals(13, len(df_most_plant))
    assert_equals(13, len(df_most_defuse))

    # Verify specific type of record. Only calculate with 'Raze'
    # because it has only 10 records and spread widly
    # To calculate it, please use the Google Spreadtsheet testset above.
    # 1) Go 'scoreboard' sheet and filter by Raze. => There are 10 records.
    # 2) Sum all FirstKills records of Raze.
    #    > 4, 4, 1, 1, 3, 7, 3, 4, 0, 2
    # 3) Find all GameIDs in the filtered data
    #    > 60895, 60895, 60889, 60904, 60921, 60921, 60879, 60880, 60882, 60883
    # 4) Go 'games' sheet, filter game id with GameIDs above
    # 5) Get their total round
    #    > 15, 15, 20, 15, 21, 21, 20, 24, 14, 14
    # 6) Calculate the kill rate of each game by dividing the pair in order
    #    > (4/15=0.2667), (4/15=0.2667), (1/20=0.05), ....
    # 7) Calculate the average all kill rates calculated
    #    > 1.585714286 / 10 = 0.158571
    index_tuple = ('Raze', 'Duelists')
    assert_equals(0.1585, df_first_kill.loc[index_tuple])
    assert_equals(0.1520, df_first_dead.loc[index_tuple])
    assert_equals(0.0050, df_last_survive.loc[index_tuple])
    assert_equals(0.1362, df_most_kill.loc[index_tuple])
    assert_equals(0.1748, df_most_dead.loc[index_tuple])
    assert_equals(0.0400, df_most_assist.loc[index_tuple])
    assert_equals(0.0247, df_most_plant.loc[index_tuple])
    assert_equals(0.0097, df_most_defuse.loc[index_tuple])


def test_q2_analyze_purchases(data_depot, print_result=False):
    """
    Test analyze_purchases in q2_post_strategies.py

    @param data_depot    The DataDepot class (Data dispatcher)
    @param print_result  If true, display the result on screen
    """
    print("Testing q2_post_strategies::analyze_purchases")
    df_result = q2.analyze_purchases(data_depot)

    # Print all result tables
    if print_result:
        print('\n* RESULT : analyze_purchases * \n\n', df_result)

    # data_test/rounds.csv has 376 records with 20 games.
    # Because 1st and 13rd round of each game should be removed,
    # the total round should be 672
    # e.g., (376 Rounds - (2 Rounds * 20 Games)) * 2 Teams = 672
    assert_equals(672, df_result['Count'].sum())

    # Since it has 5 purchase types, the sum of percent should 4.0
    assert_equals(4.0, df_result['Percent'].sum())

    # Verify specific type of record
    df_result_fullbuy = df_result[df_result.index.
                                  get_level_values('BuyType').
                                  isin(['full-buy'])]
    assert_equals([164, 238], list(df_result_fullbuy['Count']))
    assert_equals([0.4079, 0.5920], list(df_result_fullbuy['Percent']))


def test_q2_analyze_kills(data_depot, print_result=False):
    """
    Test analyze_kills in q2_post_strategies.py

    @param data_depot    The DataDepot class (Data dispatcher)
    @param print_result  If true, display the result on screen
    """
    print("Testing q2_post_strategies::analyze_kills")
    df_enemy_kills, df_enemy_first_kills, df_enemy_first_rate, df_agent = \
        q2.analyze_kills(data_depot)

    # Print all result tables
    if print_result:
        print('\n* RESULT : analyze_kills * \n\n',
              df_enemy_kills, '\n\n', df_enemy_first_kills, '\n\n',
              df_enemy_first_rate, '\n\n', df_agent, '\n\n')

    # Since data_test/games.csv has 20 game records, it should be 40
    # => 20 Games * 2 Teams = 40 Records
    assert_equals(40, df_enemy_kills['Count'].sum())

    # Since it has 4 True/False distributions, the sum of percent should 4.0
    assert_equals(4.0, df_enemy_kills['Percent'].sum())

    # 11 FirstKills distributions (3.0 ~ 14.0) in test dataset
    # 7 FirstKillRate distributions (2.0 ~ 8.0) in test dataset
    assert_equals(11.0, df_enemy_first_kills['Percent'].sum())

    # There are total 200 scoreboards in the test dataset, and 15 distribution
    # (4 Controllers + 4 Duelists + 4 Initiators + 3 Sentinels = 15)
    assert_equals(200, df_agent['Count'].sum())
    assert_equals(15.0, df_agent['Percent'].sum())


def test_q2_analyze_strategy(data_depot, print_result=False):
    """
    Test analyze_strategy in q2_post_strategies.py

    @param data_depot    The DataDepot class (Data dispatcher)
    @param print_result  If true, display the result on screen
    """
    print("Testing q2_post_strategies::analyze_strategy")
    df_plants, df_defuses = q2.analyze_strategy(data_depot)

    # Print all result tables
    if print_result:
        print('\n* RESULT : analyze_strategy * \n\n',
              df_plants, '\n\n', df_defuses, '\n\n')

    # Since data_test/games.csv has 20 game records, the total is 40
    # However, Both team in the game ID 60883 has no Defuses, it should be 39
    # => 20 Games * 2 Teams - 1 Side of the Game = 39 Records
    assert_equals(39, df_plants['Count'].sum())
    assert_equals(39, df_defuses['Count'].sum())

    # When record is 0, team that planted a boom won 1 record out of 4.
    # also, teams didn't defuse a boom won 3 record out of 9.
    assert_equals(1/4, df_plants.loc[(0, True)]['Percent'])
    assert_equals(3/9, df_defuses.loc[(0, True)]['Percent'])


def test_q3_build_ml_dataset(data_depot, print_result=False):
    """
    Test build_ml_dataset in q3_machine_learning.py

    @param data_depot    The DataDepot class (Data dispatcher)
    @param print_result  If true, display the result on screen
    """
    print("Testing q3_machine_learning::build_ml_dataset")
    df_data = q3.build_ml_dataset(data_depot)

    # Print all result tables
    if print_result:
        print('\n* RESULT : build_ml_dataset * \n\n', df_data)

    df_games = data_depot.get_games()

    # Since it separates the game set by two teams, the total number
    # of record is twice of the total game.
    assert_equals(len(df_games) * 2, len(df_data))

    # Check it has all the original data
    # TotalRounds = [ (Team1 Total Round) + (Team2 Total Round) ] * 2
    assert_equals((df_games['Team1_TotalRounds'].sum() +
                  df_games['Team2_TotalRounds'].sum()) * 2,
                  df_data['TotalRounds'].sum())


def main():
    """
    Main test method. Conduct all tests for analysis questions.
    """
    # Prepare the pre-processed data from test data folder
    data_depot = DataDepot(test_mode=True)

    test_q1_analyze_maps(data_depot, False)
    test_q1_analyze_agents(data_depot, False)

    test_q2_analyze_purchases(data_depot, False)
    test_q2_analyze_kills(data_depot, False)
    test_q2_analyze_strategy(data_depot, False)

    test_q3_build_ml_dataset(data_depot, False)

    print("Test Done.")


if __name__ == '__main__':
    main()
