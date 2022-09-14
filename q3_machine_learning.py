"""
Young You / Hao Le
CSE 163 AE

This module researches the third question.
=> Can machine learning predict a winner based on these records?

[Q3.1] Can ML predict the winning team based only on the information
       we used in Q1 and Q2?
[Q3.2] Among the seven ML models, which one shows the best accuracy
       and which one shows the worst?
       (Seven Models: DecisionTreeClassifier, RandomForestClassifier,
                      KNeighborsClassifier, LogisticRegression, SVC,
                      GradientBoostingClassifier, AdaBoostClassifier)
[Q3.3] Can performance of the lowest-performing model be improved
       by tuning hyperparameters? How much performance can be improved
       with GridSearchCV and RandomizedSearchCV?

The ML data frame consists of these fields.
The data will be made in prepare_ml_dataset() method automatically.
    TotalRounds     : Total rounds of the game.
    Num_Eco         : Total number of rounds in which the team
                      purchased an item with 'eco' mode.
    Num_SemiEco     : Total number of rounds in which the team
                      purchased an item with 'semi-eco' mode.
    Num_SemiBuy     : Total number of rounds in which the team
                      purchased an item with 'semi-buy' mode.
    Num_FullBuy     : Total number of rounds in which the team
                      purchased an item with 'full-buy' mode.
    AttackFirstHalf : First half role (1 - Attack, 0 - Defence)
    Kills           : Total number of enemy kills in entire rounds
    Deaths          : Total number of death in entire rounds
    FirstKills      : Total number of first kills in entire rounds
    FirstDeaths     : Total number of first deaths in entire rounds
    OnevOne         : Total number of 1 vs 1 fight in entire rounds
    OnevTwo         : Total number of 1 vs 2 fight in entire rounds
    OnevThree       : Total number of 1 vs 3 fight in entire rounds
    OnevFour        : Total number of 1 vs 4 fight in entire rounds
    OnevFive        : Total number of 1 vs 5 fight in entire rounds
    Plants          : Total number of attempting to plant bombs
                      in entire rounds
    Defuses         : Total number of defusing bombs in entire rounds
    Controllers     : Total number of Controllers agent class
    Duelists        : Total number of Duelists agent class
    Initiators      : Total number of Initiators agent class
    Sentinels       : Total number of Sentinels agent class
    Map             : Game map name
    TeamWin         : Result whether or not the team won.

When the analysis is complete, this file writes the corresponding results
to the /result folder.
"""

from data_depot import DataDepot
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import re
import time
import pandas as pd
import q_helper_functions as q_helper
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Const information
ML_DATA_SAVE_FILE = "q3.1_ml_data"
ML_TEST_DATASET_RATIO = 0.3

# Model for simple and hyperparam test.
ML_TEST_DEFAULT_MODEL = DecisionTreeClassifier

# Models for comparison test
ML_TEST_MODELS = (DecisionTreeClassifier, RandomForestClassifier,
                  KNeighborsClassifier, LogisticRegression, SVC,
                  GradientBoostingClassifier, AdaBoostClassifier)

# The field list that is excluded the second test.
ML_TEST_EXCLUDED_FIELDS = ['Kills', 'Deaths', 'OnevOne', 'OnevTwo',
                           'OnevThree', 'OnevFour', 'OnevFive']

# Test repeat count. If it is 0, test will be skipped.
ML_TEST_MODELS_TEST_COUNT = 0
ML_TEST_HYPER_PARAMETERS_TEST_COUNT = 0


def build_ml_dataset(data_depot, ex_field=[]):
    """
    build a machine learning dataset in the following steps.

    1) Exclude all records that be able to assume identification of past
       records, such as PlayerID, PlayerName, and Team.
    2) To prove our analytics of questions 1 and 2, only remain the fields
       that we used for those questions.
    3) Make a new data frame by joining related tables.
    4) Shuffle the data set randomly
    5) One-hot encoding for string fields, such as Agent, AgentType, and Role.

    The new data frame will be saved to 'result/[ML_DATA_SAVE_FILE].csv'

    @param data_depot  The DataDepot class (Data dispatcher)
    @param ex_field    The fields that need to be excluded addtionally
    """
    df_scoreboard = data_depot.get_scoreboard()
    df_games = data_depot.get_games()
    df_teams = data_depot.get_teams()

    # Slice the dataframe to keep only necessary columns
    col_scoreboard = ['GameID', 'TeamAbbr', 'Agent', 'Kills', 'Deaths',
                      'FirstKills', 'FirstDeaths', 'OnevOne', 'OnevTwo',
                      'OnevThree', 'OnevFour', 'OnevFive',
                      'Plants', 'Defuses']
    df_scoreboard = df_scoreboard[col_scoreboard]

    col_games = ['MatchID', 'Team1ID', 'Team2ID', 'Map',
                 'Team1_TotalRounds', 'Team2_TotalRounds',
                 'Team1_SideFirstHalf', 'Team2_SideFirstHalf',
                 'Team1_Eco', 'Team2_Eco',
                 'Team1_SemiEco', 'Team2_SemiEco',
                 'Team1_SemiBuy', 'Team2_SemiBuy',
                 'Team1_FullBuy', 'Team2_FullBuy',
                 'WinnerTeamIdx']
    df_games = df_games[col_games]

    # Assign team's id into the scoreboard
    df_scoreboard = q_helper.scoreboard_team_record_assigner(
        df_scoreboard, df_games, df_teams)

    # Assign agent type
    df_scoreboard['AgentType'] = \
        df_scoreboard.apply(q_helper.agents_class_classifier, axis=1)

    # Make a scoreboard with the sum values for each team
    df_scoreboard_sum = df_scoreboard.groupby(['GameID', 'TeamID']).sum()

    # Remove very old records because it has lots of missing information
    df_games = df_games[df_games['MatchID'] > 2000]

    # Calculate total rounds
    df_games['TotalRounds'] = \
        df_games['Team1_TotalRounds'] + df_games['Team2_TotalRounds']

    # Seperate games dataframe into two teams
    df_games_1 = df_games.rename(
        columns={"Team1ID": "TeamID", "Team1_SideFirstHalf": "SideFirstHalf",
                 "Team1_Eco": "Num_Eco", "Team1_SemiEco": "Num_SemiEco",
                 "Team1_SemiBuy": "Num_SemiBuy",
                 "Team1_FullBuy": "Num_FullBuy"})
    df_games_2 = df_games.rename(
        columns={"Team2ID": "TeamID", "Team2_SideFirstHalf": "SideFirstHalf",
                 "Team2_Eco": "Num_Eco", "Team2_SemiEco": "Num_SemiEco",
                 "Team2_SemiBuy": "Num_SemiBuy",
                 "Team2_FullBuy": "Num_FullBuy"})

    col_games = ['TeamID', 'Map', 'TotalRounds',
                 'SideFirstHalf', "Num_Eco", "Num_SemiEco", "Num_SemiBuy",
                 "Num_FullBuy", "WinnerTeamIdx"]
    df_games_1 = df_games_1[col_games]
    df_games_2 = df_games_2[col_games]

    # change winner team information according to each team
    df_games_1['TeamWin'] = df_games_1.apply(
        lambda row: 1 if (row['WinnerTeamIdx'] == 1) else 0, axis=1
    )
    df_games_2['TeamWin'] = df_games_2.apply(
        lambda row: 1 if (row['WinnerTeamIdx'] == 2) else 0, axis=1
    )

    # Combine two game tables
    df_games = pd.concat([df_games_1, df_games_2])

    # Convert text information to index if it is replacable
    df_games['AttackFirstHalf'] = df_games.apply(
        lambda row: 1 if (row['SideFirstHalf'] == 'attack') else 0, axis=1
    )

    # Remove necessary columns
    col_scoreboard = ['Kills', 'Deaths', 'FirstKills', 'FirstDeaths',
                      'OnevOne', 'OnevTwo', 'OnevThree', 'OnevFour',
                      'OnevFive', 'Plants', 'Defuses']
    df_scoreboard_sum = df_scoreboard_sum[col_scoreboard]

    # Group by agent type
    # GameID TeamID AgentType    Agent
    # 60879  4680   Controllers      1
    #               Duelists         2
    # .....
    df_scoreboard_agent = \
        df_scoreboard[['GameID', 'TeamID', 'AgentType', 'Agent']]
    df_scoreboard_agent = \
        df_scoreboard_agent.groupby(['GameID', 'TeamID', 'AgentType']).count()

    # Now makt it a pivot table to swap row and column.
    # GameID TeamID  Controllers  Duelists  Initiators  Sentinels
    # 60879  4680            1.0       2.0         1.0        1.0
    #        6436            1.0       1.0         2.0        1.0
    # .....
    df_scoreboard_agent = \
        df_scoreboard_agent.pivot_table(
            'Agent', ['GameID', 'TeamID'], 'AgentType')

    # Merge the agent info table to the sum table
    df_scoreboard_sum = \
        df_scoreboard_sum.merge(df_scoreboard_agent,
                                left_on=['GameID', 'TeamID'],
                                right_on=['GameID', 'TeamID'], how='left')

    # Fill NaN with 0 (i.e. No AgentType => 0)
    df_scoreboard_sum = df_scoreboard_sum.fillna(0)

    # Merge the sum tables to the game table
    df_games = df_games.merge(df_scoreboard_sum,
                              left_on=['GameID', 'TeamID'],
                              right_on=['GameID', 'TeamID'], how='left')

    # Remove all identification info
    df_games = df_games.reset_index()
    df_games = df_games.drop(
        columns=['GameID', 'TeamID', 'SideFirstHalf', 'WinnerTeamIdx'])

    # Only take valid records
    df_games = df_games[(df_games['Kills'] > 0) | (df_games['Deaths'] > 0)]
    df_games = df_games[(df_games['TotalRounds'] > 0)]
    df_games = df_games[(df_games['Num_Eco'] > 0) |
                        (df_games['Num_SemiEco'] > 0) |
                        (df_games['Num_SemiBuy'] > 0) |
                        (df_games['Num_FullBuy'] > 0)]

    # Remove given fields for additional tests.
    if len(ex_field) > 0:
        df_games = df_games.drop(columns=ex_field)

    # Shuffle data and do one-hot encoding
    df_games = shuffle(df_games)
    df_data = pd.get_dummies(df_games)

    # Write the result to the disk
    data_depot.to_csv(df_data, ML_DATA_SAVE_FILE, save_index=False)

    return(df_data)


def prepare_ml_datasets(df_data):
    """
    Prepare datasets for machine learning.

    @param df_data   The pre-processed machine learning data set
    @return          The machine learning data set.
                     (features_train, features_test, labels_train, labels_test)
    """
    features = df_data.loc[:, df_data.columns != 'TeamWin']
    labels = df_data['TeamWin']

    return train_test_split(features, labels, test_size=ML_TEST_DATASET_RATIO)


def train_and_predict_model(model, data):
    """
    Train dataset and do predict.

    @param model The machine learning model object.
    @param data  The machine learning data set.
                 (features_train, features_test, labels_train, labels_test)
    @return      The accuracy result tuple
                 (Trainnig set accuracy, Test set accuracy)

    """
    features_train, features_test, labels_train, labels_test = data
    model.fit(features_train, labels_train)

    # Make a prediction and calculate the accuracy
    train_predictions = model.predict(features_train)
    train_acc = accuracy_score(labels_train, train_predictions)

    test_predictions = model.predict(features_test)
    test_acc = accuracy_score(labels_test, test_predictions)

    return (train_acc, test_acc)


def compare_ml_models_with_default_hyper_params(data_depot, df_data,
                                                models, type, trials):
    """
    Conduct machine learning with a several of models with default
    hyper_parameters

    @param data_depot  The DataDepot class (Data dispatcher)
    @param df_data     The pre-processed machine learning data set
    @param models      ML models for test
    @param type        Test type information. (For filename)
    @param trials      The total trial count
    @return            The accuracy result list
                       Tuple of (Name, Trainnig accuracy, Test accuracy)
    """
    result = []

    # LogisticRegression shows warning a message due to lack of params.
    # It could be removed with a setting like solver='liblinear',
    # but it is worth considering this test is for the same
    # default parameter conditions.
    for model_class in models:
        # Make a model with default parameter.
        model = model_class()
        model_name = re.sub(r'\W+', '', str(model))

        for iter in range(0, trials):
            # Make data set. We should change it every test trial
            ml_data = prepare_ml_datasets(df_data)

            # Test model accuracy
            print(f"Testing ML {model_name} model. {iter + 1}/{trials}")
            acc_train, acc_test = train_and_predict_model(model, ml_data)
            result.append((model_name, acc_train, acc_test))

    # Write the result to the disk
    if trials > 0:
        df_result = pd.DataFrame.from_records(
            result, columns=['Model', 'AccTrain', 'AccTest'])

        data_depot.to_csv(
            df_result, F"q3.2_ml_models_accuracy_{type}",
            save_index=False)
    else:
        print("- Test is skipped. Set the global const",
              "ML_TEST_MODELS_TEST_COUNT to greater than 0",
              "to do ML model comparison test.")
        df_result = pd.read_csv(DataDepot.get_result_path() +
                                F"q3.2_ml_models_accuracy_{type}.csv")

    # Plot the results
    plot_ml_models_with_default_hyper_params(df_result, type)

    return result


def plot_ml_models_with_default_hyper_params(df_result, type):
    """
    Plot charts with the data frame analyzed
    in compare_ml_models_with_default_hyper_params

    @param df_result  Result data frame. See q3.2_ml_models_accuracy.csv
    @param type       Test type information. (For filename)
    """
    df_data = df_result
    df_data = df_data.reset_index()
    record_count = len(df_data.groupby('Model').count())

    # Order the models
    model_order = ['DecisionTreeClassifier', 'RandomForestClassifier',
                   'KNeighborsClassifier', 'LogisticRegression', 'SVC',
                   'GradientBoostingClassifier', 'AdaBoostClassifier']

    # Make a plot for AccTrain
    if(record_count > 1):
        fig, ax = plt.subplots(1, figsize=(16, 4))
        sns.stripplot(ax=ax, x="Model", y="AccTrain", data=df_data,
                      order=model_order, dodge=True)
        sns.boxplot(ax=ax, x="Model", y="AccTrain", data=df_data,
                    order=model_order, boxprops=dict(alpha=.25))

        plt.title(F"Train Accuracy for each ML Model ({type})",
                  pad=30, fontsize=15)
    else:
        fig, ax = plt.subplots(1, figsize=(6, 3))
        sns.stripplot(ax=ax, x="Model", y="AccTrain", data=df_data, dodge=True)
        sns.boxplot(ax=ax, x="Model", y="AccTrain", data=df_data,
                    boxprops=dict(alpha=.25))

        plt.title(F"Train Accuracy ({type})", pad=30, fontsize=15)

    # Set y-axis to percent format
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    plt.xlabel('Models', labelpad=15)
    plt.ylabel('Train Accuracy Percentage', labelpad=15)

    file_name = \
        DataDepot.get_result_path() + \
        F"q3.2_ml_models_accuracy_{type}_train.png"
    fig.savefig(file_name, bbox_inches='tight')

    # Make a plot for AccTest
    if(record_count > 1):
        fig, ax = plt.subplots(1, figsize=(16, 4))
        sns.stripplot(ax=ax, x="Model", y="AccTest", data=df_data,
                      order=model_order, dodge=True)
        sns.boxplot(ax=ax, x="Model", y="AccTest", data=df_data,
                    order=model_order, boxprops=dict(alpha=.25))

        plt.title(F"Test Accuracy for each ML Model ({type})",
                  pad=30, fontsize=15)
    else:
        fig, ax = plt.subplots(1, figsize=(6, 3))
        sns.stripplot(ax=ax, x="Model", y="AccTest", data=df_data, dodge=True)
        sns.boxplot(ax=ax, x="Model", y="AccTest", data=df_data,
                    boxprops=dict(alpha=.25))

        plt.title(F"Test Accuracy ({type})", pad=30, fontsize=15)

    # Set y-axis to percent format
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    plt.xlabel('Models', labelpad=15)
    plt.ylabel('Test Accuracy Percentage', labelpad=15)

    file_name = \
        DataDepot.get_result_path() + \
        F"q3.2_ml_models_accuracy_{type}_test.png"
    fig.savefig(file_name, bbox_inches='tight')

    plt.close(fig)


def adjust_ml_hyper_parameters(data_depot, df_data, trials):
    """
    This methods adjust hyper parameters for DecisionTreeClassifier,
    and return its result.

    To find that, it uses GridSearchCV and RandomizedSearchCV.

    @param data_depot  The DataDepot class (Data dispatcher)
    @param df_data     The pre-processed machine learning data set
    @param trials      The total trial count
    @return            The accuracy result list
                       Tuple of (Name, Param, Estimator, Accuracy Score)
    """
    result = []

    # Use DecisionTreeClassifier model
    model = ML_TEST_DEFAULT_MODEL()

    for iter in range(0, trials):
        # Make a ML dataset from data frame
        features_train, features_test, labels_train, labels_test = \
            prepare_ml_datasets(df_data)

        # 1. Use GridSearchCV first.
        # Show a status message
        print(f"- Testing GridSearchCV. {iter + 1}/{trials}")
        start = time.time()

        params = {
            'criterion': ['entropy', 'gini'],
            'max_depth': [3, 5, 10, 20, 30],
            'max_leaf_nodes': [3, 5, 10, 20, 30],
            'min_samples_leaf': [2, 5, 10, 20],
            'min_samples_split': [2, 5, 10, 20],
        }

        # Estimate it with 3 folds and full CPU.
        grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy',
                               cv=3, n_jobs=-1)
        grid_cv.fit(features_train, labels_train)

        result.append(('GridSearchCV', grid_cv.best_params_,
                      grid_cv.best_score_, time.time() - start))

        # 2. Use RandomizedSearchCV
        # Show a status message
        print(f"- Testing RandomizedSearchCV. {iter + 1}/{trials}")
        start = time.time()

        params = {
            'criterion': ['entropy', 'gini'],
            'max_depth': range(3, 31, 3),
            'max_leaf_nodes': range(3, 31, 3),
            'min_samples_leaf': range(2, 21, 2),
            'min_samples_split': range(2, 21, 2),
        }

        # Estimate 100 times with 3 folds and full CPU.
        rand_cv = RandomizedSearchCV(model, param_distributions=params,
                                     n_iter=100, cv=3, n_jobs=-1)
        rand_cv.fit(features_train, labels_train)

        result.append(('RandomizedSearchCV', rand_cv.best_params_,
                      rand_cv.best_score_, time.time() - start))

    # Write the result to the disk
    if trials > 0:
        df_result = pd.DataFrame.from_records(
            result, columns=['CV Type', 'Best Param', 'Accuracy', 'Time'])

        data_depot.to_csv(df_result, 'q3.3_ml_models_hyperparams',
                          save_index=False)
    else:
        print("- Test is skipped. Set the global const",
              "ML_TEST_HYPER_PARAMETERS_TEST_COUNT to greater than 0",
              "to do ML hyper parameter tests")
        df_result = pd.read_csv(DataDepot.get_result_path() +
                                "q3.3_ml_models_hyperparams.csv")

    # Plot the results
    plot_ml_hyper_parameters(df_result)

    return result


def plot_ml_hyper_parameters(df_result):
    """
    Plot charts with the data frame analyzed in adjust_ml_hyper_parameters

    @param df_result  Result data frame. See q3.3_ml_models_hyperparams.csv
    """
    df_data = df_result
    df_data = df_data.reset_index()

    # Make a plot for Accuracy
    fig, ax = plt.subplots(1, figsize=(16, 4))
    sns.stripplot(ax=ax, x="CV Type", y="Accuracy", data=df_data, dodge=True)
    sns.boxplot(ax=ax, x="CV Type", y="Accuracy", data=df_data,
                boxprops=dict(alpha=.25))

    # Set y-axis to percent format
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    plt.title("Train Accuracy for each CV Model", pad=30, fontsize=15)
    plt.xlabel('CV Type', labelpad=15)
    plt.ylabel('Train Accuracy Percentage', labelpad=15)

    file_name = \
        DataDepot.get_result_path() + 'q3.3_ml_models_hyperparams_accuracy.png'
    fig.savefig(file_name, bbox_inches='tight')

    # Make a plot for Time
    fig, ax = plt.subplots(1, figsize=(16, 4))
    sns.stripplot(ax=ax, x="CV Type", y="Time", data=df_data, dodge=True)
    sns.boxplot(ax=ax, x="CV Type", y="Time", data=df_data,
                boxprops=dict(alpha=.25))

    plt.title("Test Time for each CV Model", pad=30, fontsize=15)
    plt.xlabel('CV Type', labelpad=15)
    plt.ylabel('Test Time (sec)', labelpad=15)

    file_name = \
        DataDepot.get_result_path() + 'q3.3_ml_models_hyperparams_time.png'
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

    print("- Analyzing question set #3")

    print("- Analyzing question 3 - build ml dataset (first)")
    # If it is the test mode, it just read the pre-processed CSV file.
    if not data_depot.is_test_mode():
        df_data = build_ml_dataset(data_depot)
    else:
        df_data = pd.read_csv(DataDepot.get_result_path() +
                              f"{ML_DATA_SAVE_FILE}.csv")

    # Single ML test - First test
    print("- Analyzing question 3 - signle ml test (first)")
    compare_ml_models_with_default_hyper_params(
        data_depot, df_data, [ML_TEST_DEFAULT_MODEL],
        'signle_first', ML_TEST_MODELS_TEST_COUNT)

    # Compare ML models - First test
    print("- Analyzing question 3 - compare ml models (first)")
    compare_ml_models_with_default_hyper_params(
        data_depot, df_data, ML_TEST_MODELS,
        'comparison_first', ML_TEST_MODELS_TEST_COUNT)

    print("- Analyzing question 3 - build ml dataset (second)")
    # If it is the test mode, it just read the pre-processed CSV file.
    if not data_depot.is_test_mode():
        df_data = build_ml_dataset(data_depot, ML_TEST_EXCLUDED_FIELDS)

    # Single ML test - Second test (exclude some fields)
    print("- Analyzing question 3 - signle ml test (first)")
    compare_ml_models_with_default_hyper_params(
        data_depot, df_data, [ML_TEST_DEFAULT_MODEL],
        'signle_second', ML_TEST_MODELS_TEST_COUNT)

    # Compare ML models - Second test (exclude some fields)
    print("- Analyzing question 3 - compare ml models (second)")
    compare_ml_models_with_default_hyper_params(
        data_depot, df_data, ML_TEST_MODELS,
        'comparison_second', ML_TEST_MODELS_TEST_COUNT)

    # Compare CV models
    print("- Analyzing question 3 - adjust hyper parameters",
          "(It will take a long time)")
    adjust_ml_hyper_parameters(
        data_depot, df_data, ML_TEST_HYPER_PARAMETERS_TEST_COUNT)


if __name__ == '__main__':
    print("Executing here only tests functions in this module.")
    print("To execute the main of the project, Please execute main.py.")

    print("- Initializing data depot")
    data_depot = DataDepot(test_mode=False)
    execute_module(data_depot)
