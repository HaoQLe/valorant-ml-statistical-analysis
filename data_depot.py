"""
Young You / Hao Le
CSE 163 AE

Because it might take too much time to process all the data for each
execution, the necessary data could be pre-processed by this class.

This class reads the original SQL database and conducts the following tasks.
1. Retrieve all data from the SQL database
2. Remove broken fields from data frames, such as NaN or floating-point error.
3. Create a new data frame by parsing a field in which the 2-dimension
   JavaScript array object is stored as a serialized string.
4. Unify the inconsistent columns of each table.
   (e.g., some tables used TeamAbbr but some used Team Fullname.
5. Merge tables to create new data by tracking information, such as Rounds,
   Teams and Players.
6. Save pre-processed data frames to CSV files on disk so that the project
   reads and uses them directly without handling the SQL database.

This class is not intended to carry out data operation operations but
only to merge and organize data. Do not conduct calculation-related work here
unless it is a very slow task that must require pre-processing.

If you want to rebuild cache CSV files, create the class with 'rebuild=True'.

The pre-processed data will be saved to:
    ~/matches.csv     :  Match information. A match includes several games.
    ~/games.csv       :  Game information. A game includes several rounds.
    ~/scoreboard.csv  :  Game records of each game.
    ~/rounds.csv      :  Each round of the game information.
    ~/teams.csv       :  Team information. A team includes several players.
    ~/players.csv     :  Player identification information.
"""

import pandas as pd
import re
import math
import sqlite3
from io import StringIO

# Path information
PATH_DATA = "./data/"
PATH_DATA_CACHE = "./data_cache/"
PATH_DATA_TEST = "./data_test/"
PATH_RESULT = "./result/"
FILE_MATCHES = "matches.csv"
FILE_GAMES = "games.csv"
FILE_ROUNDS = "rounds.csv"
FILE_TEAMS = "teams.csv"
FILE_PLAYERS = "players.csv"
FILE_SCOREBOARD = "scoreboard.csv"


class DataDepot:
    """
    DataDepot class integrates and manages all datasets.
    It builds pre-processed data and stores all the data in the Pandas
    data fields so that it sends quickly when it needed.
    """

    def __init__(self, test_mode=False, rebuild=False):
        """
        The constructor of the DataDepot class.
        Prepare the pre-processed data, read the data, and ready it for use.

        If test_mode is True. it reads data from PATH_DATA_TEST folder.
        If rebuild is True, it rebuilds the processed CSV data from sqlite and
        save them into PATH_DATA_CACHE folder. It doesn't work in test mode.

        @param test_mode  Test flag (True = Test Mode / False = Normal Mode)
        @param rebuild    If true, rebuilds all the pre-processed CSV data
        """
        # Initialize test mode flag.
        self._test_mode = test_mode
        if test_mode is True:
            print("Note: The data depot is open in the TEST mode")
        else:
            if rebuild is True:
                # Connect the SQL database
                self._database = sqlite3.connect(f"{PATH_DATA}valorant.sqlite")

        # Initialize all data fields
        df_matches = self._prepare_matches(rebuild)
        df_games = self._prepare_games(rebuild)
        df_rounds = self._prepare_rounds(rebuild)
        df_teams = self._prepare_teams(df_rounds, rebuild)
        df_players = self._prepare_players(df_teams, rebuild)
        df_scoreboard = self._prepare_scoreboard(rebuild)

        # Assign all data fields to the global
        self._data_matches = df_matches
        self._data_games = df_games
        self._data_rounds = df_rounds
        self._data_teams = df_teams
        self._data_players = df_players
        self._data_scoreboard = df_scoreboard

    def _prepare_matches(self, rebuild):
        """
        Pre-process all matches data and return processed data frame

        @param rebuild  If true, forcibly rebuilds the pre-processed data
        @return         The processed data frame
        """
        print("Preparing match datasets")
        output_file = self._get_preprocessed_path(FILE_MATCHES)

        # Return the cacheed CSV file unless rebuild is True.
        # Always use cache & test files in the test mode.
        if self._test_mode or (not rebuild):
            return pd.read_csv(output_file, index_col='MatchID')

        # Retrieve records through the database
        df_src_matches = pd.read_sql_query("SELECT * FROM matches",
                                           self._database)

        # Slice the dataframe to keep only necessary columns
        df_matches = df_src_matches[['MatchID', 'EventID', 'Team1ID',
                                     'Team2ID', 'Team1', 'Team2',
                                     'Team1_MapScore', 'Team2_MapScore']]
        df_matches = df_matches.set_index('MatchID')

        # Set some type to int. Otherwise it will make an error on merging.
        df_matches.index = df_matches.index.astype(int)
        df_matches['Team1ID'] = df_matches['Team1ID'].astype(int)
        df_matches['Team2ID'] = df_matches['Team2ID'].astype(int)

        # Save pre-processed table into the file
        df_matches.to_csv(output_file, index=True)

        # To match datatype we will use, reload CSV and use it.
        # Otherwise, it can make a different result because datatype read from
        # SQL database and CSV are different
        return self._prepare_matches(rebuild=False)

    def _prepare_games(self, rebuild):
        """
        Pre-process all games data and return processed data frame

        @param rebuild  If true, forcibly rebuilds the pre-processed data
        @return         The processed data frame
        """
        print("Preparing game datasets")
        output_file = self._get_preprocessed_path(FILE_GAMES)

        # Return the cacheed CSV file unless rebuild is True.
        # Always use cache & test files in the test mode.
        if self._test_mode or (not rebuild):
            return pd.read_csv(output_file, index_col='GameID')

        # Retrieve records through the database
        df_src_games = pd.read_sql_query("SELECT * FROM games", self._database)

        # Slice the dataframe to keep only necessary columns
        fields = ['GameID', 'MatchID', 'Map', 'Team1ID', 'Team2ID', 'Team1',
                  'Team2', 'Winner', 'Team1_TotalRounds', 'Team2_TotalRounds',
                  'Team1_SideFirstHalf', 'Team2_SideFirstHalf',
                  'Team1_RoundsFirstHalf', 'Team1_RoundsSecondtHalf',
                  'Team1_RoundsOT', 'Team2_RoundsFirstHalf',
                  'Team2_RoundsSecondtHalf', 'Team2_RoundsOT',
                  'Team1_PistolWon', 'Team1_Eco', 'Team1_EcoWon',
                  'Team1_SemiEco', 'Team1_SemiEcoWon', 'Team1_SemiBuy',
                  'Team1_SemiBuyWon', 'Team1_FullBuy', 'Team1_FullBuyWon',
                  'Team2_PistolWon', 'Team2_Eco', 'Team2_EcoWon',
                  'Team2_SemiEco', 'Team2_SemiEcoWon', 'Team2_SemiBuy',
                  'Team2_SemiBuyWon', 'Team2_FullBuy', 'Team2_FullBuyWon']
        df_games = df_src_games[fields]
        df_games = df_games.set_index('GameID')

        # Make a new field for winner team index
        df_games['WinnerTeamIdx'] = df_games.apply(
            lambda row: '1' if (row['Winner'] == row['Team1']) else '2', axis=1
        )

        # Set some type to int. Otherwise it will make an error on merging.
        df_games.index = df_games.index.astype(int)
        df_games['MatchID'] = df_games['MatchID'].astype(int)
        df_games['Team1ID'] = df_games['Team1ID'].astype(int)
        df_games['Team2ID'] = df_games['Team2ID'].astype(int)

        # Save pre-processed table into the file
        df_games.to_csv(output_file, index=True)

        # To match datatype we will use, reload CSV and use it.
        # Otherwise, it can make a different result because datatype read from
        # SQL database and CSV are different
        return self._prepare_games(rebuild=False)

    def _prepare_rounds(self, rebuild):
        """
        Pre-process all rounds data and return processed data frame.
        It builds a rounds data frame by parsing 2-dim string object fields.

        @param rebuild  If true, forcibly rebuilds the pre-processed data
        @return         The processed data frame
        """
        print("Preparing gameround datasets")
        output_file = self._get_preprocessed_path(FILE_ROUNDS)

        # Return the cacheed CSV file unless rebuild is True.
        # Always use cache & test files in the test mode.
        if self._test_mode or (not rebuild):
            return pd.read_csv(output_file)

        # Retrieve records through the database
        df_src_rounds = pd.read_sql_query("SELECT * FROM game_rounds",
                                          self._database)

        # Make a custom header
        csv = 'GameID,Team1ID,Team2ID,WinnerTeam,'
        csv += 'Team1Score,Team2Score,WinnerTeamIdx,WinType,'
        csv += 'Team1Bank,Team2Bank,Team1BuyType,Team2BuyType'

        cnt_process = 0

        # Unfortunately, this 2-dimemsion object doesn't follow formal JSON
        # and uses both integer and string key, so it cannot be converted by
        # simple commands such as json.load, pickle.loads, and ast.literal_eval
        for index, row in df_src_rounds.iterrows():
            # It includes broken data, so the type must be verified.
            round_info = row['RoundHistory']
            if round_info == '' or not isinstance(round_info, str):
                continue

            # Build a common data for all round information
            head = f"\n{int(row['GameID'])},{row['Team1ID']},{row['Team2ID']}"

            # Remove the first and last two paranthesis
            round_info = round_info[1:-2]

            # Initialize last round score of each team
            # To calculate winning team, it compares the last round score
            team1_score = 0
            team2_score = 0

            # Iterate each round record in the game record
            list_rounds = round_info.split('}')
            for round_row in list_rounds:
                start_pos = round_row.find('{') + 1
                round_row = round_row[start_pos:]

                # Add common header part into the row
                csv_line = head

                # Initialize key index
                key_index = 0

                # Iterate each value in each round
                list_values = round_row.split(',')
                for key_and_value in list_values:
                    value_pos = key_and_value.find(':') + 1
                    value = key_and_value[value_pos:]

                    # Handle additional work according to the key
                    if key_index == 0 or key_index == 2 or \
                       key_index == 5 or key_index == 6:
                        # RoundWinner / WinType / Team1BuyType / Team2BuyType
                        # Remove white space and quotes
                        value = re.sub(r"[ ']", '', value)
                    elif key_index == 1:
                        # ScoreAfterRound
                        # Seperate data for calculating for each team
                        value = value.replace("'", '').replace('-', ',')

                        # Check which team won by comparing the last record
                        scores = value.split(',')
                        if int(scores[0]) > team1_score:
                            win_team_id = 1
                            team1_score = int(scores[0])
                        elif int(scores[1]) > team2_score:
                            win_team_id = 2
                            team2_score = int(scores[1])
                        else:
                            win_team_id = 0

                        value = f"{value},{win_team_id}"
                    elif key_index == 3 or key_index == 4:
                        # Team1Bank / Team2Bank
                        # Since it has floating point error, convert it to int.
                        value = math.floor(float(value))

                    # Append into the csv data
                    csv_line += f",{value}"

                    key_index += 1

                # Append csv data
                csv += csv_line

            # Print a status message
            cnt_process = cnt_process + 1
            if cnt_process % 1000 == 0:
                print(f"Rounds: {cnt_process} records are processed.")

        # Make data field from CSV memory
        df = pd.read_csv(StringIO(csv), sep=",")

        # Save pre-processed table into the file
        df.to_csv(output_file, index=False)

        # To match datatype we will use, reload CSV and use it.
        # Otherwise, it can make a different result because datatype read from
        # SQL database and CSV are different
        return self._prepare_rounds(rebuild=False)

    def _prepare_teams(self, df_rounds, rebuild):
        """
        Build a team data frame by gathering information from several tables

        @param df_rounds  The pre_processed data frame for rounds
        @param rebuild    If true, forcibly rebuilds the pre-processed data
        @return           The processed data frame for team
        """
        print("Preparing team datasets")
        output_file = self._get_preprocessed_path(FILE_TEAMS)

        # Return the cacheed CSV file unless rebuild is True.
        # Always use cache & test files in the test mode.
        if self._test_mode or (not rebuild):
            return pd.read_csv(output_file, index_col='TeamID')

        # Retrieve records through the database
        df_src_games = pd.read_sql_query("SELECT * FROM games", self._database)

        # Take each team data from the game table
        df_all_team1 = df_src_games[['Team1ID', 'Team1']]
        df_all_team2 = df_src_games[['Team2ID', 'Team2']]

        # Combine two tables
        df_all_team1.columns = ['TeamID', 'TeamName']
        df_all_team2.columns = ['TeamID', 'TeamName']
        df_all_teams = pd.concat([df_all_team1, df_all_team2])

        # Remove white spaces of the team name
        df_all_teams['TeamName'] = df_all_teams['TeamName'].str.strip()

        # Group by team id so that it remains unique team ID
        df_all_teams = df_all_teams.groupby('TeamID').first()

        # Now, take each team abbr from the round table
        df_all_team1_abbr = df_rounds[df_rounds['WinnerTeamIdx'] == 1]
        df_all_team1_abbr = df_all_team1_abbr[['Team1ID', 'WinnerTeam']]
        df_all_team2_abbr = df_rounds[df_rounds['WinnerTeamIdx'] == 2]
        df_all_team2_abbr = df_all_team2_abbr[['Team2ID', 'WinnerTeam']]

        # Combine two tables
        df_all_team1_abbr.columns = ['TeamID', 'TeamAbbr']
        df_all_team2_abbr.columns = ['TeamID', 'TeamAbbr']
        df_all_teams_abbr = pd.concat([df_all_team1_abbr, df_all_team2_abbr])

        # Make TeamID as an integer type and group by team id
        df_all_teams_abbr['TeamID'] = df_all_teams_abbr['TeamID'].astype(int)
        df_all_teams_abbr = df_all_teams_abbr.groupby('TeamID').first()

        # Left join two tables
        df_teams = df_all_teams.merge(df_all_teams_abbr, left_on='TeamID',
                                      right_on='TeamID', how='left')

        # Set index type to int. Otherwise it will make an error on merging.
        df_teams.index = df_teams.index.astype(int)

        # Sort in reverse order to find the most recent data first.
        # The latest one will be used if names are duplicated while merging.
        df_teams = df_teams.sort_index(ascending=False)

        # Save pre-processed table into the file
        df_teams.to_csv(output_file, index=True)

        # To match datatype we will use, reload CSV and use it.
        # Otherwise, it can make a different result because datatype read from
        # SQL database and CSV are different
        return self._prepare_teams(None, rebuild=False)

    def _prepare_players(self, df_teams, rebuild):
        """
        Pre-process all players data and return processed data frame

        @param df_teams  The pre_processed data frame for teams
        @param rebuild   If true, forcibly rebuilds the pre-processed data
        @return          The processed data frame
        """
        print("Preparing player datasets")
        output_file = self._get_preprocessed_path(FILE_PLAYERS)

        # Return the cacheed CSV file unless rebuild is True.
        # Always use cache & test files in the test mode.
        if self._test_mode or (not rebuild):
            return pd.read_csv(output_file, index_col='PlayerID')

        # Retrieve records through the database
        df_src_score = pd.read_sql_query("SELECT * FROM game_scoreboard",
                                         self._database)

        # Slice the dataframe to keep only necessary columns
        fields = ['PlayerID', 'PlayerName', 'TeamAbbreviation']
        df_src_score = df_src_score[fields]
        df_src_score.columns = ['PlayerID', 'PlayerName', 'TeamAbbr']

        # Remove white spaces and change ID as an integer type
        df_src_score['PlayerName'] = df_src_score['PlayerName'].str.strip()
        df_src_score['TeamAbbr'] = df_src_score['TeamAbbr'].str.upper()

        # Group by player id so that it remains unique player ID
        df_src_score = df_src_score.dropna(subset=['PlayerID'])
        df_src_score['PlayerID'] = df_src_score['PlayerID'].astype(int)
        df_src_score = df_src_score.groupby('PlayerID', as_index=False).first()

        # Left join two tables
        df_teams = df_teams.reset_index()
        df_players = df_src_score.merge(df_teams.drop_duplicates('TeamAbbr'),
                                        left_on='TeamAbbr',
                                        right_on='TeamAbbr', how='left')
        df_players = df_players.set_index('PlayerID')

        # Sort in reverse order to find the most recent data first.
        # The latest one will be used if names are duplicated while merging.
        df_players = df_players.sort_index(ascending=False)

        # Save pre-processed table into the file
        df_players.to_csv(output_file, index=True)

        # To match datatype we will use, reload CSV and use it.
        # Otherwise, it can make a different result because datatype read from
        # SQL database and CSV are different
        return self._prepare_players(None, rebuild=False)

    def _prepare_scoreboard(self, rebuild):
        """
        Pre-process all scoreboard data and return processed data frame

        @param rebuild  If true, forcibly rebuilds the pre-processed data
        @return         The processed data frame
        """
        print("Preparing scoreboard datasets")
        output_file = self._get_preprocessed_path(FILE_SCOREBOARD)

        # Return the cacheed CSV file unless rebuild is True.
        # Always use cache & test files in the test mode.
        if self._test_mode or (not rebuild):
            return pd.read_csv(output_file)

        # Retrieve records through the database
        df_src_score = pd.read_sql_query("SELECT * FROM game_scoreboard",
                                         self._database)

        # Slice the dataframe to keep only necessary columns
        fields = ['GameID', 'PlayerID', 'PlayerName', 'TeamAbbreviation',
                  'Agent', 'ACS', 'Kills', 'Deaths', 'Assists', 'PlusMinus',
                  'KAST_Percent', 'ADR', 'HS_Percent', 'FirstKills',
                  'FirstDeaths', 'FKFD_PlusMinus', 'Num_2Ks', 'Num_3Ks',
                  'Num_4Ks', 'Num_5Ks', 'OnevOne', 'OnevTwo', 'OnevThree',
                  'OnevFour', 'OnevFive', 'Econ', 'Plants', 'Defuses']

        df_score = df_src_score[fields]

        # Uppercase team abbr
        df_score = df_score.rename(columns={"TeamAbbreviation": "TeamAbbr"})
        df_score['TeamAbbr'] = df_score['TeamAbbr'].str.upper()
        df_score['Agent'] = df_score['Agent'].str.capitalize()

        # Set some type to int. Otherwise it will make an error on merging.
        df_score['GameID'] = df_score['GameID'].astype(int)

        # Save pre-processed table into the file
        df_score.to_csv(output_file, index=False)

        # To match datatype we will use, reload CSV and use it.
        # Otherwise, it can make a different result because datatype read from
        # SQL database and CSV are different
        return self._prepare_scoreboard(rebuild=False)

    def get_matches(self):
        """
        Return data frame for match information
        It returns copied data frame to avoid unexpected modifying dataset.

        @return  The copied data frame about matches
        """
        return self._data_matches.copy()

    def get_games(self):
        """
        Return data frame for game information
        It returns copied data frame to avoid unexpected modifying dataset.

        @return  The copied data frame about games
        """
        return self._data_games.copy()

    def get_rounds(self):
        """
        Return data frame for round information
        It returns copied data frame to avoid unexpected modifying dataset.

        @return  The copied data frame about rounds
        """
        return self._data_rounds.copy()

    def get_teams(self):
        """
        Return data frame for team information
        It returns copied data frame to avoid unexpected modifying dataset.

        @return  The copied data frame about teams
        """
        return self._data_teams.copy()

    def get_players(self):
        """
        Return data frame for player information
        It returns copied data frame to avoid unexpected modifying dataset.

        @return  The copied data frame about players
        """
        return self._data_players.copy()

    def get_scoreboard(self):
        """
        Return data frame for scoreboard information
        It returns copied data frame to avoid unexpected modifying dataset.

        @return  The copied data frame about scoreboard
        """
        return self._data_scoreboard.copy()

    def get_result_path(self=None):
        """
        Return the path of the result to disk

        @return  The path of the result
        """
        return PATH_RESULT

    def is_test_mode(self):
        """
        Return whether or not it is the test mode now

        @return  Return True if test mode. Otherwise, return False
        """
        return self._test_mode

    def to_csv(self, data_frame, file_name, save_index=True,
               test_mode_save=False):
        """
        Save the given data frame to a CSV file to disk.
        If it is in the test mode, it doesn't save the file unless
        the parameter 'test_mode_save' is True.

        @param data_frame      The data frame to save
        @param file_name       The filename of the CSV file
        @param save_index      Whether or not saving indexes as well
        @param test_mode_save  Save data even if it is in the test mode
        """
        if not test_mode_save and self.is_test_mode():
            return

        data_frame.to_csv(f"{PATH_RESULT}{file_name}.csv", index=save_index)

    def _get_preprocessed_path(self, file_type):
        """
        Return the path for preprocessed data files.
        If test mode is on, it returns PATH_DATA_TEST.
        Otherwise, it returns PATH_DATA_CACHE.

        @param file_type  The type of the CSV file
        @return           The path for preprocessed data files
        """
        # No test set for player and team because they only have strings
        if file_type == FILE_TEAMS or \
           file_type == FILE_PLAYERS:
            return PATH_DATA_CACHE + file_type

        if self._test_mode is True:
            return PATH_DATA_TEST + file_type
        else:
            return PATH_DATA_CACHE + file_type


if __name__ == '__main__':
    print("This is not an executable module. Please execute main.py.")
