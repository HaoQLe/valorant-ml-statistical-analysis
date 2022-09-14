"""
Young You / Hao Le
CSE 163 AE

This module is for helper functions that modularize repeated processing.
This module can be called in the processing regardless of the type of the
questions, so do not make the module work only for any specific question.
"""

# Define agent classes
AGENT_CLASSES_INFO = {
    "Duelists": ["Jett", "Phoenix", "Neon", "Raze", "Reyna", "Yoru"],
    "Controllers": ["Astra", "Brimstone", "Omen", "Viper"],
    "Initiators": ["Breach", "Kayo", "Skye", "Sova", "Fade"],
    "Sentinels": ["Chamber", "Cypher", "Killjoy", "Sage"],
}


def agents_class_classifier(row):
    """
    The helper function to classify the class of the agent.
    (Four classes : Duelists / Controllers / Initiators / Sentinels)

    @param row  A row of the data frame
    @return     The class of the agent
    """
    agent = row['Agent']
    if agent in AGENT_CLASSES_INFO['Duelists']:
        return "Duelists"
    elif agent in AGENT_CLASSES_INFO['Controllers']:
        return "Controllers"
    elif agent in AGENT_CLASSES_INFO['Initiators']:
        return "Initiators"
    elif agent in AGENT_CLASSES_INFO['Sentinels']:
        return "Sentinels"
    else:
        return None


def scoreboard_team_record_assigner(df_scoreboard, df_games, df_teams):
    """
    The helper function to find the team's record in the scoreboard data frame.
    This function finds the TeamID from the team data frame and its win record
    the game data frame, and then merge the information.

    For best performance, drop all unnecessary columns in the df_scoreboard
    before calling it. Otherwise, it could be overloaded while merging.

    @param df_scoreboard  The data frame about score board
    @param df_games       The data frame about games
    @param df_teams       The data frame about teams
    @return               The processed score board data frame
    """
    # Slice the dataframe to keep only necessary columns.
    # However, it doesn't slice df_scoreboard because we should return it
    # with keeping all the given columns.
    col_games = ['Team1ID', 'Team2ID', 'WinnerTeamIdx']
    df_games = df_games[col_games]

    col_teams = ['TeamAbbr']
    df_teams = df_teams[col_teams]
    df_teams = df_teams.reset_index()

    # Drop all uncompleted fields
    df_games = df_games.dropna()
    df_teams = df_teams.dropna()

    # Merge team data to find TeamID
    # GameID    Agent TeamAbbr  ...  TeamID
    #  60894     Jett     BOOS  ...    6903
    #  60894  Chamber     BOOS  ...    6903
    # .....
    df_scoreboard = df_scoreboard.merge(df_teams.drop_duplicates('TeamAbbr'),
                                        left_on=['TeamAbbr'],
                                        right_on=['TeamAbbr'], how='left')

    # Merge game data to find TeamIDs and WinnerTeam
    # GameID    Agent TeamAbbr  ...  TeamID  Team1ID  Team2ID  WinnerTeamIdx
    #  60894     Jett     BOOS  ...    6903     6903     6020              1
    #  60894  Chamber     BOOS  ...    6903     6903     6020              1
    # .....
    df_scoreboard = df_scoreboard.merge(df_games, left_on=['GameID'],
                                        right_on=['GameID'], how='left')

    # Remove if team ID is invalid or unknown
    df_scoreboard = df_scoreboard[
                        (df_scoreboard['TeamID'] == df_scoreboard['Team1ID']) |
                        (df_scoreboard['TeamID'] == df_scoreboard['Team2ID'])
                    ]

    # Calculate if the team won
    df_scoreboard['TeamWin'] = df_scoreboard.apply(
        lambda row:
            (row['TeamID'] == row['Team1ID']) if (row['WinnerTeamIdx'] == 1)
            else (row['TeamID'] == row['Team2ID']), axis=1
    )

    return df_scoreboard


if __name__ == '__main__':
    print("This module is not executable.")
    print("To execute the main of the project, Please execute main.py.")
