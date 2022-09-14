# CSE 163 Final Project - ESports(Valorant) Research
Young You (cuperido@uw.edu)
Hao Le (lehao206@uw.edu)


# Setup
In order to run this project, you do not need any additional libraries other than those provided by the CSE163 course.
Below is the full list of libraries used by this project.

* Numpy
* Pandas
* Re
* Math
* Time
* SQLite3
* StringIO
* Matplotlib
* Seaborn
* SKlearn


# File List
| Path        | File                   | Explanation                                               |
|-------------|------------------------|-----------------------------------------------------------|
| /data       |                        | Folder for the original data source                       |
|             | valorant.sqlite        | Original SQL data set                                     |
| /data_cache |                        | Folder for parsed data sets (for caching)                 |
|             | matches.csv            | Match information. A match includes several games.        |
|             | games.csv              | Game information. A game includes several rounds.         |
|             | scoreboard.csv         | Game records of each game.                                |
|             | rounds.csv             | Each round of the game information.                       |
|             | teams.csv              | Team information. A team includes several players.        |
|             | players.csv            | Player identification information.                        |
| /data_test  |                        | Folder for test data sets                                 |
|             | *.csv                  | Test data sets                                            |
| /result     |                        | Folder for result data                                    |
|             | *.csv                  | Data frame results                                        |
|             | *.png                  | Chart results                                             |
| /           |                        |                                                           |
|             | main.py                | The main module of the project                            |
|             | data_depot.py          | Data parsing and caching module                           |
|             | q_helper_functions.py  | Helper module for all questions                           |
|             | q1_pre_strategies.py   | Research module for the first question - Pre-strategies   |
|             | q2_post_strategies.py  | Research module for the second question - Post-strategies |
|             | q3_machine_learning.py | Research module for the third question - Machine Learning |
|             | tester.py              | Tester module for all questions                           |
|             | cse163_utils.py        | Helper module provided by CSE163                          |
|             | README.md              |                                                           |


# Execute the project
There are three options to execute the project.

* 1: Main project
To execute the entire project, please execute `main.py`. It will execute all questions.

* 2: Each question
If you want to execute each question separately, please execute `q1_pre_strategies.py`, `q2_post_strategies.py`, or `q3_machine_learning.py`.

* 3: Tester
If you want to execute the tester module, please execute `tester.py`.


# Dataset

#### Original Data Set
The original dataset was provided as a SQL database file.
The project doesn't use it directly every time due to performance issues because lots of data need to be parsed in advance. 

For more information about the original SQL data set, please refer this site.
https://www.kaggle.com/datasets/visualize25/valorant-pro-matches-full-data


#### Processed Data Set
`data_depot.py` reproduces these CSV files from the SQL dataset and saves them into `/data_cache` folder.
The project reads these files when the project starts.

| File           | Data origin                                                                                           |
|----------------|-------------------------------------------------------------------------------------------------------|
| matches.csv    | Mostly derived from `matches` tables in the SQL database                                              |
| games.csv      | Mostly derived from `games` tables in the SQL database                                                |
| scoreboard.csv | Mostly derived from `game_scoreboard` tables in the SQL database                                      |
| rounds.csv     | Built from the `RoundHistory` serialized string field in the `game_roundss` tables in the SQL database. |
| teams.csv      | Reproduced from `matches`, `games`, `game_rounds` tables in the SQL database                          |
| players.csv    | Reproduced from `game_scoreboard` tables in the SQL database                                          |

To rebuild the data set and use it, call `data_depot.py` class with `rebuild=True` option.

    # main.py :: line 22
    data_depot = DataDepot(rebuild=True)


#### Test Data Set
Test data set for `tester.py` is prepared in `/data_test`.

Although it is similar to the top 20 records of real game records, it has been modified appropriately for the test purpose.
To rebuild the test set, please refer to the doc-comment in the `tester.py` folder.

Since the test set is still complicated for some tasks, it is highly recommended to validate the test value with the spreadsheet test.
The way to validate values using the SpreadSheet filter is described in the comments of `tester.py`.
https://docs.google.com/spreadsheets/d/136dxtK0nFMaxDbPqj14jztwnFK-LmqMQ5OWpbbE8H30

Besides using `tester.py`, you can also run the entire project with test data for quick code validation.
To execute the project with the test data set, call `data_depot.py` class with `test_mode=True` option.

    # main.py :: line 22
    data_depot = DataDepot(test_mode=True)

In test mode, the research results also will be not saved to disk, and the `Rebuild` option will be ignored too.
