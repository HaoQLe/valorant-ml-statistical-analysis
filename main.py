"""
Young You / Hao Le
CSE 163 AE

This is the main module of the final project.
It dispatches all questions and saves its results.
"""

from data_depot import DataDepot
import q1_pre_strategies as q1
import q2_post_strategies as q2
import q3_machine_learning as q3


def main():
    """
    Main method of the final project
    It dispatches all questions and saves its results.
    """
    # Initialize the data depot class
    print("- Initializing data depot")
    data_depot = DataDepot(test_mode=False, rebuild=False)

    # Question set 1
    q1.execute_module(data_depot)

    # Question set 2
    q2.execute_module(data_depot)

    # Question set 3
    q3.execute_module(data_depot)

    # Done
    print("- Done.")


if __name__ == '__main__':
    main()
