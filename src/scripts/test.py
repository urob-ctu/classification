import os

from tests import Assignment
from tests import test_assignment_1_1, test_assignment_1_2, test_assignment_1_3
from tests import test_assignment_2_1
from tests import (
    test_assignment_3_1,
    test_assignment_3_2,
    test_assignment_3_3,
    test_assignment_3_4,
)
from tests import (
    test_assignment_4_1,
    test_assignment_4_2,
    test_assignment_4_3,
    test_assignment_4_4,
)


SEED = 69
GENERATE = False

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, os.pardir, os.pardir))
SRC_DIR = os.path.join(PROJECT_DIR, "src", "assignments_solution")

ASSIGNMENTS = [
    Assignment(
        name="assignment_1_1",
        test_func=test_assignment_1_1,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_1_1.npy"),
    ),
    Assignment(
        name="assignment_1_2",
        test_func=test_assignment_1_2,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_1_2.npy"),
    ),
    Assignment(
        name="assignment_1_3",
        test_func=test_assignment_1_3,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_1_3.npy"),
    ),
    Assignment(
        name="assignment_2_1",
        test_func=test_assignment_2_1,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_2_1.npy"),
    ),
    Assignment(
        name="assignment_3_1",
        test_func=test_assignment_3_1,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_3_1.pt"),
    ),
    Assignment(
        name="assignment_3_2",
        test_func=test_assignment_3_2,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_3_2.pt"),
    ),
    Assignment(
        name="assignment_3_3",
        test_func=test_assignment_3_3,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_3_3.pt"),
    ),
    Assignment(
        name="assignment_3_4",
        test_func=test_assignment_3_4,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_3_4.pt"),
    ),
    Assignment(
        name="assignment_4_1",
        test_func=test_assignment_4_1,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_4_1.pt"),
    ),
    Assignment(
        name="assignment_4_2",
        test_func=test_assignment_4_2,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_4_2.pt"),
    ),
    Assignment(
        name="assignment_4_3",
        test_func=test_assignment_4_3,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_4_3.pt"),
    ),
    Assignment(
        name="assignment_4_4",
        test_func=test_assignment_4_4,
        verification_file=os.path.join(PROJECT_DIR, "data/tests/assignment_4_4.pt"),
    ),
]


def test():
    print("\n================= TESTING ASSIGNMENTS =================\n")
    for assignment in ASSIGNMENTS:
        results = assignment.test_assignment(
            generate=GENERATE, seed=SEED
        )
        print(f"{assignment.name}: \n\t{results['message']}\n")


if __name__ == "__main__":
    test()
