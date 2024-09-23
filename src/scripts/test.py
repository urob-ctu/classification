import os
import argparse

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

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)


def create_assignments(dataset_dir: str) -> list:
    return [
        Assignment(
            name="assignment_1_1",
            test_func=test_assignment_1_1,
            verification_file=os.path.join(dataset_dir, "assignment_1_1.npy"),
        ),
        Assignment(
            name="assignment_1_2",
            test_func=test_assignment_1_2,
            verification_file=os.path.join(dataset_dir, "assignment_1_2.npy"),
        ),
        Assignment(
            name="assignment_1_3",
            test_func=test_assignment_1_3,
            verification_file=os.path.join(dataset_dir, "assignment_1_3.npy"),
        ),
        Assignment(
            name="assignment_2_1",
            test_func=test_assignment_2_1,
            verification_file=os.path.join(dataset_dir, "assignment_2_1.npy"),
        ),
        Assignment(
            name="assignment_3_1",
            test_func=test_assignment_3_1,
            verification_file=os.path.join(dataset_dir, "assignment_3_1.pt"),
        ),
        Assignment(
            name="assignment_3_2",
            test_func=test_assignment_3_2,
            verification_file=os.path.join(dataset_dir, "assignment_3_2.pt"),
        ),
        Assignment(
            name="assignment_3_3",
            test_func=test_assignment_3_3,
            verification_file=os.path.join(dataset_dir, "assignment_3_3.pt"),
        ),
        Assignment(
            name="assignment_3_4",
            test_func=test_assignment_3_4,
            verification_file=os.path.join(dataset_dir, "assignment_3_4.pt"),
        ),
        Assignment(
            name="assignment_4_1",
            test_func=test_assignment_4_1,
            verification_file=os.path.join(dataset_dir, "assignment_4_1.pt"),
        ),
        Assignment(
            name="assignment_4_2",
            test_func=test_assignment_4_2,
            verification_file=os.path.join(dataset_dir, "assignment_4_2.pt"),
        ),
        Assignment(
            name="assignment_4_3",
            test_func=test_assignment_4_3,
            verification_file=os.path.join(dataset_dir, "assignment_4_3.pt"),
        ),
        Assignment(
            name="assignment_4_4",
            test_func=test_assignment_4_4,
            verification_file=os.path.join(dataset_dir, "assignment_4_4.pt"),
        ),
    ]


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--brute", action="store_true")

    args = parser.parse_args()

    if args.brute:
        dataset_dir = os.path.join(PROJECT_DIR, "data", "brute_tests")
    else:
        dataset_dir = os.path.join(PROJECT_DIR, "data", "tests")

    print("\n================= TESTING ASSIGNMENTS =================\n")
    for assignment in create_assignments(dataset_dir):
        results = assignment.test_assignment(generate=args.generate, seed=args.seed)
        print(f"{assignment.name}: \n\t{results['message']}\n")


if __name__ == "__main__":
    test()
