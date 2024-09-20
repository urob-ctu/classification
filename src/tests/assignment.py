from dataclasses import dataclass


@dataclass
class Assignment:
    name: str
    test_func: callable
    verification_file: str

    def test_assignment(self, generate: bool, seed: int):
        kwargs = dict(
            seed=seed,
            generate=generate,
            verification_file=self.verification_file,
        )
        results = self.test_func(**kwargs)
        return results
