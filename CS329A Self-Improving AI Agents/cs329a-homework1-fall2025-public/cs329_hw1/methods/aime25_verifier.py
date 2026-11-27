from cs329_hw1.tasks.math_utils import (
    strip_string,
    extract_answer,
    math_equal,
    run_with_timeout,
)


class AIME25Verifier:
    """
    Verifier for AIME25 dataset.
    AIME problems typically have integer answers, so we need specialized verification.
    """

    def __init__(self):
        pass

    def verify(
        self, solution: str, ground_truth: str, normalize_prediction=True
    ) -> int:
        """
        Verify if the solution matches the ground truth answer.

        Args:
            solution (str): The model's solution
            ground_truth (str): The correct answer
            normalize_prediction (bool): Whether to normalize the prediction

        Returns:
            int: 1 if correct, 0 if incorrect
        """
        if normalize_prediction:
            # Extract the final answer from the solution
            extracted_answer = strip_string(extract_answer(solution, "math"))
        else:
            extracted_answer = solution

        extracted_gt_answer = ground_truth
        # print("extracted_answer: ", extracted_answer)

        # Use timeout for mathematical equality check. Self-study change: a worker process (run_with_timeout)
        # instead of timeout_decorator's SIGALRM, which Windows lacks; there every answer silently scored 0.
        try:
            return int(run_with_timeout(math_equal, extracted_answer, extracted_gt_answer, timeout=2))
        except TimeoutError:
            return 0
        except Exception as e:
            return 0

    def __call__(self, *args, **kwargs):
        return self.verify(*args, **kwargs)
