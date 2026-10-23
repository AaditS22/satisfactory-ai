from layout.best_of import keep_best, compare


def res(score, A=0.5, B=0.6, kind="valid"):
    return {"score": score, "A": A, "B": B, "kind": kind}


def test_keep_best_per_task():
    tries = [[res(0.1), res(0.7), res(-1.2, kind="failed")],
             [res(0.4), res(0.2), res(0.3)],
             [res(0.3), res(0.6), res(-0.5, kind="invalid")]]
    best = keep_best(tries)
    assert [b["score"] for b in best] == [0.4, 0.7, 0.3]
    assert best[2]["kind"] == "valid"


def test_keep_best_one_try_unchanged():
    one = [res(0.1), res(0.2)]
    assert keep_best([one]) == one


def test_keep_best_does_not_change_input():
    tries = [[res(0.1)], [res(0.9)]]
    keep_best(tries)
    assert tries[0][0]["score"] == 0.1


def test_compare_hand():
    results = [res(0.6, A=0.5, B=0.6), res(0.5, A=0.5, B=0.7), res(0.4, A=0.5, B=0.55),
               res(-1.2, A=-1.25, B=-1.25, kind="failed")]
    s = compare(results)
    assert s["solvable"] == 3
    assert abs(s["beats_A"] - 1 / 3) < 1e-9 and abs(s["ties_A"] - 1 / 3) < 1e-9
    assert abs(s["beats_B"] - 1 / 3) < 1e-9 and s["invalid"] == 0
    assert abs(s["score"] - (0.6 + 0.5 + 0.4) / 3) < 1e-9


def test_compare_counts_invalid_on_solvable_only():
    s = compare([res(-0.4, kind="invalid"), res(0.7), res(-1.3, A=-1.4, kind="failed")])
    assert s["solvable"] == 2 and s["invalid"] == 0.5


TESTS = [test_keep_best_per_task, test_keep_best_one_try_unchanged, test_keep_best_does_not_change_input,
    test_compare_hand, test_compare_counts_invalid_on_solvable_only]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok ", t.__name__)
    print(f"all {len(TESTS)} passed")