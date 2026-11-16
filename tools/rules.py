from tools.curve import avg, bad_numbers, drift

def check(rows, evals, flags, sup):
    """The kill rules that can be judged from the log rows so far: a list of (rule, why)"""
    out = []
    if not rows:
        return out
    last = rows[-1]["round"]
    bad = bad_numbers(rows)
    if bad:
        out.append(("nan", f"{bad['column']} is nan or infinite at round {bad['round']}"))
    d = drift(rows, evals, flags)
    if d:
        out.append(("drift", f"drift fingerprint at rounds {d['rounds']}: entropy {d['entropy']:+.2f}, "
                             f"valid score {d['valid_score']:+.3f} vs rounds 0-4, eval not rising"))
    n = sup["broken_window"]
    share = avg(rows, "broken", last - n + 1, last)
    if last + 1 >= n and share is not None and share > sup["broken_share"]:
        out.append(("broken", f"{share:.0%} of training games broken over the last {n} rounds "
                              f"(limit {sup['broken_share']:.0%})"))
    entropy = avg(rows, "entropy", last - 9, last)
    if last + 1 >= sup["collapse_after"] and entropy is not None and entropy < flags["entropy_collapse"]:
        out.append(("entropy_collapse", f"entropy {entropy:.2f} over the last 10 rounds "
                                        f"(below {flags['entropy_collapse']})"))
    return out

def hang(quiet_s, any_row, sup):
    """The hang rule: log.csv has not grown for too long"""
    limit = sup["hang_s"] if any_row else sup["start_grace_s"]
    if quiet_s > limit:
        return ("hang", f"log.csv has not grown for {quiet_s / 60:.0f} min (limit {limit / 60:.0f} min)")
    return None

def low_ram(free_gb, sup):
    """The memory rule: free RAM is nearly gone"""
    if free_gb < sup["kill_free_ram_gb"]:
        return ("low_ram", f"only {free_gb:.1f} GB of RAM free (limit {sup['kill_free_ram_gb']} GB)")
    return None
