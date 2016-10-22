def is_sorted(l):
    return all(b <= a for a, b in zip(l, l[1:]))
