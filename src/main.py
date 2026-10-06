# Myers' diff - Assignment 1
#
#   python main.py lines     A B   -> Part A: line diff
#   python main.py highlight A B   -> Part B: line diff + changed character ranges
#
# Flow: read_lines -> diff_pairs -> lcs_pairs -> middle_snake -> build_output
#
# Edit graph:
#   x = position in A, y = position in B.
#   Move right (x+1)      = delete a line of A
#   Move down  (y+1)      = insert a line of B
#   Diagonal   (x+1, y+1) = same line in both, keep it (costs nothing)
#   Diagonal number k = x - y. d = number of edits so far.
#   Goal: go from (0,0) to (N,M) with the smallest d.
import sys


# Read a file and split it into lines, following the rules in Section 1.
def read_lines(path):
    # Open in binary mode. Text mode would turn "\r\n" into "\n".
    with open(path, "rb") as f:
        data = f.read()

    # Split only on b"\n". Any "\r" stays inside the line.
    parts = data.split(b"\n")

    # If the file ends with "\n", the last piece is empty. Remove it.
    # An empty file gives [b""], which becomes [] (no lines).
    if parts and parts[-1] == b"":
        parts.pop()
    return parts


# Find the middle snake of X[a_lo:a_hi] and Y[b_lo:b_hi].
#
# Two searches run at the same time:
#   v1 = forward search from the start. v1[k] = furthest x on diagonal k.
#   v2 = backward search from the end. v2[k] = how far back from the end.
# When the two searches overlap on a diagonal, that point is in the middle
# of a shortest path. We return it as (x, y), relative to the range.
# We return None if X and Y have nothing in common.
#
# This uses O(N+M) memory, because we keep only two V arrays
# instead of saving a copy of V for every d.
def middle_snake(X, Y, a_lo, a_hi, b_lo, b_hi):
    N = a_hi - a_lo
    M = b_hi - b_lo

    # Each search only needs to go half way.
    max_d = (N + M + 1) // 2

    # k can be negative, so we store diagonal k at index off + k.
    off = max_d
    size = 2 * max_d + 2

    # -1 means this diagonal has not been reached yet.
    v1 = [-1] * size
    v2 = [-1] * size

    # Start value, like V[1] = 0 in the paper.
    # It makes the first step (d = 0, k = 0) start at x = 0.
    v1[off + 1] = 0
    v2[off + 1] = 0

    # delta = difference between the forward and backward diagonals.
    # If delta is odd, we check for overlap in the forward pass.
    # If delta is even, we check in the backward pass.
    delta = N - M
    front = (delta % 2 != 0)

    # Used to skip diagonals that went outside the grid (only for speed).
    k1start = k1end = k2start = k2end = 0

    for d in range(max_d):

        # ---------- forward pass ----------
        # For each d, k goes -d, -d+2, ..., d.
        for k1 in range(-d + k1start, d + 1 - k1end, 2):
            i = off + k1

            # Decide how we reach diagonal k:
            #   k == -d: we can only come down from k+1 (insert)
            #   k == d : we can only come right from k-1 (delete)
            #   otherwise: come from the diagonal that got further
            if k1 == -d or (k1 != d and v1[i - 1] < v1[i + 1]):
                x1 = v1[i + 1]          # insert: x stays the same
            else:
                x1 = v1[i - 1] + 1      # delete: x moves one step
            y1 = x1 - k1                # since k = x - y

            # Snake: while the items are equal, move along the diagonal.
            if x1 < N and y1 < M and X[a_lo + x1] == Y[b_lo + y1]:
                xa = a_lo + x1 + 1      # use absolute indexes in the loop
                yb = b_lo + y1 + 1
                while xa < a_hi and yb < b_hi and X[xa] == Y[yb]:
                    xa += 1
                    yb += 1
                x1 = xa - a_lo          # back to relative indexes
                y1 = yb - b_lo

            v1[i] = x1                  # save the furthest x on diagonal k

            if x1 > N:                  # went past the right edge
                k1end += 2
            elif y1 > M:                # went past the bottom edge
                k1start += 2
            elif front:
                # Forward diagonal k1 is backward diagonal delta - k1.
                j = off + delta - k1
                if 0 <= j < size and v2[j] != -1:
                    # v2[j] is counted from the end, so its x is N - v2[j].
                    if x1 >= N - v2[j]:
                        return x1, y1   # the searches overlap

        # ---------- backward pass ----------
        # Same steps as the forward pass, but reading X and Y from the end.
        for k2 in range(-d + k2start, d + 1 - k2end, 2):
            i = off + k2
            if k2 == -d or (k2 != d and v2[i - 1] < v2[i + 1]):
                x2 = v2[i + 1]
            else:
                x2 = v2[i - 1] + 1
            y2 = x2 - k2

            # Snake, going backwards.
            if x2 < N and y2 < M and X[a_hi - 1 - x2] == Y[b_hi - 1 - y2]:
                xa = a_hi - 2 - x2
                yb = b_hi - 2 - y2
                while xa >= a_lo and yb >= b_lo and X[xa] == Y[yb]:
                    xa -= 1
                    yb -= 1
                x2 = a_hi - 1 - xa
                y2 = b_hi - 1 - yb

            v2[i] = x2

            if x2 > N:
                k2end += 2
            elif y2 > M:
                k2start += 2
            elif not front:
                # delta is even, so we check for overlap here.
                j = off + delta - k2
                if 0 <= j < size and v1[j] != -1:
                    x1 = v1[j]
                    y1 = off + x1 - j   # j = off + k, and y = x - k
                    if x1 >= N - x2:
                        return x1, y1

    # No overlap found: nothing in common, so everything is delete + insert.
    return None


# Find all matched pairs (i, j), meaning X[i] and Y[j] are kept.
# This is the longest common subsequence (LCS).
# Edits = len(X) + len(Y) - 2 * len(LCS), so the largest LCS gives the
# smallest diff.
#
# We split the problem at the middle snake and solve both halves.
# We use our own stack instead of recursion, so big inputs do not hit
# Python's recursion limit.
def lcs_pairs(X, Y):
    pairs = []
    stack = [(0, len(X), 0, len(Y))]          # (a_lo, a_hi, b_lo, b_hi)

    while stack:
        a_lo, a_hi, b_lo, b_hi = stack.pop()

        # Equal items at the start are kept.
        while a_lo < a_hi and b_lo < b_hi and X[a_lo] == Y[b_lo]:
            pairs.append((a_lo, b_lo))
            a_lo += 1
            b_lo += 1

        # Equal items at the end are kept.
        while a_lo < a_hi and b_lo < b_hi and X[a_hi - 1] == Y[b_hi - 1]:
            a_hi -= 1
            b_hi -= 1
            pairs.append((a_hi, b_hi))

        # If one side is empty, there is nothing left to match.
        if a_lo == a_hi or b_lo == b_hi:
            continue

        mid = middle_snake(X, Y, a_lo, a_hi, b_lo, b_hi)
        if mid is None:
            continue

        # Split at (x, y) and solve the left and right parts separately.
        x, y = mid
        stack.append((a_lo + x, a_hi, b_lo + y, b_hi))   # right part
        stack.append((a_lo, a_lo + x, b_lo, b_lo + y))   # left part

    # Pairs were found in mixed order, so sort them.
    pairs.sort()
    return pairs


# Diff any two sequences: lists of lines (bytes) or strings (characters).
# Part A and Part B both use this.
def diff_pairs(A, B):
    # Give each different item a small integer id.
    # Comparing integers is faster than comparing long lines.
    ids = {}
    a_ids = [ids.setdefault(x, len(ids)) for x in A]
    b_ids = [ids.setdefault(x, len(ids)) for x in B]

    # An item that appears in only one sequence can never be matched.
    # Removing it first makes the input smaller and does not change the LCS.
    in_a = set(a_ids)
    in_b = set(b_ids)
    a_map = [i for i, v in enumerate(a_ids) if v in in_b]   # new index -> old index
    b_map = [j for j, v in enumerate(b_ids) if v in in_a]
    X = [a_ids[i] for i in a_map]
    Y = [b_ids[j] for j in b_map]

    # Convert the indexes back to the original positions.
    return [(a_map[i], b_map[j]) for i, j in lcs_pairs(X, Y)]


# Turn the kept positions into changed ranges for Part B.
# matched = sorted list of kept positions. The gaps between them are changed.
# Each gap is as large as possible, so ranges never touch or overlap.
def ranges(n, matched):
    out = []
    prev = 0                                  # first position not checked yet
    for idx in matched + [n]:                 # n marks the end of the line
        if idx > prev:                        # positions prev .. idx-1 changed
            out.append("%d-%d" % (prev, idx)) # end is not included
        prev = idx + 1
    return ",".join(out) if out else "."      # nothing changed -> "."


# Build the "? old | new" line for one paired - line and + line.
def highlight_line(old_bytes, new_bytes):
    # Decode to str so each index is one Unicode code point.
    # An emoji counts as 1. "\r" also counts as 1.
    old = old_bytes.decode("utf-8", "surrogateescape")
    new = new_bytes.decode("utf-8", "surrogateescape")

    # Same diff as Part A, but on characters.
    pairs = diff_pairs(old, new)
    old_r = ranges(len(old), [i for i, _ in pairs])
    new_r = ranges(len(new), [j for _, j in pairs])
    return ("? %s | %s\n" % (old_r, new_r)).encode("ascii")


# Build the output text from the matched pairs.
#
# Everything between two matched pairs is one change block:
#   A[i:mi] are deleted, B[j:mj] are inserted.
# We print all "-" lines first and then all "+" lines,
# so the delete-first rule is always followed.
def build_output(A, B, pairs, highlight):
    out = []
    i = j = 0                                 # next unprinted line in A and B

    # Add (len(A), len(B)) at the end so the last block is also printed.
    for mi, mj in pairs + [(len(A), len(B))]:
        dels = A[i:mi]
        ins = B[j:mj]

        for line in dels:
            out.append(b"-" + line + b"\n")

        for k, line in enumerate(ins):
            out.append(b"+" + line + b"\n")
            # Part B: pair the k-th "-" line with the k-th "+" line.
            # Extra lines on either side stay unpaired.
            if highlight and k < len(dels):
                out.append(highlight_line(dels[k], line))

        if mi < len(A):                       # the last entry is not a real match
            out.append(b" " + A[mi] + b"\n")

        i, j = mi + 1, mj + 1                 # continue after the matched line

    # Join once at the end. Adding to a string in a loop can be slow.
    return b"".join(out)


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in ("lines", "highlight"):
        print("usage: main.py lines|highlight A_PATH B_PATH", file=sys.stderr)
        return 2

    command, a_path, b_path = sys.argv[1:]

    # Read both files first. If one cannot be read, print nothing on stdout,
    # print an error on stderr, and exit with code 2.
    try:
        A = read_lines(a_path)
        B = read_lines(b_path)
    except OSError as e:
        print("error: cannot read file: %s" % e, file=sys.stderr)
        return 2

    pairs = diff_pairs(A, B)

    # Write raw bytes, so lines that are not valid UTF-8 print correctly.
    sys.stdout.buffer.write(build_output(A, B, pairs, command == "highlight"))
    sys.stdout.buffer.flush()
    return 0


raise SystemExit(main())