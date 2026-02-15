from __future__ import annotations
import random
from typing import Any, Dict, List, Tuple, Optional


def generate_addition_batch(
    batch_size: int,
    length: int,
    *,
    seed: Optional[int] = None,
    log: Optional[bool] = False
) -> Tuple[List[str], List[str]]:
    """
    Generate (X, Y) for forward-order binary addition in the paper's format.

    Format:
      X:  a + b > ####...   (fills remaining positions with EOS '#')
      Y:  ***** s ###...    ('*' ignored before EOQ, then sum bits, then trailing '#')

    Where:
      - a and b are length-bit binary strings (leading zeros allowed)
      - s is (length+1)-bit binary sum (leading zero kept)
      - '>' is EOQ, '#' is EOS, '*' is ignored output
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if length <= 0:
        raise ValueError("length must be positive")

    rng = random.Random(seed)

    X: List[str] = []
    Y: List[str] = []

    # Matches the appendix example for addition:
    # For length=3 (3-bit + 3-bit), input has 6 EOS tokens and output ends with 3 EOS tokens. :contentReference[oaicite:2]{index=2}
    num_eos_in_input = length + 3

    for _ in range(batch_size):
        a_int = rng.randrange(0, 1 << length)
        b_int = rng.randrange(0, 1 << length)
        if log:
            print("A: ", a_int)
            print("B: ", b_int)
            print("Y: ", a_int + b_int)
            print("formula: ", f"{a_int} + {b_int} = {a_int + b_int}")

        a = format(a_int, f"0{length}b")
        b = format(b_int, f"0{length}b")
        s = format(a_int + b_int, f"0{length+1}b")  # keep leading 0 if present

        query = f"{a}+{b}>"
        x = query + ("#" * num_eos_in_input)

        # Output: ignore everything up to (but not including) EOQ position, then emit sum bits,
        # then fill remaining positions with EOS '#'.
        stars = "*" * (len(query) - 1)
        total_len = len(x)
        y_core = stars + s
        y = y_core + ("#" * (total_len - len(y_core)))

        X.append(x)
        Y.append(y)

    return X, Y


def decode_addition_xy(x: str, y: str) -> Dict[str, Any]:
    """
    Decode a single (x, y) sample from the paper's addition format into numbers.

    Expected:
      x: "<a_bits>+<b_bits>>########..."
      y: "*****<sum_bits>####..."

    Returns:
      dict with a_bits, b_bits, pred_sum_bits, and integer values.
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length. Got {len(x)} vs {len(y)}")

    plus = x.find("+")
    gt = x.find(">")
    if plus == -1 or gt == -1 or not (0 < plus < gt):
        raise ValueError(f"Bad x format. Expected '<a>+<b>>...'. Got: {x!r}")

    a_bits = x[:plus]
    b_bits = x[plus + 1 : gt]

    if not a_bits or not b_bits or any(c not in "01" for c in (a_bits + b_bits)):
        raise ValueError(f"Non-binary or empty a/b bits in x: a={a_bits!r}, b={b_bits!r}")

    # In the paper format, y has '*' up to (gt - 1), then the predicted sum begins.
    # We read the sum bits from y starting at index (gt - 1) until the first '#'.
    sum_start = gt - 1
    if sum_start < 0 or sum_start >= len(y):
        raise ValueError("Could not locate sum start in y.")

    # Find first EOS after the sum starts
    eos_pos = y.find("#", sum_start)
    if eos_pos == -1:
        eos_pos = len(y)

    pred_sum_bits = y[sum_start:eos_pos]

    # Clean up any stray ignored tokens (shouldn't be there if generated correctly)
    pred_sum_bits = pred_sum_bits.replace("*", "")
    if pred_sum_bits and any(c not in "01" for c in pred_sum_bits):
        raise ValueError(f"Non-binary predicted sum bits in y region: {pred_sum_bits!r}")

    a_int = int(a_bits, 2)
    b_int = int(b_bits, 2)
    true_sum_int = a_int + b_int

    pred_sum_int = int(pred_sum_bits, 2) if pred_sum_bits else None

    return {
        "a_bits": a_bits,
        "b_bits": b_bits,
        "pred_sum_bits": pred_sum_bits,
        "a_int": a_int,
        "b_int": b_int,
        "true_sum_int": true_sum_int,
        "pred_sum_int": pred_sum_int,
        "correct": (pred_sum_int == true_sum_int) if pred_sum_int is not None else False,
        "bit_length": len(a_bits),
        "sum_bit_length": len(pred_sum_bits),
    }


def render_addition_xy(x: str, y: str) -> str:
    """
    Pretty render a single (x, y) sample with real numbers and correctness.
    """
    d = decode_addition_xy(x, y)

    a_bits = d["a_bits"]
    b_bits = d["b_bits"]
    s_bits = d["pred_sum_bits"]

    a = d["a_int"]
    b = d["b_int"]
    s_true = d["true_sum_int"]
    s_pred = d["pred_sum_int"]

    ok = "✅" if d["correct"] else "❌"

    return (
        f"X: {a_bits}+{b_bits}>\n"
        f"Y: (ignored) + {s_bits}\n"
        f"Numbers: {a} + {b} = {s_true}\n"
        f"Model:   {s_pred}   {ok}\n"
    )
    
def render_addition_batch(X: List[str], Y: List[str], max_items: int = 5) -> str:
    lines = []
    for i, (x, y) in enumerate(zip(X, Y)):
        if i >= max_items:
            lines.append(f"... ({len(X) - max_items} more)")
            break
        lines.append(f"[{i}]\n{render_addition_xy(x, y)}")
    return "\n".join(lines)


if __name__ == "__main__":
    X,Y = generate_addition_batch(1,8, log=True)
    print("X: ", X)
    print("Y: ", Y)
    d = render_addition_batch(X, Y, max_items=5)
    print(d)
    