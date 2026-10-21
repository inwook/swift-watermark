# Rewrite Strategies

Each strategy has two equivalent forms, `s0` and `s1`. The watermark picks one per
applicable strategy and function.

## Python

| Strategy | s0 | s1 |
|----------|----|----|
| AugAssign | `a += b` | `a = a + b` |
| CompareSwap | `a < b` | `b > a` |
| Ternary | `x = A if C else B` | `if C: x = A` / `else: x = B` |
| RangeStyle | `range(n)` | `range(0, n)` |
| ListInit | `[]` | `list()` |
| DictInit | `{}` | `dict()` |
| ListCompStyle | `[x for ...]` | `list(x for ...)` |
| ReturnVar | `return expr` | `result = expr; return result` |
| PairAssign | `a = 1; b = 2` | `a, b = 1, 2` |
| NegateIfElse | `if a == b: A else: B` | `if a != b: B else: A` |
| ChainedCompare | `a < x < b` | `a < x and x < b` |
| NotIn | `x not in y` | `not x in y` |
| IsCompare | `x is None` | `x == None` |
| YodaEquals | `x == 5` | `5 == x` |
| DeMorgan | `not a or not b` | `not (a and b)` |
| NegateIfCall | `if f(): A else: B` | `if not f(): B else: A` |

## Java

| Strategy | s0 | s1 | Notes |
|----------|----|----|-------|
| AugAssign | `a += b` (also `-= *= /= %=`) | `a = a + b` | non-`+=` forms need a simple RHS |
| CompareSwap | `a < b` | `b > a` | |
| BooleanReturn | `return a > b;` | `if (a > b) { return true; } else { return false; }` | |
| Ternary | `x = c ? a : b;` | `if (c) { x = a; } else { x = b; }` | skips `x = c ? x : b` |
| NegateIfElse | `if (a == b) { A } else { B }` | `if (a != b) { B } else { A }` | |
| PrefixPostfix | `i++;` | `++i;` | standalone statements only |
| ReturnVar | `return expr;` | `T result = expr; return result;` | `T` is the declared return type |
| YodaEquals | `x == literal` | `literal == x` | |
| WhileForSwap | `while (c) { ... }` | `for (; c;) { ... }` | first convertible loop only |
| OptionalBraces | `if (c) stmt;` | `if (c) { stmt; }` | no else, single-line statement |
| DiamondInference | `new ArrayList<>()` | `new ArrayList<String>()` | local declarations |
| ArrayTypeStyle | `int[] arr` | `int arr[]` | 1-D local declarations |

Methods annotated `@InlineMe` are skipped by every Java strategy.

## JavaScript

| Strategy | s0 | s1 | Notes |
|----------|----|----|-------|
| AugAssign | `a += b` | `a = a + b` | |
| CompareSwap | `a < b` | `b > a` | |
| ArrayInit | `[]` | `new Array()` | empty arrays |
| ObjectInit | `{}` | `new Object()` | empty objects |
| BooleanReturn | `return expr;` | `if (expr) { return true; } else { return false; }` | |
| ArrowFunctionReturn | `=> expr` | `=> { return expr; }` | |
| PrefixPostfix | `i++;` | `++i;` | standalone statements only |
| NegateIfElse | `if (a === b) { A } else { B }` | `if (a !== b) { B } else { A }` | also `==` / `!=` |
| Ternary | `x = c ? a : b;` | `if (c) { x = a; } else { x = b; }` | |
| YodaEquals | `x === literal` | `literal === x` | `===`, `!==`, `==`, `!=` |
| ReturnVar | `return expr;` | `const result = expr; return result;` | top-level function body |
| WhileForSwap | `while (c) { ... }` | `for (; c;) { ... }` | |
| BooleanCast | `!!x` | `Boolean(x)` | |
| PropertyShorthand | `{ x }` | `{ x: x }` | |
| OptionalBraces | `if (c) stmt;` | `if (c) { stmt; }` | no else, single-line statement |
