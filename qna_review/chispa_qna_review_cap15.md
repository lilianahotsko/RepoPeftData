# Generated QA dataset review

## split: cr_test  (15 commits)

### MrPowers/chispa @ commit #0 `a75eeb69b0` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** This commit introduces the initial package structure for `chispa` and adds a column-comparison helper in `chispa/column_comparer.py`, along with a local `SparkSession` instance in `chispa/spark.py` for working with PySpark DataFrames.

- **[new_api] Q:** What new function is introduced in `chispa/column_comparer.py`, and what does it do?
  **A:** The new function is `assert_column_equality(df, col_name1, col_name2)`. It selects the two named columns from a DataFrame, collects them to Python lists, compares the lists for equality, and prints each pair of values, marking mismatches with the prefix `MISMATCH:` when the columns differ.

- **[signature_change] Q:** What is the signature of the newly added column comparison function, and what parameters does it accept?
  **A:** The function is defined as `assert_column_equality(df, col_name1, col_name2)`, so it accepts three positional parameters: a DataFrame-like object `df` and two column names, `col_name1` and `col_name2`.

- **[behavior_change] Q:** How does the runtime behavior of column comparison work after this change when the two columns are not equal?
  **A:** If the collected values from `col_name1` and `col_name2` are not equal, the function zips the two lists together and prints each pair. Matching pairs are printed as `value | value`, while differing pairs are printed as `MISMATCH: left | right`.

- **[refactor] Q:** Why does the new column comparison code appear to be structured as a simple print-based implementation instead of raising an exception?
  **A:** The file includes a commented-out block showing a prior Scala-style exception-based approach (`ColumnMismatch` with `ArrayPrettyPrint.showTwoColumnString`), but the active Python implementation was simplified to print the compared values instead of throwing.

- **[dependency] Q:** What external Spark-related dependency is added in `chispa/spark.py`, and how is it used?
  **A:** `chispa/spark.py` imports `SparkSession` from `pyspark.sql` and uses it to build a session with `SparkSession.builder.master("local").appName("chispa").getOrCreate()`.

- **[scope] Q:** Which files are added by this commit, and what does each one contain?
  **A:** This commit adds four files: `README.md` with the project title `# chispa`, `chispa/__init__.py` as an empty package initializer, `chispa/column_comparer.py` containing `assert_column_equality`, and `chispa/spark.py` containing a `SparkSession` named `spark`.

- **[usage] Q:** How would a developer use the newly added Spark session and column comparison helper from this commit?
  **A:** A developer would import `spark` from `chispa.spark` to get a local `SparkSession`, then call `assert_column_equality(df, col_name1, col_name2)` from `chispa.column_comparer` on a DataFrame to compare two columns.

- **[edge_case] Q:** What does `assert_column_equality` do when the two selected columns contain identical values row by row?
  **A:** When the collected lists are equal, the function does not enter the mismatch-printing block, so it produces no comparison output for that case.

- **[risk] Q:** What compatibility or regression risk is introduced by the current column comparison implementation?
  **A:** The function only prints differences and does not raise an exception, so existing tests or callers that expect a failing assertion on mismatched columns will not automatically fail.

### MrPowers/chispa @ commit #4 `d766794336` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the dataframe and column comparison helpers?
  **A:** It improves equality assertions by adding an approximate column comparison helper and by making schema mismatch errors more informative with a row-by-row PrettyTable diff.

- **[new_api] Q:** What new function was introduced in `chispa/column_comparer.py`, and what does it do?
  **A:** The new function is `assert_approx_column_equality(df, col_name1, col_name2, precision)`, which compares two selected columns row by row and treats values as equal when their absolute difference is less than `precision`; otherwise it raises `ColumnsNotEqualError` with a PrettyTable of the mismatched rows.

- **[signature_change] Q:** Did any existing function signature or type annotation change in this diff, and what is the relevant API shape after the change?
  **A:** No existing function signature was changed in the diff. The new API added is `assert_approx_column_equality(df, col_name1, col_name2, precision)`, which takes a DataFrame, two column names, and a precision threshold.

- **[behavior_change] Q:** How does the runtime behavior of equality checks differ after this change when columns or schemas do not match?
  **A:** For schema mismatches, `assert_df_equality` now raises `SchemasNotEqualError` with a PrettyTable showing `schema1` versus `schema2` instead of the fixed message `These schemas aren't equal`. For approximate column comparisons, rows whose numeric difference is below `precision` are treated as matching and highlighted, while larger differences mark the comparison as failing.

- **[refactor] Q:** What existing construct was simplified in `assert_column_equality`, and why does the new form matter?
  **A:** The repeated inline construction of colored row values was refactored into local variables `first` and `second` before calling `t.add_row([first, second])`. This makes the code shorter and easier to read without changing the output.

- **[dependency] Q:** Which imported functionality is effectively relied on more heavily after this change, and for what purpose?
  **A:** `PrettyTable` is now used to format detailed mismatch output in both `assert_column_equality` and `assert_df_equality`, turning failures into table-based diffs instead of plain text messages.

- **[scope] Q:** Which files were modified in this commit, and what does each one change?
  **A:** `chispa/column_comparer.py` was updated to refactor the colored row construction and add `assert_approx_column_equality`. `chispa/dataframe_comparer.py` was updated so `assert_df_equality` produces a PrettyTable diff when schemas differ.

- **[usage] Q:** How would a developer use the new approximate column comparison helper added by this commit?
  **A:** They would call `assert_approx_column_equality(df, col_name1, col_name2, precision)` on a DataFrame to compare two columns with a numeric tolerance, for example when slight floating-point differences should still count as equal.

- **[edge_case] Q:** What threshold rule does the new approximate comparison use for values near the precision boundary?
  **A:** It uses a strict less-than check: `abs(elements[0] - elements[1]) < precision`. Values exactly equal to `precision` are treated as unequal and will cause `ColumnsNotEqualError` if any row fails the test.

- **[risk] Q:** What compatibility risk does the new schema error formatting introduce for callers that inspect exception text?
  **A:** Callers that relied on the old fixed message `These schemas aren't equal` may break, because `assert_df_equality` now raises `SchemasNotEqualError` with a multi-line PrettyTable string instead.

### MrPowers/chispa @ commit #8 `10845509b6` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the README update in this commit?
  **A:** The README was expanded from a short setup section into a full usage-oriented introduction for chispa, adding examples that show how to test PySpark transformations with `assert_column_equality` and `assert_df_equality` and describing the library as PySpark testing helper methods with descriptive error messages.

- **[new_api] Q:** Which chispa helper functions are introduced in the README examples, and what do they demonstrate?
  **A:** The README introduces `assert_column_equality` from `chispa.column_comparer` for comparing one DataFrame column against another column, and `assert_df_equality` from `chispa.dataframe_comparer` for checking whether two DataFrames are equal.

- **[signature_change] Q:** What code-level call pattern is shown for the column comparison helper in the new README examples?
  **A:** The example calls `assert_column_equality(df, "clean_name", "expected_name")`, showing the helper is used with a DataFrame plus two column-name arguments.

- **[behavior_change] Q:** How does the README say the test output differs when a comparison fails?
  **A:** It says failed comparisons return a nicely formatted error message with highlighted rows: in the column comparison example the mismatching row is shown in red while equal rows are blue, and the DataFrame comparison example also produces a formatted error image.

- **[refactor] Q:** How was the README structurally reworked in this commit?
  **A:** The file was restructured from a brief setup block into multiple documentation sections, including "Column equality", "DataFrame equality", "Approximate column equality", "Approximate DataFrame equality", "Schema error messages", "Benchmarks", "Limitations", "Developing on your local machine", and "Contributing".

- **[dependency] Q:** What import-related changes are shown in the README examples, and why are they needed?
  **A:** The examples add `from chispa.column_comparer import *` and `from chispa.dataframe_comparer import *` so the README can demonstrate the comparison helpers, alongside `import pyspark.sql.functions as F` for `F.col` and `F.regexp_replace` in the sample transformation.

- **[scope] Q:** Which parts of the repository are affected by this commit, and what kind of change does each part receive?
  **A:** Only `README.md` is changed in the diff, and it receives documentation updates: a new product description, setup replacement, code examples for column and DataFrame equality, screenshots of failure output, and placeholder headings for additional topics.

- **[usage] Q:** How does the README show a developer should use chispa to test a string-cleaning transformation?
  **A:** It shows creating a SparkSession, building a DataFrame with source and expected values, adding a cleaned column with `remove_non_word_characters(F.col("name"))`, and then asserting equality with either `assert_column_equality(df, "clean_name", "expected_name")` or `assert_df_equality(actual_df, expected_df)`.

- **[edge_case] Q:** What edge-case input does the new README examples include in the equality tests?
  **A:** Both the passing and failing examples include `None` values in the input data, and the expected output also contains `None`, showing the documented tests cover null handling in the sample transformation.

- **[risk] Q:** What compatibility or maintenance risk does this README update introduce for users following the docs?
  **A:** The README now references additional helpers and sections that are not implemented in the diff itself, such as the empty headings for approximate equality and schema error messages, so readers may expect documented functionality or examples that are not yet fully shown here.

### MrPowers/chispa @ commit #12 `293dce310c` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the dataframe and schema comparison utilities?
  **A:** It refactors the comparison helpers to operate on schema objects directly instead of DataFrame objects, by extracting `df.schema` in `chispa/dataframe_comparer.py` and changing `assert_schema_equality` / `assert_schema_equality_ignore_nullable` in `chispa/schema_comparer.py` to accept schema inputs.

- **[new_api] Q:** What new way does `assert_schema_equality` support for callers after this change, and what does it compare?
  **A:** `assert_schema_equality` now accepts two schema objects, `s1` and `s2`, and compares them directly; if they differ, it builds a `PrettyTable` showing the two schemas side by side and raises `SchemasNotEqualError`.

- **[signature_change] Q:** How did the function signatures change in `chispa/schema_comparer.py`?
  **A:** `assert_schema_equality` changed from `assert_schema_equality(df1, df2)` to `assert_schema_equality(s1, s2)`, and `assert_schema_equality_ignore_nullable` changed from `assert_schema_equality_ignore_nullable(df1, df2)` to `assert_schema_equality_ignore_nullable(s1, s2)`.

- **[behavior_change] Q:** How does runtime behavior differ when schema comparison fails after this commit?
  **A:** The schema comparison tables now use `six.moves.zip_longest`, so mismatched schema lengths are represented in the error table instead of truncating at the shorter sequence length, making missing fields visible in the failure output.

- **[refactor] Q:** Why was the schema extraction logic moved out of the schema comparison functions?
  **A:** The schema comparison helpers were simplified to focus only on comparing schemas, while `chispa/dataframe_comparer.py` now extracts `df1.schema` and `df2.schema` before calling them, reducing duplicated schema-access logic.

- **[dependency] Q:** What new import was added to support the schema comparison changes, and what is it used for?
  **A:** `import six` was added in `chispa/schema_comparer.py` so the code can use `six.moves.zip_longest` when iterating over schema fields.

- **[scope] Q:** Which files are changed in this commit, and what does each file change?
  **A:** `chispa/dataframe_comparer.py` now assigns `df1.schema` and `df2.schema` to local variables and passes schemas into the equality helpers. `chispa/schema_comparer.py` changes the helper signatures to accept schemas directly, adds `six`, and replaces plain `zip` with `six.moves.zip_longest` in schema iteration.

- **[usage] Q:** How should a developer call the updated schema equality helpers after this change?
  **A:** A developer should pass schema objects instead of DataFrames, for example `assert_schema_equality(df1.schema, df2.schema)` or `assert_schema_equality_ignore_nullable(df1.schema, df2.schema)`.

- **[edge_case] Q:** What edge case is handled by using `zip_longest` in the schema comparison code?
  **A:** When two schemas have different numbers of fields, `six.moves.zip_longest` preserves the unmatched trailing field from the longer schema so the failure table can show the mismatch instead of silently dropping it.

- **[risk] Q:** What backward-compatibility concern does this refactor introduce for existing callers?
  **A:** Code that previously called `assert_schema_equality(df1, df2)` or `assert_schema_equality_ignore_nullable(df1, df2)` with DataFrames will now be incompatible, because these functions expect schema objects after the change.

### MrPowers/chispa @ commit #17 `d13560da47` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the DataFrame comparison code?
  **A:** It extends `chispa.dataframe_comparer.assert_df_equality` so callers can pass a list of `transforms` that are applied to both DataFrames before their schemas and rows are compared.

- **[new_api] Q:** What new capability was added to `assert_df_equality` in `chispa/dataframe_comparer.py`, and what does it do?
  **A:** A new `transforms` parameter was added to `assert_df_equality(df1, df2, ignore_nullable=False, transforms=[])`; the function now uses `functools.reduce` to apply each transform function in `transforms` to `df1` and `df2` before performing equality checks.

- **[signature_change] Q:** How did the signature of `assert_df_equality` change in `chispa/dataframe_comparer.py`?
  **A:** Its signature changed from `assert_df_equality(df1, df2, ignore_nullable=False)` to `assert_df_equality(df1, df2, ignore_nullable=False, transforms=[])`, adding a fourth optional `transforms` argument with a default empty list.

- **[behavior_change] Q:** How does the runtime behavior of `assert_df_equality` differ after this change?
  **A:** Before comparing schemas or rows, `assert_df_equality` now transforms both input DataFrames by running every function in `transforms` over them in order, so equality is checked on the transformed DataFrames instead of the originals.

- **[refactor] Q:** Why was `assert_df_equality` restructured around `reduce` in this commit?
  **A:** It was restructured to apply an arbitrary sequence of DataFrame transformation functions in a compact way, using `reduce(lambda acc, fn: fn(acc), transforms, df)` to thread each DataFrame through the transform pipeline.

- **[dependency] Q:** What import was added to support the new comparison behavior, and why is it needed?
  **A:** `from functools import reduce` was added in `chispa/dataframe_comparer.py` so the function can fold the `transforms` list over each DataFrame.

- **[scope] Q:** Which files were touched in this commit, and what kind of change did each receive?
  **A:** `chispa/dataframe_comparer.py` was modified to add the `transforms` parameter and apply it with `reduce`; `README.md` was updated with a CI badge; and a new `LICENSE.md` file was added containing the MIT license text.

- **[usage] Q:** How would a developer use the new `transforms` argument when calling `assert_df_equality`?
  **A:** They would pass a list of functions to `assert_df_equality`, for example `assert_df_equality(df1, df2, transforms=[fn1, fn2])`, and each function will be applied to both DataFrames before comparison.

- **[edge_case] Q:** What default value and handling does the new `transforms` parameter have when no transforms are supplied?
  **A:** `transforms` defaults to `[]`, and `reduce` over an empty list leaves each DataFrame unchanged because the original DataFrame is used as the initial accumulator.

- **[risk] Q:** What compatibility or correctness risk is introduced by this change in `assert_df_equality`?
  **A:** Adding a mutable default value (`transforms=[]`) can be a backward-compatibility and state-sharing risk if the list is ever mutated, and the new transform pipeline also means any function in `transforms` will affect both inputs before schema and row comparison.

### MrPowers/chispa @ commit #22 `b1d5cef931` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in `chispa/dataframe_comparer.py`?
  **A:** It adds reusable support for generic DataFrame equality checks and new ordering options, so callers can ignore column order and row order while comparing DataFrames, and it also refactors approximate and NaN-aware comparisons to use the new generic helper.

- **[new_api] Q:** What new helper function was introduced in `chispa/dataframe_comparer.py`, and what does it do?
  **A:** The new function is `assert_generic_df_equality(df1, df2, row_equality_fun, row_equality_fun_args)`. It asserts schema equality, collects rows, compares them with a caller-supplied row comparison function, and raises `DataFramesNotEqualError` with a pretty-printed diff when rows do not match.

- **[signature_change] Q:** How did the signature of `assert_df_equality` change in `chispa/dataframe_comparer.py`?
  **A:** `assert_df_equality` changed from `assert_df_equality(df1, df2, ignore_nullable=False, transforms=[], allow_nan_equality=False)` to `assert_df_equality(df1, df2, ignore_nullable=False, transforms=None, allow_nan_equality=False, ignore_column_order=False, ignore_row_order=False)`, adding two boolean flags and changing `transforms` to default to `None`.

- **[behavior_change] Q:** How does the runtime behavior of `assert_df_equality` differ after this change when `ignore_column_order` or `ignore_row_order` is enabled?
  **A:** When `ignore_column_order` is `True`, the function appends a transform that selects columns in sorted name order. When `ignore_row_order` is `True`, it appends a transform that sorts each DataFrame by `df.columns`. These transforms are applied before schema and row comparison, allowing equality to ignore column or row ordering.

- **[refactor] Q:** Why was the NaN-equality comparison logic in `assert_df_equality` restructured?
  **A:** The inline loop that handled `allow_nan_equality` was replaced with a call to `assert_generic_df_equality(df1, df2, are_rows_equal_enhanced, [True])`, which removes duplicated row-comparison code and centralizes the common equality-and-diff logic in one helper.

- **[dependency] Q:** What import or dependency-related change was made in `chispa/number_helpers.py`, and why does it matter?
  **A:** The bare `except:` in `isnan(x)` was narrowed to `except TypeError:`. This keeps `math.isnan(x)` behavior for supported numeric values while only falling back to `False` for unsupported types, instead of swallowing all exceptions.

- **[scope] Q:** Which files were modified in this commit, and what kind of changes were made in each?
  **A:** `chispa/dataframe_comparer.py` was updated with new equality options, a new generic equality helper, and refactored comparison logic. `chispa/number_helpers.py` was changed to narrow the exception handling in `isnan` from a broad catch-all to `TypeError`.

- **[usage] Q:** How would a developer use the new DataFrame comparison options added by this commit?
  **A:** A developer can call `assert_df_equality(df1, df2, ignore_column_order=True)` to compare DataFrames while ignoring column order, or `assert_df_equality(df1, df2, ignore_row_order=True)` to ignore row order. The existing `allow_nan_equality` path now delegates to the generic row-comparison helper internally.

- **[edge_case] Q:** What default-value or empty-list edge case was changed for the `transforms` argument in `assert_df_equality`?
  **A:** `transforms` now defaults to `None` instead of an empty list. The function initializes it to a new empty list when `transforms is None`, avoiding shared mutable default state across calls.

- **[risk] Q:** What backward-compatibility or regression risk does this change introduce around row and column ordering?
  **A:** Enabling `ignore_row_order` now sorts each DataFrame by `df.columns`, which may change behavior for DataFrames with complex column types or non-deterministic ordering assumptions. Also, the new `transforms=None` handling and appended order-ignoring transforms change how user-provided transforms are composed with the built-in ones.

### MrPowers/chispa @ commit #27 `500793efe1` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this README update regarding `ignore_nullable=True` in DataFrame equality checks?
  **A:** It documents that `ignore_nullable=True` now also ignores the nullable property on `ArrayType()` elements, not just the nullable property on the column schema.

- **[new_api] Q:** What new documented example function is shown, and what does it demonstrate?
  **A:** The README adds an example function named `ignore_nullable_property_array()`, which demonstrates that two DataFrames with the same array values but different `ArrayType(..., nullable)` settings can be compared with `assert_df_equality` when `ignore_nullable=True` is used.

- **[signature_change] Q:** Did this change alter the signature of any Python function or method in the repository, and what does the diff show instead?
  **A:** No function or method signature is changed in the diff; the update is documentation-only and adds an example around `assert_df_equality(df1, df2, ignore_nullable=True)`.

- **[behavior_change] Q:** How does the documented equality behavior differ after this change when comparing DataFrames with array columns?
  **A:** The documentation now states that `ignore_nullable=True` bypasses differences in the nullable flag both on the column schema and on the elements inside an `ArrayType()`, so those nullable mismatches do not cause equality failures.

- **[refactor] Q:** What part of the README was reworked to clarify an existing concept, and how was it simplified or expanded?
  **A:** The section about ignoring nullable properties was expanded with a new explanation and an explicit array-column example, clarifying that nested array element nullability is included in the same `ignore_nullable=True` behavior.

- **[dependency] Q:** Were there any import or dependency changes introduced by this commit, and what does the diff indicate about that?
  **A:** No imports or package dependencies are changed in the diff; only `README.md` is updated with additional explanatory text and example code.

- **[scope] Q:** Which file is modified by this commit, and what kind of change is made there?
  **A:** Only `README.md` is modified, and the change is documentation: it adds text and a Python example explaining how `ignore_nullable=True` treats `ArrayType()` element nullability.

- **[usage] Q:** How should a developer use the described behavior when comparing DataFrames with nullable array elements?
  **A:** They should call `assert_df_equality(df1, df2, ignore_nullable=True)` so that differences in both the field nullable flag and the `ArrayType` element nullable flag are ignored during comparison.

- **[edge_case] Q:** What edge case involving array columns does the README say `ignore_nullable=True` covers?
  **A:** It covers the case where two DataFrames have identical array values but different `ArrayType(DoubleType(), ...)` nullability settings for the array elements, such as `True` versus `False`.

- **[risk] Q:** What compatibility concern might a reader infer from this documentation change about `ignore_nullable=True`?
  **A:** The main compatibility consideration is that users who expected `ignore_nullable=True` to affect only top-level column nullability now need to know it also suppresses nested `ArrayType()` element nullability differences, which could make some previously failing comparisons pass.

### MrPowers/chispa @ commit #31 `88cd6f33be` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** It extends dataframe and row comparison support to handle approximate equality with a configurable `precision` while also allowing `allow_nan_equality` to be combined with approximate comparisons.

- **[new_api] Q:** What new helper function is introduced in `chispa/number_helpers.py`, and what does it do?
  **A:** The new function `nan_safe_approx_equality(x, y, precision)` returns `True` when `abs(x - y) <= precision` or when both values are NaN, giving approximate equality a NaN-safe check.

- **[signature_change] Q:** How did the signature of `assert_approx_df_equality` in `chispa/dataframe_comparer.py` change?
  **A:** `assert_approx_df_equality` now has the signature `assert_approx_df_equality(df1, df2, precision, ignore_nullable=False, transforms=None, allow_nan_equality=False, ignore_column_order=False, ignore_row_order=False, ignore_schema=False)`, whereas it previously only accepted `df1, df2, precision, ignore_nullable=False`.

- **[behavior_change] Q:** How does `assert_df_equality` behave differently after this change when `precision` is nonzero?
  **A:** `assert_df_equality` now routes to `assert_generic_rows_equality(df1, df2, are_rows_approx_equal, [precision, allow_nan_equality])` whenever `precision != 0`, instead of only using the non-approximate row comparison paths.

- **[refactor] Q:** Why was `assert_approx_df_equality` restructured in `chispa/dataframe_comparer.py`?
  **A:** It was simplified to delegate to `assert_df_equality` so the approximate-equality logic is centralized in one place rather than duplicating schema and row comparison handling.

- **[dependency] Q:** What import change was made in `chispa/row_comparer.py`, and why is it relevant?
  **A:** `chispa/row_comparer.py` now imports `nan_safe_approx_equality` from `chispa.number_helpers` in addition to `nan_safe_equality`, so approximate row comparison can apply NaN-aware equality checks.

- **[scope] Q:** Which files are modified in this commit, and what kind of change does each file receive?
  **A:** `README.md` updates the Poetry install example; `chispa/dataframe_comparer.py` adds precision-aware dataframe equality behavior and expands `assert_approx_df_equality`; `chispa/number_helpers.py` adds `nan_safe_approx_equality`; and `chispa/row_comparer.py` updates approximate row comparison to accept `allow_nan_equality` and use the new helper.

- **[usage] Q:** How would a developer now call the approximate dataframe assertion if they also want NaN equality and schema/column-order options?
  **A:** They can call `assert_approx_df_equality(df1, df2, precision, ignore_nullable=False, transforms=None, allow_nan_equality=False, ignore_column_order=False, ignore_row_order=False, ignore_schema=False)`, for example passing `allow_nan_equality=True` and setting the desired ordering or schema flags in the same call.

- **[edge_case] Q:** What edge case does `are_rows_approx_equal` handle differently for float values when `allow_nan_equality` is enabled?
  **A:** When `allow_nan_equality` is `True`, float pairs are considered equal if `nan_safe_approx_equality(d1[key], d2[key], precision)` returns `True`, which treats two NaN values as equal in addition to values within the precision threshold.

- **[risk] Q:** What compatibility risk is introduced by combining `precision` with `allow_nan_equality` in `assert_df_equality`?
  **A:** The new `precision != 0` branch changes how row equality is evaluated whenever a nonzero precision is passed, so callers that previously relied on the older approximate comparison path or expected exact handling when `allow_nan_equality` was set could see different comparison results.

### MrPowers/chispa @ commit #35 `4d7617d6b3` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the schema comparison code?
  **A:** It consolidates nullability-ignoring schema comparison logic inside `chispa/schema_comparer.py` and removes the separate `chispa/structfield_comparer.py` module, while improving comparison of nested arrays and structs.

- **[new_api] Q:** What new helper functions are introduced in `chispa/schema_comparer.py`, and what do they do?
  **A:** `are_structfields_equal_ignore_nullable(sf1, sf2)` compares two `StructField` values while allowing nullability differences, and `are_datatypes_equal_ignore_nullable(dt1, dt2)` compares two Spark data types recursively, descending into arrays and structs while ignoring nullability.

- **[signature_change] Q:** How did the schema-comparison helper API change in terms of function names and parameters?
  **A:** The old `check_type_equal_ignore_nullable(sf1, sf2)` helper was removed and replaced by `are_datatypes_equal_ignore_nullable(dt1, dt2)`, which accepts data types directly instead of `StructField` objects. In addition, the call sites now use `are_structfields_equal_ignore_nullable(sf1, sf2)` instead of `are_structfields_equal(sf1, sf2, True)`.

- **[behavior_change] Q:** How does runtime comparison behavior differ after this change when comparing nested arrays or structs?
  **A:** Array element types are now compared recursively with `are_datatypes_equal_ignore_nullable(dt1.elementType, dt2.elementType)` instead of using a direct equality check, and struct data types are compared through `are_schemas_equal_ignore_nullable(dt1, dt2)`, so nested structures are checked more deeply while still ignoring nullability.

- **[refactor] Q:** What existing comparison logic was restructured, and why does the new structure make the code simpler?
  **A:** The former `check_type_equal_ignore_nullable` logic was split into `are_structfields_equal_ignore_nullable` and `are_datatypes_equal_ignore_nullable`, which separates StructField name handling from recursive datatype comparison and removes the need to thread an `ignore_nullability` flag through the helper.

- **[dependency] Q:** Which import was removed, and what does that indicate about the dependency between modules?
  **A:** `from chispa.structfield_comparer import are_structfields_equal` was removed from `chispa/schema_comparer.py`, indicating that schema comparison no longer depends on the deleted `chispa/structfield_comparer.py` module and now uses helpers defined locally in `schema_comparer.py`.

- **[scope] Q:** Which files are affected by this commit, and what changed in each file?
  **A:** `chispa/schema_comparer.py` was updated to use new local helpers, recursively compare nested types, and replace the old `check_type_equal_ignore_nullable` logic. `chispa/structfield_comparer.py` was deleted entirely.

- **[usage] Q:** How would a developer use the updated code to compare two Spark schemas while ignoring nullability?
  **A:** A developer would call `are_schemas_equal_ignore_nullable(s1, s2)` to check whether two schemas are equivalent while ignoring nullable differences, or `assert_schema_equality_ignore_nullable(s1, s2)` to raise a `SchemasNotEqualError` and build a `PrettyTable` diff when they are not equal.

- **[edge_case] Q:** How does the new StructField comparison handle `None` values for one or both inputs?
  **A:** `are_structfields_equal_ignore_nullable(sf1, sf2)` returns `True` only when both `sf1` and `sf2` are `None`; if exactly one is `None`, it returns `False`.

- **[risk] Q:** What backward-compatibility risk is introduced by deleting `chispa/structfield_comparer.py`?
  **A:** Any external code importing `are_structfields_equal` from `chispa.structfield_comparer` will break, because that module was deleted and the function name changed to `are_structfields_equal_ignore_nullable` inside `chispa/schema_comparer.py`.

### MrPowers/chispa @ commit #39 `fa17888e48` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in chispa's schema and DataFrame comparison code?
  **A:** It adds support for ignoring Spark schema metadata during equality checks, by threading an `ignore_metadata` flag through `assert_df_equality`, `assert_schema_equality`, and `are_structfields_equal`.

- **[new_api] Q:** What new comparison helper was introduced in `chispa/schema_comparer.py`, and what does it do?
  **A:** The new helper is `assert_schema_equality_full(s1, s2, ignore_nullable=False, ignore_metadata=False)`. It performs a full schema comparison that can ignore nullable differences and/or metadata differences, and raises `SchemasNotEqualError` with a `PrettyTable` diff when the schemas do not match.

- **[signature_change] Q:** How did the signature of `assert_df_equality` change in `chispa/dataframe_comparer.py`?
  **A:** `assert_df_equality` gained a new trailing parameter named `ignore_metadata=False`, so it now calls `assert_schema_equality(df1.schema, df2.schema, ignore_nullable, ignore_metadata)`.

- **[behavior_change] Q:** How does schema comparison behavior differ at runtime after this change when metadata differs between two schemas?
  **A:** Previously `assert_schema_equality` only handled exact equality or nullable-ignoring comparisons; after this change, callers can pass `ignore_metadata=True` so schemas with different `StructField.metadata` values can still compare as equal, while mismatches are still reported when `ignore_metadata` is false.

- **[refactor] Q:** Why was `assert_schema_equality` restructured instead of keeping the old nullable-only branch?
  **A:** It was refactored to route the exact-match case directly to `assert_basic_schema_equality(s1, s2)` when both `ignore_nullable` and `ignore_metadata` are false, and to use the new fuller comparison path otherwise, so one API can handle both nullable and metadata tolerance.

- **[dependency] Q:** What import or dependency usage changed in the schema comparison logic, and why?
  **A:** No new external dependency was added, but the code now uses the existing `PrettyTable`, `six.moves.zip_longest`, and `blue` formatting in `assert_schema_equality_full` to produce a readable diff while supporting metadata-aware comparisons.

- **[scope] Q:** Which files were modified, and what kind of changes were made in each?
  **A:** `chispa/dataframe_comparer.py` was changed to add the `ignore_metadata` argument and pass it into schema comparison. `chispa/schema_comparer.py` was changed to add metadata-aware schema comparison logic, introduce `assert_schema_equality_full`, update `are_structfields_equal`, and make some existing helper comments and boolean checks clearer.

- **[usage] Q:** How would a developer use the new metadata-ignoring behavior when comparing two DataFrames?
  **A:** They would call `assert_df_equality(df1, df2, ignore_metadata=True)`; that flag is forwarded to schema comparison so differing field metadata does not cause the assertion to fail.

- **[edge_case] Q:** What edge case does the new `ignore_metadata` handling cover in `are_structfields_equal`?
  **A:** When `ignore_metadata` is enabled, two `StructField` values can still be considered equal even if their `metadata` dictionaries differ, as long as their names and data types match under the nullable comparison rules.

- **[risk] Q:** What backward-compatibility or regression risk is introduced by this change?
  **A:** Adding the new `ignore_metadata` parameter changes the public signatures of `assert_df_equality` and `assert_schema_equality`, which can affect callers using positional arguments; additionally, enabling metadata-ignoring comparisons may cause some previously failing schema mismatches to pass.

### MrPowers/chispa @ commit #43 `591e58f888` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the change in `chispa/terminal_str_formatter.py`?
  **A:** The change expands the set of supported terminal style names in `format_string(input, formats)` by adding more ANSI format keys, including `invert`, `hide`, and several color variants such as `black`, `green`, `yellow`, `purple`, `cyan`, and light-colored aliases.

- **[new_api] Q:** What new formatting names does `format_string` now recognize, and what do they do?
  **A:** `format_string` now recognizes additional ANSI style names mapped in its `formats_dict`, including `invert` for reverse video (`\033[7m`), `hide` for hidden text (`\033[8m`), and color names such as `black`, `green`, `yellow`, `purple`, `cyan`, `light_gray`, `dark_gray`, `light_red`, `light_green`, `light_yellow`, `light_blue`, `light_purple`, and `light_cyan`, each mapped to an ANSI escape code.

- **[signature_change] Q:** Did the function signature of `format_string` change in this commit, and if not, what changed instead?
  **A:** The signature of `format_string(input, formats)` did not change. The update only changed the internal `formats_dict` mapping by replacing some keys and adding many more ANSI style entries.

- **[behavior_change] Q:** How does the runtime behavior of `format_string` differ after this change when a caller passes style names?
  **A:** Before the change, only `inverted` and `hidden` were available for reverse and hidden text, and the color palette was smaller. After the change, callers can use `invert` and `hide` instead, along with additional color names like `green`, `yellow`, `cyan`, and light variants, and those names will now be converted into ANSI escape codes during formatting.

- **[refactor] Q:** Why were the existing terminal style keys restructured in `terminal_str_formatter.py`?
  **A:** The style dictionary was restructured to use shorter or more conventional names for some effects, such as `invert` and `hide`, and to centralize a broader set of color aliases in the same `formats_dict` so `format_string` can support more terminal styles from one lookup table.

- **[dependency] Q:** Did this commit add or remove any external imports or dependencies, and what changed in the formatting lookup instead?
  **A:** No imports or external dependencies were changed in this diff. The only change was within the local `formats_dict` in `chispa/terminal_str_formatter.py`, where additional ANSI escape-code mappings were added and renamed.

- **[scope] Q:** Which file is modified by this commit, and what kind of change is made there?
  **A:** Only `chispa/terminal_str_formatter.py` is modified, and the change is limited to the `formats_dict` inside `format_string`, where several ANSI style keys are renamed and many new color/style entries are added.

- **[usage] Q:** How would a developer use the newly supported formatting names with `format_string`?
  **A:** A developer would call `format_string(input, formats)` with the `formats` list containing the new keys, for example `['bold', 'invert']` or `['red', 'hide']`, and `format_string` will apply the corresponding ANSI escape codes from its internal dictionary.

- **[edge_case] Q:** What happens if a caller uses the old keys `inverted` or `hidden` after this change?
  **A:** Those keys are no longer present in `formats_dict`, so `format_string` will no longer have explicit mappings for `inverted` or `hidden`; the supported equivalents in this commit are `invert` and `hide`.

- **[risk] Q:** What backward-compatibility risk does this change introduce for existing callers of `format_string`?
  **A:** The main compatibility risk is that code still passing `inverted` or `hidden` may stop working as expected because those names were removed from the lookup table and replaced with `invert` and `hide`, even though the underlying ANSI behavior remains the same.

### MrPowers/chispa @ commit #47 `39c3bb18c7` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** The commit modernizes and cleans up the chispa codebase by switching to external `prettytable`, tightening `None` checks, reformatting several Python modules, and adding GitHub issue and pull request templates.

- **[new_api] Q:** What new repository-level templates were added, and what are they for?
  **A:** Three new GitHub templates were added under `.github/`: `ISSUE_TEMPLATE/bug_report.md` for bug reports, `ISSUE_TEMPLATE/feature_request.md` for feature requests, and `pull_request_template.md` for standardizing pull request descriptions and checklists.

- **[signature_change] Q:** How did the `assert_df_equality` method on `chispa.Chispa` change in terms of its function signature formatting and parameter order?
  **A:** The `chispa.Chispa.assert_df_equality` method kept the same parameters and order, but its signature was reformatted across multiple lines for readability; it still accepts `df1`, `df2`, `ignore_nullable`, `transforms`, `allow_nan_equality`, `ignore_column_order`, `ignore_row_order`, `underline_cells`, and `ignore_metadata`.

- **[behavior_change] Q:** What runtime behavior changes in `assert_approx_column_equality` when comparing `None` values after this commit?
  **A:** `chispa.column_comparer.assert_approx_column_equality` now uses identity checks (`is None` / `is not None`) instead of equality checks (`== None` / `!= None`) when deciding whether one or both values are `None`.

- **[refactor] Q:** Why was the `Chispa` class definition and its `assert_df_equality` wrapper restructured?
  **A:** The `chispa.__init__.py` file reformats `class Chispa()` to `class Chispa` and expands the wrapper call over multiple lines, but the structure still delegates directly to `assert_df_equality` while passing `self.formats` explicitly as the last argument.

- **[dependency] Q:** What import change was made for `PrettyTable`, and what does it imply about the dependency source?
  **A:** `chispa.column_comparer` switched from importing `PrettyTable` via `from chispa.prettytable import PrettyTable` to `from prettytable import PrettyTable`, which means the code now depends on the external `prettytable` package instead of the vendored module.

- **[scope] Q:** Which files were changed in this commit, and what kind of changes did each receive?
  **A:** The commit adds `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`, and `.github/pull_request_template.md`; edits `README.md` to remove the vendored-dependencies section; and updates `chispa/__init__.py`, `chispa/bcolors.py`, `chispa/column_comparer.py`, and `chispa/dataframe_comparer.py` mainly for formatting, import cleanup, and `None`-comparison fixes.

- **[usage] Q:** How would a developer call the updated `assert_df_equality` wrapper from a `Chispa` instance after this change?
  **A:** A developer would still call `chispa_instance.assert_df_equality(df1, df2, ...)`; the wrapper in `chispa.Chispa` forwards the call to the module-level `assert_df_equality` function and passes `self.formats` automatically as the final `formats` argument.

- **[edge_case] Q:** What edge-case handling was tightened in the column comparison logic?
  **A:** In `chispa.column_comparer.assert_approx_column_equality`, the checks for one-or-both values being `None` now explicitly use `is None` and `is not None`, and the branch for both values being `None` remains treated as equal by adding the formatted `None` values to the table.

- **[risk] Q:** What compatibility risk does the removal of vendored `PrettyTable` create for users of chispa?
  **A:** Because `chispa.column_comparer` now imports `PrettyTable` from the external `prettytable` package and the README removes the vendored-dependencies note, environments that relied on the bundled copy of PrettyTable may need the package installed separately to avoid import failures.

### MrPowers/chispa @ commit #52 `580631aa33` (in_repo_split=val)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** This commit introduces the new `FormattingConfig`-based formatting API and updates the library to accept either a `FormattingConfig` instance or an older dataclass-style formats object when rendering comparison error messages.

- **[new_api] Q:** What new formatting-related API is exposed from `chispa/__init__.py`, and what does it provide?
  **A:** `chispa/__init__.py` now exports `Color`, `Style`, `Format`, and `FormattingConfig` from `chispa.formatting`, giving users typed formatting options for custom error-message styling.

- **[signature_change] Q:** How did the `Chispa` constructor signature change in `chispa/__init__.py`?
  **A:** `Chispa.__init__` changed from `def __init__(self, formats=DefaultFormats(), default_output=None)` to `def __init__(self, formats: FormattingConfig | None = None, default_output=None)`, and it now normalizes the `formats` argument before storing it.

- **[behavior_change] Q:** How does runtime handling of the `formats` argument differ after this change in `Chispa` and `assert_df_equality`?
  **A:** If `formats` is omitted or falsy, the code now creates a fresh `FormattingConfig()` instead of using a shared `DefaultFormats()` default object; if `formats` is not already a `FormattingConfig`, it is converted with `FormattingConfig._from_arbitrary_dataclass(...)`.

- **[refactor] Q:** Why was the README example for custom formatting restructured?
  **A:** The README examples were updated to show `FormattingConfig(...)` construction instead of a custom `@dataclass` like `MyFormats`, reflecting the new formatting API and showing both string-based and enum-based color/style values.

- **[dependency] Q:** What import-level dependency change was made around formatting, and why?
  **A:** Files that previously imported `DefaultFormats` now import `FormattingConfig` from `chispa.formatting`, and `chispa/__init__.py` additionally re-exports `Color`, `Format`, and `Style` so callers can build formatting configs without reaching into internal modules.

- **[scope] Q:** Which files were touched in this commit, and what kind of change did each receive?
  **A:** `README.md` was updated with new formatting examples; `chispa/__init__.py`, `chispa/dataframe_comparer.py`, and related modules were updated to use `FormattingConfig`; `chispa/bcolors.py` and `chispa/column_comparer.py` only received `from __future__ import annotations`; `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md` were newline-only fixes; and `NOTICE-binary.md` was added as a new notice file.

- **[usage] Q:** How should a developer now pass custom row/cell colors when calling `assert_basic_rows_equality`?
  **A:** They should construct a `FormattingConfig`, for example `formats = FormattingConfig(mismatched_rows={"color": "light_yellow"}, matched_rows={"color": "cyan", "style": "bold"}, mismatched_cells={"color": "purple"}, matched_cells={"color": "blue"})`, and pass it as `formats=formats` to `assert_basic_rows_equality(...)`.

- **[edge_case] Q:** What happens when the `formats` argument is `None`, missing, or otherwise falsy in the updated comparison code?
  **A:** Both `Chispa.__init__` and `assert_df_equality` now replace a missing or falsy `formats` value with a new `FormattingConfig()` instance before proceeding.

- **[risk] Q:** What backward-compatibility or regression risk does this formatting refactor introduce?
  **A:** Code that relied on the old `DefaultFormats()` default object or passed an arbitrary dataclass with matching fields now depends on conversion through `FormattingConfig._from_arbitrary_dataclass`, so any incompatibility in that conversion path or in the new enum/string-based formatting objects could affect existing callers.

### MrPowers/chispa @ commit #56 `50c2411e88` (in_repo_split=test)

- **[intent] Q:** What is the main purpose of the changes in `chispa/schema_comparer.py` and `chispa/bcolors.py`?
  **A:** The change adds richer schema-diff reporting for schema assertion failures, including a new tree-style diff for wide or nested schemas and reusable color helpers for formatting diff lines.

- **[new_api] Q:** What new functions were introduced in `chispa/schema_comparer.py`, and what does each one do?
  **A:** The commit adds `print_schema_diff(...)`, which prints either a table or tree diff based on `output_format`; `create_schema_comparison_tree(...)`, which builds a colored side-by-side tree representation of two `StructType` schemas; `create_schema_comparison_table(...)`, which builds a colored `PrettyTable` comparison; `check_if_schemas_are_wide(...)`, which detects nested structs or more than 10 columns; and `handle_schemas_not_equal(...)`, which chooses the appropriate diff format and raises `SchemasNotEqualError`.

- **[signature_change] Q:** Which function signature was expanded to support the new schema-diff flow, and what parameters does it take now?
  **A:** `print_schema_diff` was added with the signature `print_schema_diff(s1: StructType, s2: StructType, ignore_nullable: bool, ignore_metadata: bool, output_format: str = "table") -> None`, allowing callers to select `table` or `tree` output while passing the same equality flags used elsewhere.

- **[behavior_change] Q:** How does the error output differ at runtime after this change when two schemas are not equal?
  **A:** Instead of always raising `SchemasNotEqualError` with a plain two-column `PrettyTable`, the code now chooses between a table diff for smaller schemas and a tree diff for wide or nested schemas, and the failing rows/lines are colorized with `line_blue` for matching entries and `line_red` for mismatches.

- **[refactor] Q:** Why were the repeated table-building blocks in the schema assertion functions refactored?
  **A:** The duplicated code that manually created a `PrettyTable`, zipped schema fields, and raised `SchemasNotEqualError` was extracted into shared helpers (`create_schema_comparison_table`, `create_schema_comparison_tree`, and `handle_schemas_not_equal`) so `assert_schema_equality_full`, `assert_basic_schema_equality`, and `assert_schema_equality_ignore_nullable` can reuse the same diff formatting logic.

- **[dependency] Q:** What import changes were made in `chispa/schema_comparer.py`, and why were they needed?
  **A:** The file now imports `bcolors`, `line_blue`, and `line_red` from `chispa.bcolors` so it can wrap schema-diff text in color codes for the new table and tree outputs.

- **[scope] Q:** Which files are modified in this commit, and what kind of changes does each file contain?
  **A:** `chispa/bcolors.py` adds the new color helper functions `line_blue` and `line_red`. `chispa/schema_comparer.py` adds the new diff-printing helpers, wide-schema detection, and refactors the schema equality assertion functions to use the shared error-handling path.

- **[usage] Q:** How would a developer use the new schema diff functionality or select its output format?
  **A:** A developer can call `print_schema_diff(s1, s2, ignore_nullable, ignore_metadata, output_format="table")` to print a table diff or pass `output_format="tree"` to print the tree representation; the schema assertion functions now automatically route failures through the same diff-building logic.

- **[edge_case] Q:** What edge cases cause the tree-style schema diff to be used instead of the table-style diff?
  **A:** `check_if_schemas_are_wide` returns true when either schema contains a nested `struct` field or when either schema has more than 10 top-level columns, and `handle_schemas_not_equal` then uses `create_schema_comparison_tree` for the error message.

- **[risk] Q:** What backward-compatibility or regression risk is introduced by the new schema-diff handling?
  **A:** The failure message format for schema mismatches can now change from the previous plain table output to a tree layout for wide or nested schemas, and `print_schema_diff` also raises a `ValueError` if `output_format` is anything other than `"table"` or `"tree"`, which could affect callers expecting the old single-format behavior.

### MrPowers/chispa @ commit #60 `3d7915663e` (in_repo_split=test)

- **[intent] Q:** What is the main purpose of this commit in the chispa package?
  **A:** The commit modernizes chispa by replacing string literals with shared enums for schema/output type names, updates imports to use `collections.abc.Callable`, and cleans up the package exports in `chispa/__init__.py` and `chispa/formatting/__init__.py`.

- **[new_api] Q:** What new definitions were added in `chispa/common_enums.py`, and what do they represent?
  **A:** The new module `chispa/common_enums.py` adds two enum classes: `OutputFormat` with values `TABLE = "table"` and `TREE = "tree"`, and `TypeName` with values `ARRAY = "array"` and `STRUCT = "struct"`. They centralize the output-format and Spark type-name constants used elsewhere in the package.

- **[signature_change] Q:** How did the `print_schema_diff` function signature change in `chispa/schema_comparer.py`?
  **A:** `print_schema_diff` changed its `output_format` parameter from `str = "table"` to `OutputFormat = OutputFormat.TABLE`, so the function now expects the `chispa.common_enums.OutputFormat` enum instead of a raw string.

- **[behavior_change] Q:** How does `print_schema_diff` behave differently at runtime after this change?
  **A:** It now branches on `OutputFormat.TABLE` and `OutputFormat.TREE` instead of comparing against the strings `"table"` and `"tree"`. If an invalid value is passed, the error message now reports the valid names via `OutputFormat.__members__` instead of a hardcoded list.

- **[refactor] Q:** Which repeated string comparisons were simplified in `chispa/schema_comparer.py`, and why is that a refactor?
  **A:** struct"` and `"array"` with `TypeName.STRUCT` and `TypeName.ARRAY` in functions like `create_schema_comparison_tree`, `check_if_schemas_are_wide`, and `are_datatypes_equal_ignore_nullable`. This restructures the code to use shared enum constants instead of duplicated string literals.

- **[dependency] Q:** What import changes were made around `Callable`, and what was the reason visible in the diff?
  **A:** `Callable` was changed from `typing.Callable` to `collections.abc.Callable` in `chispa/__init__.py`, `chispa/dataframe_comparer.py`, and `chispa/rows_comparer.py`. The diff shows this as a type-import cleanup/update, with no other logic changes tied to the import swap.

- **[scope] Q:** Which modules were touched by this commit, and what kind of change did each one receive?
  **A:** `chispa/__init__.py` was cleaned up to remove the Spark-path bootstrapping logic, switch to `collections.abc.Callable`, and reorder `__all__`; `chispa/common_enums.py` was added with new enums; `chispa/dataframe_comparer.py` and `chispa/rows_comparer.py` only updated their `Callable` imports; `chispa/schema_comparer.py` adopted the new enums for output format and type-name comparisons; and `chispa/formatting/__init__.py` reordered its `__all__` exports.

- **[usage] Q:** How would a developer use the new `OutputFormat` enum when calling the schema diff printer?
  **A:** A developer would pass `OutputFormat.TABLE` or `OutputFormat.TREE` to `print_schema_diff` in `chispa/schema_comparer.py` instead of passing the strings `"table"` or `"tree"`.

- **[edge_case] Q:** What default or fallback behavior exists for the `output_format` argument in `print_schema_diff` after this change?
  **A:** The default remains the table view: `output_format` defaults to `OutputFormat.TABLE`, so calling `print_schema_diff(...)` without specifying the argument prints the PrettyTable-based schema comparison.

- **[risk] Q:** What backward-compatibility risk does this commit introduce for callers of `print_schema_diff` or related schema helpers?
  **A:** Callers that passed plain strings such as `"table"`, `"tree"`, `"struct"`, or `"array"` may no longer match the new enum-based comparisons, because the code now compares against `OutputFormat` and `TypeName` members instead of raw strings.

