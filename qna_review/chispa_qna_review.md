# Generated QA dataset review

## split: cr_test  (57 commits)

### MrPowers/chispa @ commit #0 `a75eeb69b0` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** It initializes the project with a minimal package layout and adds a basic `assert_column_equality` helper in `chispa/column_comparer.py`, alongside a local Spark session setup in `chispa/spark.py`.

- **[new_api] Q:** What new function is introduced in `chispa/column_comparer.py`, and what does it do?
  **A:** The new function is `assert_column_equality(df, col_name1, col_name2)`. It collects the two named columns from a DataFrame, compares their values row by row, and prints matching pairs or `MISMATCH: ...` lines when the column values differ.

- **[signature_change] Q:** What is the signature of the new column comparison helper, including its parameters?
  **A:** The function signature is `assert_column_equality(df, col_name1, col_name2)`, taking a DataFrame-like `df` and two column-name arguments, `col_name1` and `col_name2`.

- **[behavior_change] Q:** How does the runtime behavior of the column comparison logic work after this change?
  **A:** It now performs a `select(col_name1, col_name2).collect()`, builds two Python lists from the collected rows, and if the lists are not equal it zips the values and prints either the equal pair as `value | value` or a mismatch line prefixed with `MISMATCH:`.

- **[refactor] Q:** Why was the column comparison logic structured the way it is in this commit?
  **A:** The implementation is simplified into a direct Python helper that compares two collected lists and prints differences, while the older Scala-style throwing logic is left only as commented-out reference code.

- **[dependency] Q:** What new dependency-related import appears in the repository, and what is it used for?
  **A:** `chispa/spark.py` imports `SparkSession` from `pyspark.sql` and uses it to create a session via `SparkSession.builder.master("local").appName("chispa").getOrCreate()`.

- **[scope] Q:** Which files are added or changed in this commit, and what does each contain?
  **A:** Three files are added: `README.md` with the project title `# chispa`, `chispa/__init__.py` as an empty package marker, `chispa/column_comparer.py` with `assert_column_equality`, and `chispa/spark.py` with a local `SparkSession` initializer.

- **[usage] Q:** How would a developer use the new Spark session setup and column comparison helper from this commit?
  **A:** A developer would import `spark` from `chispa.spark` to get a ready-made local `SparkSession`, then call `assert_column_equality(df, "col1", "col2")` from `chispa.column_comparer` to compare two DataFrame columns.

- **[edge_case] Q:** How does `assert_column_equality` handle identical versus differing values when comparing columns?
  **A:** When paired values are equal, it prints the value twice separated by ` | `. When values differ, it prints `MISMATCH: <left> | <right>`. The function does not raise an exception in the current code.

- **[risk] Q:** What backward-compatibility or regression risk is introduced by this implementation of column comparison?
  **A:** The helper no longer throws a `ColumnMismatch` or any exception on unequal columns; it only prints messages. That can break callers that expect a failing assertion-like behavior from a column comparison utility.

### MrPowers/chispa @ commit #1 `9292f52e5b` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** The commit adds packaging/setup metadata in a new `setup.py`, documents basic installation and test commands in `README.md`, and changes `chispa/column_comparer.py` so column mismatches raise a custom exception with a formatted message instead of printing to stdout.

- **[new_api] Q:** What new class is introduced in `chispa/column_comparer.py`, and what does it represent?
  **A:** The new class is `ColumnsNotEqualError(Exception)`, a custom exception used to signal that two DataFrame columns are not equal.

- **[signature_change] Q:** What function signature or type annotation changed in this commit?
  **A:** No function or method signature was changed in the diff; `assert_column_equality(df, col_name1, col_name2)` keeps the same parameters, and the main API addition is the new `ColumnsNotEqualError` exception class.

- **[behavior_change] Q:** How does `assert_column_equality` behave differently at runtime after this change?
  **A:** Before the change, mismatched values were only printed line by line. After the change, `assert_column_equality` builds a message in `lines` and raises `ColumnsNotEqualError` with the joined message when the two collected column value lists are not equal.

- **[refactor] Q:** Why was the mismatch-reporting logic in `assert_column_equality` restructured?
  **A:** It was restructured so the function now accumulates comparison output in a `lines` list and raises one exception containing the full diff-style message, rather than emitting separate `print` statements for each row.

- **[dependency] Q:** What dependency or packaging-related changes were introduced, and why are they relevant?
  **A:** A new `setup.py` was added using `setuptools.setup` and `find_packages`, declaring `pytest` in both `install_requires` and `tests_require`, plus extras like `test`, `all`, `docs`, and `lint`. This makes the package installable and documents its test dependency.

- **[scope] Q:** Which files were touched in this commit, and what changed in each one?
  **A:** Three files changed: `README.md` gained a short setup section with conda, pip, and pytest commands; `chispa/column_comparer.py` added `ColumnsNotEqualError` and changed mismatch handling from printing to raising that exception; `setup.py` was added to define package metadata and dependencies.

- **[usage] Q:** How would a developer use the updated column comparison code after this change?
  **A:** A developer would call `assert_column_equality(df, col_name1, col_name2)` as before, but now they should expect a `ColumnsNotEqualError` to be raised on mismatches and catch that exception if they want to inspect the formatted comparison message.

- **[edge_case] Q:** What happens when the two compared columns are equal, and what message structure is used when they are not?
  **A:** If `colName1Elements` and `colName2Elements` are equal, the function returns without raising an exception. If they differ, it creates a message starting with an empty string element in `lines`, then appends one line per pair as either `value | value` or `MISMATCH: value | value`, and raises `ColumnsNotEqualError` with the lines joined by newlines.

- **[risk] Q:** What backward-compatibility or regression risk does this change introduce?
  **A:** Code that previously relied on `assert_column_equality` only printing mismatches will now fail with a raised `ColumnsNotEqualError`, so existing tests or callers may need to be updated to catch the exception instead of expecting console output.

### MrPowers/chispa @ commit #2 `261d0a24ee` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** The commit updates `chispa/column_comparer.py` to add ANSI color formatting to matching column values in `assert_column_equality`, and it bumps the package version in `setup.py` from `0.0.1` to `0.0.2`.

- **[new_api] Q:** What new helper was introduced in `chispa/column_comparer.py`, and what does it provide?
  **A:** A new `bcolors` class was added. It defines ANSI escape-code constants such as `HEADER`, `OKBLUE`, `OKGREEN`, `WARNING`, `FAIL`, `ENDC`, `BOLD`, and `UNDERLINE` for colored terminal output.

- **[signature_change] Q:** Did the signature of `assert_column_equality` change in this commit, and if not, what changed inside it?
  **A:** The function signature of `assert_column_equality(df, col_name1, col_name2)` did not change. Only its internal formatting logic changed so that equal values are wrapped in `bcolors.OKBLUE` and `bcolors.ENDC`.

- **[behavior_change] Q:** How does the runtime behavior of `assert_column_equality` differ after this change when two compared column values match?
  **A:** For matching pairs, the error message now prints the values in blue using ANSI color codes instead of plain text. The mismatch case no longer prefixes the line with `MISMATCH:` and instead prints just `value1 | value2`.

- **[refactor] Q:** Why was the string construction in `assert_column_equality` restructured?
  **A:** The construction was simplified to highlight equal values with color formatting by using the new `bcolors` constants, and the previous explicit `MISMATCH:` label was removed from the non-equal branch.

- **[dependency] Q:** What import or dependency-related change was made to support the new output formatting?
  **A:** No new external dependency was added; instead, `chispa/column_comparer.py` now defines its own local `bcolors` class with ANSI escape sequences, avoiding any extra import for colored output.

- **[scope] Q:** Which files are touched by this commit, and what does each change do?
  **A:** `chispa/column_comparer.py` is modified to add the `bcolors` class and change `assert_column_equality` message formatting. `setup.py` is updated to bump the package version from `0.0.1` to `0.0.2`. A new file, `chispa/dataframe_comparer.py`, is added but is empty in the diff.

- **[usage] Q:** How would a developer use the changed code to get the new colored comparison output?
  **A:** They would continue calling `assert_column_equality(df, col_name1, col_name2)` from `chispa/column_comparer.py`. When the function raises `ColumnsNotEqualError`, matching values in the message are now rendered in blue via the new `bcolors` constants.

- **[edge_case] Q:** What does the new formatting logic do for equal versus unequal compared values in the raised error message?
  **A:** Equal values are wrapped with `bcolors.OKBLUE` and `bcolors.ENDC`, while unequal values are printed without the old `MISMATCH:` prefix, as plain `value1 | value2` text.

- **[risk] Q:** What compatibility risk could this change introduce for consumers of the comparison error message?
  **A:** Code or tests that assert the exact `ColumnsNotEqualError` message may break because the message now includes ANSI color escape codes for matching values and no longer includes the `MISMATCH:` prefix on mismatches.

### MrPowers/chispa @ commit #3 `02558e588b` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** The commit refactors the comparison helpers to use prettier table-based error messages, adds a shared ANSI color module, introduces DataFrame comparison helpers in a new module, and removes the old top-level setup.py packaging file.

- **[new_api] Q:** What new public helper is introduced in chispa/dataframe_comparer.py, and what does it do?
  **A:** The new `assert_df_equality(df1, df2)` function compares two Spark DataFrames by schema and collected rows, raising `SchemasNotEqualError` when schemas differ and `DataFramesNotEqualError` with a `PrettyTable` diff when the rows differ.

- **[signature_change] Q:** How did the error-reporting code in `assert_column_equality` change its signature or output construction in `chispa/column_comparer.py`?
  **A:** The function signature of `assert_column_equality(df, col_name1, col_name2)` did not change, but its failure output was changed to build a `PrettyTable` using the two column names as headers and raise `ColumnsNotEqualError` with `"\n" + t.get_string()` instead of joining plain text lines.

- **[behavior_change] Q:** How does the runtime behavior of `assert_column_equality` differ after this change when two compared columns contain equal values?
  **A:** When paired values are equal, the error table rows now include ANSI color codes from `bcolors.LightBlue` and `bcolors.LightRed` around both values before the `ColumnsNotEqualError` is raised, instead of emitting the earlier plain-colored text line format.

- **[refactor] Q:** Why was the color handling in `chispa/column_comparer.py` restructured?
  **A:** The inline `bcolors` class was removed from `column_comparer.py` and replaced by an import from `chispa.bcolors`, so the ANSI color constants can be shared from a dedicated module rather than duplicated inside the comparer.

- **[dependency] Q:** What new third-party dependency is imported in the updated comparer modules, and why?
  **A:** Both `chispa/column_comparer.py` and `chispa/dataframe_comparer.py` now import `PrettyTable` from `prettytable` so they can format mismatch details as tabular error messages instead of plain string lists.

- **[scope] Q:** Which files were added, modified, or removed in this commit, and what changed in each?
  **A:** `chispa/bcolors.py` was added with ANSI color constants; `chispa/dataframe_comparer.py` was added with `DataFramesNotEqualError`, `SchemasNotEqualError`, `assert_df_equality`, and `are_dfs_equal`; `chispa/column_comparer.py` was modified to import `bcolors` and `PrettyTable` and to emit a table in its error path; `chispa/row_comparer.py` was added but contains only an incomplete `def are_rows_equal:` stub; and `setup.py` was deleted.

- **[usage] Q:** How would a developer use the new DataFrame equality helper introduced by this commit?
  **A:** A developer would call `assert_df_equality(df1, df2)` from `chispa.dataframe_comparer`; if the DataFrames have different schemas it raises `SchemasNotEqualError`, and if their collected rows differ it raises `DataFramesNotEqualError` with a pretty-printed table of `df1` and `df2` values.

- **[edge_case] Q:** How does `assert_df_equality` handle schema mismatches versus row mismatches?
  **A:** It checks schema equality first and immediately raises `SchemasNotEqualError("These schemas aren't equal")` if the schemas differ; only when the schemas match does it collect rows and compare them for data mismatch.

- **[risk] Q:** What backward-compatibility or regression risk is introduced by this commit?
  **A:** The change alters error-message formatting in `assert_column_equality` and adds a new schema check in `assert_df_equality`, so any tests or callers that depend on the previous plain-text mismatch output or on row-only comparison behavior could break or observe different exceptions and messages.

### MrPowers/chispa @ commit #4 `d766794336` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repo?
  **A:** It improves equality assertions by adding approximate numeric column comparison and by making DataFrame schema mismatch errors more informative with tabular diff output.

- **[new_api] Q:** What new function is introduced in chispa/column_comparer.py, and what does it do?
  **A:** The new function is assert_approx_column_equality(df, col_name1, col_name2, precision). It compares two columns row by row and treats values as equal when their absolute difference is less than precision; otherwise it raises ColumnsNotEqualError with a PrettyTable diff.

- **[signature_change] Q:** Which new parameter is added by the new column-comparison API, and what type of comparison does it control?
  **A:** The new assert_approx_column_equality function adds a precision parameter. It controls approximate comparison by checking whether abs(elements[0] - elements[1]) < precision for each row.

- **[behavior_change] Q:** How does schema comparison behave differently after this change when two DataFrames have different schemas?
  **A:** assert_df_equality now builds a PrettyTable with columns 'schema1' and 'schema2', adds paired schema elements row by row, and raises SchemasNotEqualError containing that table instead of the previous generic message 'These schemas aren't equal'.

- **[refactor] Q:** What small refactor was made inside assert_column_equality, and what is its effect on the code structure?
  **A:** The code that built the highlighted row for equal values was split into two local variables, first and second, before calling t.add_row([first, second]). This simplifies the inline list construction without changing the existing equal-value highlighting behavior.

- **[dependency] Q:** Which existing dependency or import is being used by the changed code to produce the new error output, and for what purpose?
  **A:** PrettyTable is used in both changed functions to format mismatch details as a table. In assert_df_equality it formats schema differences, and in assert_approx_column_equality it formats column value differences.

- **[scope] Q:** Which files are touched in this commit, and what kind of change does each file receive?
  **A:** chispa/column_comparer.py gains the new assert_approx_column_equality function and a small row-formatting refactor in assert_column_equality. chispa/dataframe_comparer.py changes assert_df_equality to emit a PrettyTable-based schema mismatch message.

- **[usage] Q:** How would a developer use the new approximate column equality check added by this commit?
  **A:** A developer would call assert_approx_column_equality(df, col_name1, col_name2, precision) on a DataFrame, passing the two column names to compare and a numeric precision threshold for acceptable absolute differences.

- **[edge_case] Q:** What tolerance rule does the new approximate comparison use when two numeric values are very close?
  **A:** It considers two values equal only when their absolute difference is strictly less than precision. If the difference is exactly equal to precision, the row is treated as unequal and included in the error table.

- **[risk] Q:** What compatibility or regression risk is introduced by the new schema-mismatch error formatting?
  **A:** Code that previously relied on the exact SchemasNotEqualError message 'These schemas aren't equal' may break, because the exception now contains a PrettyTable string prefixed by a newline instead of the old fixed text.

### MrPowers/chispa @ commit #5 `06b3255cc1` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the change in `chispa/column_comparer.py` for `assert_approx_column_equality`?
  **A:** The change makes `assert_approx_column_equality(df, col_name1, col_name2, precision)` handle `None` values explicitly while comparing two columns approximately, so `None` in one column and a non-`None` value in the other is treated as unequal instead of causing a numeric subtraction check to fail.

- **[new_api] Q:** What new comparison behavior was introduced inside `assert_approx_column_equality`, and what does it do for row formatting?
  **A:** A new `None`-aware branch was added inside `assert_approx_column_equality` that checks whether exactly one side of a row is `None`; in that case it marks the columns as not equal and adds the raw string values to the `PrettyTable` instead of highlighting them as approximately equal.

- **[signature_change] Q:** Did the signature of `assert_approx_column_equality` change in this commit, and what type-related behavior changed in its body?
  **A:** The function signature did not change; it is still `assert_approx_column_equality(df, col_name1, col_name2, precision)`. The body changed to explicitly compare `None` values with `== None` / `!= None` before performing `abs(elements[0] - elements[1]) < precision`.

- **[behavior_change] Q:** How does the runtime behavior of approximate column comparison differ after this change when one value is `None`?
  **A:** Before the change, the code would reach `abs(elements[0] - elements[1]) < precision` for non-matching values and could fail on `None` arithmetic. After the change, if one element is `None` and the other is not, the row is immediately treated as unequal, `all_rows_equal` is set to `False`, and the row is added to the table as plain strings.

- **[refactor] Q:** What was simplified in the row-formatting logic inside `assert_approx_column_equality`?
  **A:** The code now computes the colored `first` and `second` strings once at the top of each loop iteration, then uses separate conditional branches for `None` handling and approximate numeric equality, instead of only creating the colored strings inside the original numeric-threshold branch.

- **[dependency] Q:** What dependency or import usage is affected by this change, and how is it used in the updated comparison code?
  **A:** The updated code continues to use `bcolors` for highlighting matching rows in `PrettyTable`, but now uses it unconditionally at the start of each loop iteration rather than only inside the numeric-approximation branch.

- **[scope] Q:** Which file is modified in this commit, and what part of that file changes?
  **A:** Only `chispa/column_comparer.py` is modified, specifically the loop inside `assert_approx_column_equality` that compares zipped column elements and builds the `PrettyTable` output.

- **[usage] Q:** How would a developer use the changed logic in practice when comparing two Spark DataFrame columns?
  **A:** A developer would still call `assert_approx_column_equality(df, col_name1, col_name2, precision)` the same way as before; the difference is that the function now safely handles rows where one or both column values are `None` during the approximate comparison.

- **[edge_case] Q:** What edge case does the new conditional logic handle, and how are rows with both values `None` treated?
  **A:** The edge case is when one column value is `None` and the other is not. In that case the row is marked unequal and shown as plain text. If both values are `None`, the row follows the equality branch and is added to the table using the highlighted `first` and `second` strings.

- **[risk] Q:** What backward-compatibility or regression risk does this change introduce in `assert_approx_column_equality`?
  **A:** The main risk is that comparisons involving `None` now have explicit semantics that differ from previous behavior; code that relied on a failure or exception from subtracting `None` during `abs(elements[0] - elements[1])` will now instead receive a structured `ColumnsNotEqualError` with a table showing the mismatched rows.

### MrPowers/chispa @ commit #6 `bed2f1afd7` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in chispa's dataframe comparison code?
  **A:** It adds approximate DataFrame equality checking based on a numeric precision tolerance, while also separating schema equality into its own helper so exact and approximate comparisons can share the same schema check.

- **[new_api] Q:** What new API is introduced by `chispa/dataframe_comparer.py`, and what does it do?
  **A:** The commit adds `assert_approx_df_equality(df1, df2, precision)`, which first verifies that `df1.schema` and `df2.schema` match, then compares collected rows with `are_rows_approx_equal` and raises `DataFramesNotEqualError` if any row pair differs beyond the given precision.

- **[signature_change] Q:** Which function signature or type annotation was added for row comparison, and what does it accept?
  **A:** A new typed helper `are_rows_equal(r1: pyspark.sql.Row, r2: pyspark.sql.Row) -> bool` was added. It accepts two Spark `Row` objects and returns whether they are exactly equal by using `r1 == r2`.

- **[behavior_change] Q:** How does runtime behavior differ for approximate DataFrame comparisons after this change?
  **A:** Approximate comparison now tolerates differences in float fields up to the provided `precision` value. When rows are considered approximately equal, they are still included in the pretty-printed table with light-blue/light-red coloring, and only rows that fail the tolerance check cause `DataFramesNotEqualError` to be raised.

- **[refactor] Q:** What existing construct was restructured in the comparer module, and why?
  **A:** The original exact schema check logic was extracted from `assert_df_equality` into a new `assert_schema_equality(df1, df2)` helper. `assert_df_equality` now delegates schema validation to that helper before continuing with exact row comparison, allowing the same schema check to be reused by the new approximate equality function.

- **[dependency] Q:** What import changes were made in `chispa/dataframe_comparer.py`, and what purpose do they serve?
  **A:** The file now imports `from chispa.bcolors import *` and `import pyspark`. The color import is used to highlight approximately equal rows in the diff output, and the `pyspark` import is used for the `pyspark.sql.Row` type annotation on `are_rows_equal`.

- **[scope] Q:** Which parts of the codebase are touched by this commit, and what changes are made in that file?
  **A:** Only `chispa/dataframe_comparer.py` is changed. The file adds color and PySpark imports, introduces `assert_schema_equality`, adds `are_rows_equal`, adds `are_rows_approx_equal`, and adds `assert_approx_df_equality` for precision-based DataFrame comparison.

- **[usage] Q:** How would a developer use the new approximate equality support added by this commit?
  **A:** A developer would call `assert_approx_df_equality(df1, df2, precision)` from `chispa.dataframe_comparer`, passing two DataFrames and a float precision threshold. The function compares schemas first and then checks each collected row pair, allowing float differences within the specified tolerance.

- **[edge_case] Q:** How does the new row approximation helper handle fields that are not both floats?
  **A:** `are_rows_approx_equal(r1, r2, precision)` converts each row to dictionaries and iterates over the intersection of their keys. For keys where both values are floats, it compares `abs(d1[key] - d2[key])` against `precision`; for all other shared keys, it falls back to exact inequality checks.

- **[risk] Q:** What backward-compatibility or regression risk is introduced by the new approximate comparison path?
  **A:** The new approximate comparison only checks keys in `d1.keys() & d2.keys()`, so mismatched keys outside that intersection are not directly validated in `are_rows_approx_equal`. Also, exact DataFrame equality still depends on `assert_schema_equality`, but the new approximate API may consider rows equal even when floating-point values differ, which can hide small discrepancies if callers use a loose precision.

### MrPowers/chispa @ commit #7 `d94ff473d4` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in `chispa`?
  **A:** The commit clarifies and separates the equality logic in `chispa/column_comparer.py` for approximate column comparison, especially the handling of `None` values versus numeric differences, and it removes the empty `chispa/row_comparer.py` stub file.

- **[new_api] Q:** What new callable or API is introduced by this change, and what does it do?
  **A:** No new function, method, or class is introduced in this diff. The only behavioral code change is a refactor of the conditional branches inside `assert_approx_column_equality(df, col_name1, col_name2, precision)` in `chispa/column_comparer.py`.

- **[signature_change] Q:** Did any function or method signature change in this commit, and if so, how?
  **A:** No function signature or type annotation changes are shown in the diff. `assert_approx_column_equality(df, col_name1, col_name2, precision)` keeps the same parameters and interface.

- **[behavior_change] Q:** How does the runtime behavior of approximate column comparison differ after this change?
  **A:** `assert_approx_column_equality` now evaluates the cases in separate branches: it first treats one-`None`/one-non-`None` pairs as unequal, then treats both-`None` pairs as equal, then checks `abs(elements[0] - elements[1]) < precision` for approximate equality. The previous version combined the both-`None` and approximate-equality checks in one condition.

- **[refactor] Q:** Why was the conditional logic in `assert_approx_column_equality` restructured?
  **A:** The condition was split into explicit branches to make the handling of `None` values and approximate numeric comparison clearer: one branch for mismatched `None`, one for both `None`, one for values within the precision threshold, and a final fallback for all other mismatches.

- **[dependency] Q:** What import or dependency change is made by this commit, and what file does it affect?
  **A:** No import or external dependency changes are shown. The only file-level dependency-like change is the deletion of `chispa/row_comparer.py`, which removed an empty `are_rows_equal` stub file.

- **[scope] Q:** Which files are touched in this commit, and what changes were made in each?
  **A:** Two files are touched: `chispa/column_comparer.py`, where the `assert_approx_column_equality` branching logic was reorganized and annotated with comments; and `chispa/row_comparer.py`, which was deleted and previously contained only an incomplete `def are_rows_equal:` stub.

- **[usage] Q:** How would a developer use the changed code after this commit?
  **A:** A developer would still call `assert_approx_column_equality(df, col_name1, col_name2, precision)` from `chispa/column_comparer.py` to compare two DataFrame columns. The comparison now explicitly treats pairs of `None` values as equal and mismatched `None` values as unequal before applying the absolute-difference precision check.

- **[edge_case] Q:** How are `None` values handled in the updated approximate column comparison logic?
  **A:** If one element is `None` and the other is not, `assert_approx_column_equality` marks the row as unequal. If both elements are `None`, the row is treated as equal and added to the table with colored string output, without attempting subtraction.

- **[risk] Q:** What backward-compatibility or regression risk does this change introduce?
  **A:** The main risk is that the deleted `chispa/row_comparer.py` file removes the `are_rows_equal` stub path entirely, which could break any code importing that module. In addition, the explicit `None` handling in `assert_approx_column_equality` could alter expectations for callers relying on the old combined condition or on how mixed `None`/numeric pairs were surfaced.

### MrPowers/chispa @ commit #8 `10845509b6` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the README update in this commit?
  **A:** The README was expanded from a short setup section into a full usage guide for chispa, showing how to use its PySpark test helpers such as `assert_column_equality` and `assert_df_equality` with example DataFrames and error screenshots.

- **[new_api] Q:** Which chispa helper functions are demonstrated as the primary APIs in the new README examples, and what do they verify?
  **A:** The README demonstrates `assert_column_equality(df, "clean_name", "expected_name")`, which verifies that two columns in a DataFrame match row by row, and `assert_df_equality(actual_df, expected_df)`, which verifies that two DataFrames are equal.

- **[signature_change] Q:** What callable signature is shown for the new column-equality helper in the README examples?
  **A:** The README shows `assert_column_equality` being called with three arguments: a DataFrame, the actual column name, and the expected column name, as in `assert_column_equality(df, "clean_name", "expected_name")`.

- **[behavior_change] Q:** How do the new examples describe the runtime behavior when an equality check fails?
  **A:** The README says a failing comparison returns a nicely formatted error message, and it shows that the mismatching row is highlighted in red while equal rows are colored blue in the output image.

- **[refactor] Q:** What existing content was restructured in the README, and how was it simplified or expanded?
  **A:** The old setup-only README section was replaced with a broader introduction and multiple topic sections, including Column equality, DataFrame equality, and placeholders for approximate equality, schema errors, benchmarks, limitations, and development instructions.

- **[dependency] Q:** What dependency-related setup change is visible in the README, and what does it replace?
  **A:** The README removes the previous conda/pip setup commands (`conda create`, `conda activate`, `pip install -r requirements.txt`, `python -m pytest`) and replaces them with a SparkSession setup example using `pyspark.sql.SparkSession`.

- **[scope] Q:** Which parts of the repository are touched by this commit, and what kind of change does each receive?
  **A:** Only `README.md` is modified in this commit, and it receives documentation changes: added project description, usage examples, failing-test examples, screenshots, and new section headings.

- **[usage] Q:** How does the README show a developer using chispa to test a string-cleaning transformation?
  **A:** It shows creating a Spark DataFrame, applying `remove_non_word_characters(F.col("name"))` with `.withColumn("clean_name", ...)`, and then calling `assert_column_equality` or `assert_df_equality` to compare the result against expected values.

- **[edge_case] Q:** What special input case is explicitly included in the README test data examples, and how is it expected to behave?
  **A:** The examples include `None` values in both the source and expected data, and the expected behavior is that `None` remains `None` after applying `remove_non_word_characters` and is treated as equal in the assertions.

- **[risk] Q:** What compatibility or regression risk could come from this change, based on what was updated in the README?
  **A:** Because the commit changes only documentation and not code, the main risk is confusion if the examples or API names in the README do not match the actual library behavior; for example, users may rely on the documented helper calls and Spark setup patterns.

### MrPowers/chispa @ commit #9 `c08a9d1171` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** It adds and documents approximate-equality support for PySpark DataFrames and columns, while also improving the formatting of schema and DataFrame equality error messages in `chispa/dataframe_comparer.py` and updating the README with installation and usage examples.

- **[new_api] Q:** What new helper function is introduced in `chispa/dataframe_comparer.py`, and what does it do?
  **A:** A new `blue(s: str) -> str` helper is added. It wraps a value with `bcolors.LightBlue` and `bcolors.LightRed` so matching schema or row values can be highlighted in equality error tables.

- **[signature_change] Q:** Which function signatures or type annotations were changed in `chispa/dataframe_comparer.py`?
  **A:** `are_rows_equal` and `are_rows_approx_equal` changed their parameter annotations from `pyspark.sql.Row` to `Row` by importing `Row` directly from `pyspark.sql` (`def are_rows_equal(r1: Row, r2: Row) -> bool` and `def are_rows_approx_equal(r1: Row, r2: Row, precision: float) -> bool`).

- **[behavior_change] Q:** How does the runtime behavior of the equality assertions change after this commit?
  **A:** `assert_schema_equality` and `assert_df_equality` now color matching items in their PrettyTable error output, and `assert_df_equality` uses renamed local variables `rows1` and `rows2` instead of the previous `df1e` and `df2e`. The comparison logic still raises `SchemasNotEqualError` or `DataFramesNotEqualError` when the collected rows or schemas differ.

- **[refactor] Q:** What existing code was restructured in `assert_df_equality`, and why is that significant?
  **A:** The collected DataFrame values were renamed from `df1e`/`df2e` to `rows1`/`rows2`, and the row comparison loop now unpacks each pair as `r1, r2`. This makes the code clearer because it explicitly treats the collected objects as rows.

- **[dependency] Q:** What import change was made in `chispa/dataframe_comparer.py`, and what does it replace?
  **A:** The file removed `import pyspark` and added `from pyspark.sql import Row`. This replaces the fully qualified `pyspark.sql.Row` type annotations with a direct `Row` import.

- **[scope] Q:** Which files were modified in this commit, and what kind of changes were made in each?
  **A:** `README.md` was expanded with installation instructions, approximate equality examples, schema mismatch messaging, and wording tweaks. `chispa/dataframe_comparer.py` was updated to add the `blue` helper, improve schema/dataframe error formatting, and change row type imports and annotations.

- **[usage] Q:** How does the updated README show a developer should call the approximate equality helpers?
  **A:** It shows calling `assert_approx_column_equality(df, "num1", "num2", 0.1)` for column comparisons and `assert_approx_df_equality(df1, df2, 0.1)` for DataFrame comparisons, using `0.1` as the precision threshold in the examples.

- **[edge_case] Q:** What edge-case handling is documented for approximate DataFrame equality in the README?
  **A:** The README says `assert_approx_df_equality` only performs approximate comparisons for floating-point numbers in DataFrames; it uses regular equality for strings and other types. The examples also include `(None, None)` rows in both matching and non-matching cases.

- **[risk] Q:** What compatibility or regression concern could arise from this change to equality error output?
  **A:** Because matching schema fields and equal rows are now colorized in the PrettyTable output, any tests or tools that assert on the exact error-message string may need updates. The use of direct `Row` imports also assumes `pyspark.sql.Row` remains available from that location.

### MrPowers/chispa @ commit #10 `b413aa2cf8` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the change to `chispa/__init__.py` in this commit?
  **A:** It makes the package’s comparison helpers available from the top-level `chispa` namespace by importing the dataframe and column assertion functions and their error types into `chispa.__init__`.

- **[new_api] Q:** Which new top-level `chispa` API is exposed by `chispa/__init__.py`, and what does it include?
  **A:** The top-level `chispa` package now exposes `assert_schema_equality`, `assert_df_equality`, `assert_approx_df_equality`, `assert_column_equality`, `assert_approx_column_equality`, plus the error classes `DataFramesNotEqualError`, `SchemasNotEqualError`, and `ColumnsNotEqualError`.

- **[signature_change] Q:** Did this commit change any function signatures or type annotations, and what is the only code change visible in the diff?
  **A:** No function signatures or type annotations are changed in the diff; the only code change is adding import statements in `chispa/__init__.py` to re-export existing names.

- **[behavior_change] Q:** How does runtime behavior change for code that imports `chispa` after this commit?
  **A:** After this commit, importing `chispa` makes the comparison functions and error classes directly accessible from the package root, so callers can reference them as `chispa.assert_df_equality`, `chispa.assert_column_equality`, and similar names without importing submodules.

- **[refactor] Q:** Why was `chispa/__init__.py` restructured in this commit instead of leaving the package empty?
  **A:** It was populated to simplify the package interface by re-exporting the main dataframe and column comparison utilities from a single entry point instead of requiring users to import from `chispa.dataframe_comparer` or `chispa.column_comparer`.

- **[dependency] Q:** What internal modules does `chispa/__init__.py` now depend on, and why were they imported?
  **A:** `chispa/__init__.py` now depends on `chispa.dataframe_comparer` and `chispa.column_comparer` so it can import and re-export their public assertion functions and error classes at the package level.

- **[scope] Q:** Which file is changed in this commit, and what kind of change does it contain?
  **A:** Only `chispa/__init__.py` is changed, and it contains two import lines that re-export names from `dataframe_comparer` and `column_comparer`.

- **[usage] Q:** How would a developer use the newly exposed package-level API after this change?
  **A:** A developer can import directly from `chispa`, for example `from chispa import assert_df_equality, assert_column_equality, DataFramesNotEqualError`, instead of importing those names from the underlying comparer modules.

- **[edge_case] Q:** What happens for callers who rely on the package root being empty or minimal, and what exact names are now present there?
  **A:** The package root is no longer empty; it now contains `DataFramesNotEqualError`, `SchemasNotEqualError`, `assert_schema_equality`, `assert_df_equality`, `assert_approx_df_equality`, `ColumnsNotEqualError`, `assert_column_equality`, and `assert_approx_column_equality`.

- **[risk] Q:** What backward-compatibility concern could this import-only change introduce for existing users?
  **A:** The main compatibility risk is namespace expansion: code that previously treated `chispa` as having no top-level exports may now see new attributes, and these re-exports also make the package root depend on the imported submodules loading successfully.

### MrPowers/chispa @ commit #11 `4ffa792194` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** It refactors DataFrame and schema comparison code into separate modules and adds an `ignore_nullable` path so `assert_df_equality` can compare DataFrame schemas while ignoring field nullability when requested.

- **[new_api] Q:** What new helper function is introduced in `chispa/schema_comparer.py`, and what does it do?
  **A:** The new function `are_schemas_equal_ignore_nullable(s1, s2)` compares two Spark schemas by checking that they have the same length, and that each corresponding field has the same `name` and `dataType`, while ignoring nullable metadata.

- **[signature_change] Q:** How did the signature of `assert_df_equality` change in `chispa/dataframe_comparer.py`?
  **A:** `assert_df_equality` changed from `assert_df_equality(df1, df2)` to `assert_df_equality(df1, df2, ignore_nullable = False)`, adding an optional `ignore_nullable` boolean parameter with a default of `False`.

- **[behavior_change] Q:** How does the runtime behavior of `assert_df_equality` differ after this change when `ignore_nullable` is enabled?
  **A:** When `ignore_nullable` is `True`, `assert_df_equality` calls `assert_schema_equality_ignore_nullable(df1, df2)` instead of strict schema equality, so schemas with matching field names and data types but different nullable settings can pass the schema check before row comparison.

- **[refactor] Q:** Why were schema and row comparison helpers moved out of `chispa/dataframe_comparer.py`?
  **A:** They were split into dedicated modules: schema-related logic moved to `chispa/schema_comparer.py`, and row comparison helpers moved to `chispa/row_comparer.py`, simplifying `dataframe_comparer.py` to focus on DataFrame-level assertions.

- **[dependency] Q:** What import changed in `chispa/dataframe_comparer.py`, and what functionality does it now rely on externally?
  **A:** `chispa/dataframe_comparer.py` now imports `assert_schema_equality`, `assert_schema_equality_ignore_nullable`, and `are_schemas_equal_ignore_nullable` from `chispa.schema_comparer`, plus `are_rows_approx_equal` from `chispa.row_comparer`, instead of defining those helpers inline.

- **[scope] Q:** Which files were changed in this commit, and what kind of changes did each file receive?
  **A:** `chispa/__init__.py` removed schema assertion exports, `chispa/bcolors.py` added a top-level `blue(s: str)` helper, `chispa/dataframe_comparer.py` was updated to use the new schema helpers and an `ignore_nullable` parameter, and two new files were added: `chispa/row_comparer.py` for row comparison helpers and `chispa/schema_comparer.py` for schema comparison and the `SchemasNotEqualError` exception.

- **[usage] Q:** How would a developer use the new nullable-ignoring comparison path exposed by this change?
  **A:** They would call `assert_df_equality(df1, df2, ignore_nullable=True)` from `chispa.dataframe_comparer` or from the package import path if re-exported, and the function will validate schemas with `assert_schema_equality_ignore_nullable` before comparing collected rows.

- **[edge_case] Q:** What edge case does `are_schemas_equal_ignore_nullable` explicitly handle when comparing schemas?
  **A:** It returns `False` if the two schemas have different lengths, or if any paired fields differ in `name` or `dataType`; nullable differences are the case it intentionally ignores.

- **[risk] Q:** What backward-compatibility or regression risk does this commit introduce in `chispa/__init__.py`?
  **A:** `chispa/__init__.py` no longer re-exports `SchemasNotEqualError` or `assert_schema_equality`, so code importing those names directly from `chispa` will break unless it switches to `chispa.schema_comparer`.

### MrPowers/chispa @ commit #12 `293dce310c` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the dataframe and schema comparison helpers?
  **A:** The commit refactors the comparison helpers to operate on schema objects directly instead of DataFrame objects, by extracting `df.schema` in `chispa/dataframe_comparer.py` and changing `assert_schema_equality` and `assert_schema_equality_ignore_nullable` in `chispa/schema_comparer.py` to accept schemas.

- **[new_api] Q:** What new way do the schema comparison functions support after this change, and what do they do with their inputs?
  **A:** `assert_schema_equality(s1, s2)` and `assert_schema_equality_ignore_nullable(s1, s2)` now take schema objects directly rather than DataFrames. They compare the provided schemas and, when they differ, build a `PrettyTable` showing `schema1` and `schema2`.

- **[signature_change] Q:** How did the function signatures change for the schema comparison helpers?
  **A:** `assert_schema_equality` changed from `assert_schema_equality(df1, df2)` to `assert_schema_equality(s1, s2)`, and `assert_schema_equality_ignore_nullable` changed from `assert_schema_equality_ignore_nullable(df1, df2)` to `assert_schema_equality_ignore_nullable(s1, s2)`. The callers in `assert_df_equality` and `assert_approx_df_equality` now pass `df.schema` values.

- **[behavior_change] Q:** How does the runtime comparison behavior differ after this change when checking DataFrame equality?
  **A:** `assert_df_equality` and `assert_approx_df_equality` now first store `df1.schema` and `df2.schema` in `s1` and `s2`, then compare those schema objects directly before collecting rows. The row comparison logic remains the same, but schema checks no longer rely on passing whole DataFrames into the schema helpers.

- **[refactor] Q:** Why was the code in the comparer modules restructured around schema objects instead of DataFrames?
  **A:** The structure was simplified so schema-only helpers in `chispa/schema_comparer.py` no longer need to extract `.schema` themselves. This makes the responsibilities clearer: `chispa/dataframe_comparer.py` extracts schemas from DataFrames, and `chispa/schema_comparer.py` compares schema inputs.

- **[dependency] Q:** What import was added to support the schema comparison changes, and how is it used?
  **A:** `import six` was added in `chispa/schema_comparer.py`. It is used with `six.moves.zip_longest` to iterate over schema fields even when the two schemas have different lengths.

- **[scope] Q:** Which files were modified, and what kind of changes were made in each one?
  **A:** `chispa/dataframe_comparer.py` was updated to extract `df.schema` into local variables and pass schemas into schema comparison helpers. `chispa/schema_comparer.py` was updated so the helpers accept schema objects directly, and it also added `six` and switched field iteration to `zip_longest`.

- **[usage] Q:** How would a caller use the changed comparison flow after this commit?
  **A:** A caller still invokes `assert_df_equality(df1, df2, ignore_nullable=False)` or `assert_approx_df_equality(df1, df2, precision)`, but those functions now internally call `assert_schema_equality(df1.schema, df2.schema)` or `assert_schema_equality_ignore_nullable(df1.schema, df2.schema)` before comparing collected rows.

- **[edge_case] Q:** What edge-case handling was added for schemas of different lengths?
  **A:** `assert_schema_equality`, `assert_schema_equality_ignore_nullable`, and `are_schemas_equal_ignore_nullable` now use `six.moves.zip_longest`, which allows them to pair fields even when one schema has more fields than the other instead of stopping at the shorter length.

- **[risk] Q:** What compatibility or regression risk is introduced by changing these helpers to accept schemas instead of DataFrames?
  **A:** Any external code that called `assert_schema_equality(df1, df2)` or `assert_schema_equality_ignore_nullable(df1, df2)` with DataFrames would now pass the wrong type, because the functions expect schema objects. That makes the change backward-incompatible for direct callers of those helpers.

### MrPowers/chispa @ commit #13 `608b51cba9` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** The commit vendors the PrettyTable dependency into `chispa/prettytable.py` and switches chispa’s comparers to import `PrettyTable` from the local package instead of the external `prettytable` installation, reducing version conflicts.

- **[new_api] Q:** What new module and class are introduced by this change, and what do they provide?
  **A:** The change adds `chispa/prettytable.py`, which contains a vendored `PrettyTable` class. It provides table formatting and rendering functionality used by chispa’s error-reporting code.

- **[signature_change] Q:** How does the `PrettyTable` constructor signature used by chispa change after this commit?
  **A:** The call site changes from importing `PrettyTable` from the external `prettytable` package to importing `PrettyTable` from `chispa.prettytable`, but the constructor remains `PrettyTable(field_names=None, **kwargs)` in the vendored module.

- **[behavior_change] Q:** How does runtime behavior differ when chispa builds its comparison error tables after this change?
  **A:** At runtime, chispa now uses the vendored `chispa.prettytable.PrettyTable` implementation to format comparison output instead of relying on whatever `prettytable` version happens to be installed in the environment.

- **[refactor] Q:** Why were the imports in the comparer modules restructured in this commit?
  **A:** The imports in `chispa/column_comparer.py` and `chispa/dataframe_comparer.py` were changed from `from prettytable import PrettyTable` to `from chispa.prettytable import PrettyTable` so chispa can depend on its local vendored copy rather than an external package.

- **[dependency] Q:** What dependency-related change was made, and how is it documented?
  **A:** `PrettyTable` is vendored into `chispa/prettytable.py`, and `README.md` adds a "Vendored dependencies" section listing `six` and `PrettyTable` as vendored to minimize version conflicts.

- **[scope] Q:** Which files are touched by this commit, and what kind of change does each file receive?
  **A:** `README.md` is updated with a vendored-dependencies note; `chispa/column_comparer.py` and `chispa/dataframe_comparer.py` change their `PrettyTable` import paths; and `chispa/prettytable.py` is added as a new vendored module containing the PrettyTable implementation.

- **[usage] Q:** How should a developer use the changed code to get a PrettyTable instance from chispa now?
  **A:** A developer should import it with `from chispa.prettytable import PrettyTable` and then construct it the same way as before, for example `PrettyTable(field_names=[...])`.

- **[edge_case] Q:** What default-handling behavior for the new vendored `PrettyTable` class is visible in the diff?
  **A:** The vendored `PrettyTable.__init__` sets defaults when keyword arguments are omitted: `encoding` defaults to `UTF-8`, `header` and `border` default to `True`, `reversesort` defaults to `False`, `print_empty` defaults to `True`, `oldsortslice` defaults to `False`, and `padding_width` defaults to `1`.

- **[risk] Q:** What compatibility risk does this vendoring change introduce for downstream users?
  **A:** Because chispa now uses its own `chispa.prettytable` implementation, any behavior differences from the externally installed `prettytable` package could change table formatting or error output for downstream code that depended on the previous third-party version.

### MrPowers/chispa @ commit #14 `ca279766f4` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in `chispa/dataframe_comparer.py`?
  **A:** It changes DataFrame comparison to handle unequal row counts more safely by using `zip_longest` instead of `zip`, so missing rows are detected and reported during equality and approximate-equality checks.

- **[new_api] Q:** Did this commit introduce any new public function or class, and what new helper is used in the comparison logic?
  **A:** No new public function or class was added. The change introduces use of `chispa.six` and, specifically, `six.moves.zip_longest` to iterate over rows even when the two DataFrames have different lengths.

- **[signature_change] Q:** Did the signatures of `assert_df_equality` or `assert_approx_df_equality` change in this commit?
  **A:** No. Both functions keep the same call signatures; the changes are internal to how they pair collected rows for comparison.

- **[behavior_change] Q:** How does runtime behavior change when one DataFrame has more collected rows than the other?
  **A:** The code now compares all rows using `six.moves.zip_longest`, so extra rows on either side are surfaced as mismatches instead of being silently ignored by `zip`, which stopped at the shorter input.

- **[refactor] Q:** Why was the row-pairing logic restructured in `assert_df_equality` and `assert_approx_df_equality`?
  **A:** It was simplified to use `zip_longest` for row pairing, which makes the comparison logic handle uneven DataFrame lengths consistently and reduces the risk of missing trailing rows.

- **[dependency] Q:** What import was added, and what does it enable in this file?
  **A:** `import chispa.six as six` was added so the code can use `six.moves.zip_longest`, providing a compatible way to iterate over both row collections until the longer one is exhausted.

- **[scope] Q:** Which file is modified by this commit, and what kinds of changes does it contain there?
  **A:** Only `chispa/dataframe_comparer.py` is changed, and the edits are limited to importing `chispa.six` and switching both equality functions from `zip` to `zip_longest`, plus adding explicit handling for one-sided `None` rows in approximate comparison.

- **[usage] Q:** How should a developer call the changed comparison code after this commit?
  **A:** The developer still calls the same APIs, such as `assert_df_equality(df1, df2, ignore_nullable=False)` and `assert_approx_df_equality(df1, df2, precision)`, with no new arguments required.

- **[edge_case] Q:** How does `assert_approx_df_equality` handle a row-pair where one side is missing because the DataFrames have different lengths?
  **A:** It checks for the edge case where one row is `None` and the other is not, marks the rows as unequal, and adds the pair directly to the `PrettyTable` instead of passing `None` into `are_rows_approx_equal`.

- **[risk] Q:** What backward-compatibility or regression risk does this change introduce for existing DataFrame comparisons?
  **A:** Comparisons that previously passed because `zip` ignored trailing rows may now fail, since extra rows are now included in the comparison and reported as mismatches.

### MrPowers/chispa @ commit #15 `168660cc2a` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the change in `chispa/row_comparer.py`?
  **A:** It adds explicit `None` handling to `are_rows_approx_equal` so that comparing two `None` rows returns `True`, and comparing one `None` row with a non-`None` row returns `False` before calling `asDict()`.

- **[new_api] Q:** What new public function behavior was introduced in `are_rows_approx_equal`, and what does it do for `None` inputs?
  **A:** `are_rows_approx_equal(r1: Row, r2: Row, precision: float) -> bool` now treats `None` as a valid input: it returns `True` when both `r1` and `r2` are `None`, and `False` when exactly one of them is `None`.

- **[signature_change] Q:** Did the signature or type annotation of `are_rows_approx_equal` change in this commit?
  **A:** No. The function signature remains `are_rows_approx_equal(r1: Row, r2: Row, precision: float) -> bool`; only the runtime checks at the start of the function were added.

- **[behavior_change] Q:** How does the runtime behavior of `are_rows_approx_equal` differ after this patch when given `None` values?
  **A:** Before the patch, the function would proceed to `r1.asDict()` and `r2.asDict()`, which would fail for `None`. After the patch, it returns immediately with `True` for `(None, None)` and `False` for mixed `None`/non-`None` inputs.

- **[refactor] Q:** Why was the start of `are_rows_approx_equal` restructured in this commit?
  **A:** It was restructured to short-circuit on `None` cases before the existing dictionary comparison logic, avoiding unnecessary calls to `asDict()` and handling null rows explicitly.

- **[dependency] Q:** What import or dependency change was made to support this patch?
  **A:** No import or dependency changes are shown in the diff. The change is limited to adding `None` checks inside `chispa/row_comparer.py`.

- **[scope] Q:** Which file is modified in this commit, and what kind of change does it contain?
  **A:** Only `chispa/row_comparer.py` is modified, and the change adds three `None`-handling guard clauses at the beginning of `are_rows_approx_equal`.

- **[usage] Q:** How would a developer use the updated `are_rows_approx_equal` with nullable rows?
  **A:** They can call `are_rows_approx_equal(r1, r2, precision)` even when `r1` or `r2` may be `None`; the function now directly returns `True` for two `None` rows and `False` when only one side is `None`.

- **[edge_case] Q:** What edge case does the new code handle before comparing row contents?
  **A:** It handles the empty/null-row edge case where both inputs are `None` or only one input is `None`, returning a deterministic boolean without attempting `asDict()`.

- **[risk] Q:** What compatibility risk might this change introduce for existing callers of `are_rows_approx_equal`?
  **A:** Callers that previously expected `None` inputs to raise an error may now receive a boolean result instead, which could mask invalid input and change test expectations for nullable rows.

### MrPowers/chispa @ commit #17 `d13560da47` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in `chispa`, based on the code changes shown?
  **A:** The commit extends `assert_df_equality` in `chispa/dataframe_comparer.py` so callers can supply a list of DataFrame transformation functions that are applied to both input DataFrames before schema and row equality checks run.

- **[new_api] Q:** What new capability was added to `assert_df_equality` in `chispa/dataframe_comparer.py`, and how does it work?
  **A:** `assert_df_equality` now accepts a new `transforms` argument, and it uses `functools.reduce` to apply each function in that list to `df1` and `df2` before comparing them.

- **[signature_change] Q:** How did the signature of `assert_df_equality` change in `chispa/dataframe_comparer.py`?
  **A:** The function signature changed from `assert_df_equality(df1, df2, ignore_nullable=False)` to `assert_df_equality(df1, df2, ignore_nullable=False, transforms=[])`, adding a fourth parameter named `transforms` with a default empty list.

- **[behavior_change] Q:** How does `assert_df_equality` behave differently at runtime after this change?
  **A:** Before performing schema comparison, it now transforms both input DataFrames by sequentially applying every function in `transforms`. This means equality is checked on the transformed DataFrames rather than on the original inputs.

- **[refactor] Q:** Why was `assert_df_equality` restructured in `chispa/dataframe_comparer.py`?
  **A:** It was restructured to factor transformation logic into a reusable preprocessing step using `reduce`, so the same sequence of DataFrame-modifying functions can be applied to both inputs before the existing comparison logic runs.

- **[dependency] Q:** What import was added to support the new DataFrame transformation behavior, and why?
  **A:** `from functools import reduce` was added in `chispa/dataframe_comparer.py` so the code can fold over the `transforms` list and apply each transformation function in order to `df1` and `df2`.

- **[scope] Q:** Which files were changed in this commit, and what kind of change did each file receive?
  **A:** `chispa/dataframe_comparer.py` was updated to add the `transforms` parameter and apply transformations before comparison. `README.md` was updated to add a CI status badge. `LICENSE.md` was added as a new MIT license file.

- **[usage] Q:** How would a developer use the updated `assert_df_equality` function to compare transformed DataFrames?
  **A:** A developer can pass a list of transformation functions via `transforms`, for example `assert_df_equality(df1, df2, transforms=[fn1, fn2])`, and each function will be applied to both DataFrames before equality is asserted.

- **[edge_case] Q:** What default behavior applies if no transformation functions are provided to `assert_df_equality`?
  **A:** `transforms` defaults to an empty list, so `reduce` returns the original `df1` and `df2` unchanged and the function behaves like the previous version without preprocessing.

- **[risk] Q:** What compatibility or correctness risk is introduced by the new `transforms=[]` parameter in `assert_df_equality`?
  **A:** The new default uses a mutable list object, which can be a backward-compatibility and maintainability risk if the default were ever mutated. More importantly, any provided transform functions are now applied to both DataFrames before comparison, so an incorrect or non-deterministic transform could change existing test outcomes.

### MrPowers/chispa @ commit #18 `e2469ef4ba` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the changes in `chispa/__init__.py`?
  **A:** The change adds startup logic to make PySpark importable when `chispa` is imported by trying `findspark` first and, if needed, manually extending `sys.path` from `SPARK_HOME` so the package can locate Spark and `py4j` before exposing `chispa`'s public comparison helpers.

- **[new_api] Q:** What new initialization behavior is introduced at the top level of `chispa`, and what does it do?
  **A:** `chispa/__init__.py` now performs an import-time Spark bootstrap: it tries `from pyspark import context`, then falls back to `findspark.init()`, and if that is unavailable it uses `SPARK_HOME` to append Spark's `python` directory and the `py4j` source zip to `sys.path`.

- **[signature_change] Q:** Did this commit change any function or method signatures in the public comparison API?
  **A:** No function or method signatures are changed in the exported API; the diff only adds import-time setup code in `chispa/__init__.py` and leaves `assert_df_equality`, `assert_approx_df_equality`, `assert_column_equality`, and `assert_approx_column_equality` imported with the same names.

- **[behavior_change] Q:** How does runtime behavior change when someone imports `chispa` after this commit?
  **A:** Importing `chispa` now may automatically configure Spark dependencies before the package's comparison functions are exposed: it can load PySpark directly, initialize `findspark`, or manually update `sys.path` using `SPARK_HOME`; previously `__init__.py` only re-exported the comparison helpers.

- **[refactor] Q:** Why was `chispa/__init__.py` restructured instead of leaving only the re-export imports there?
  **A:** It was restructured so the package can self-initialize its Spark-related environment at import time, making the re-exported comparison utilities usable even when PySpark is not already on the Python path.

- **[dependency] Q:** Which new dependency-related imports are added, and what purpose do they serve?
  **A:** `chispa/__init__.py` now imports `sys`, `os`, and `glob` from the standard library, and it conditionally imports `findspark`; these are used to locate Spark installation files, modify `sys.path`, and initialize PySpark support if it is not already available.

- **[scope] Q:** Which file is changed, and what kind of modification does it receive?
  **A:** Only `chispa/__init__.py` is modified, and the change is an import-time environment bootstrap added before the existing public re-exports of dataframe and column comparison helpers.

- **[usage] Q:** How would a developer rely on this change when using `chispa` in a project without PySpark already on the path?
  **A:** A developer can import `chispa` as usual, and the package will first try to resolve PySpark automatically via `findspark` or, failing that, via `SPARK_HOME`; no new public function call is introduced for this setup.

- **[edge_case] Q:** What happens if `SPARK_HOME` is set but the `py4j` source archive cannot be found under it?
  **A:** The code builds a `py4j_src_zip` glob under `SPARK_HOME/python/lib`; if no matching archive is found, it raises `ValueError('py4j source archive not found in %s' % os.path.join(spark_home, 'python', 'lib'))`.

- **[risk] Q:** What backward-compatibility or regression risk does the new import-time setup introduce?
  **A:** Because `chispa` now performs environment detection and can print an error and call `exit(-1)` when `SPARK_HOME` is missing, importing the package can fail earlier than before in environments without PySpark, `findspark`, or a correctly configured Spark installation.

### MrPowers/chispa @ commit #20 `1a469ae2ae` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this change in the repository?
  **A:** It adds README documentation showing how to perform order-independent DataFrame comparisons with `assert_df_equality`, including examples for ignoring row order and column order.

- **[new_api] Q:** What new API usage is introduced in the README, and what does it do?
  **A:** The README introduces `assert_df_equality(df1, df2, transforms=[lambda df: df.sort(df.columns)])` and `assert_df_equality(df1, df2, transforms=[lambda df: df.select(sorted(df.columns))])`, which compare DataFrames after applying a transform that normalizes row order or column order.

- **[signature_change] Q:** Did this commit change any function or method signatures in the code shown, and what interface is documented instead?
  **A:** No code signature is changed in the diff; the README documents the existing `assert_df_equality(df1, df2, transforms=...)` call pattern and shows that it accepts a `transforms` argument containing a list of DataFrame-transforming lambdas.

- **[behavior_change] Q:** How does the documented comparison behavior differ after this change?
  **A:** The README now explains that DataFrame equality can be checked while ignoring row order by sorting rows with `df.sort(df.columns)`, or ignoring column order by reselecting columns in sorted order with `df.select(sorted(df.columns))` before comparison.

- **[refactor] Q:** What existing concept was restructured in the documentation, and why?
  **A:** The equality example was extended with an 'Order independent DataFrame comparisons' section to organize a common comparison pattern separately from the earlier standard equality examples.

- **[dependency] Q:** What dependency or import-related change is reflected in the diff, and why was it made?
  **A:** There is no import or dependency change in the code; the diff only adds README examples that rely on the existing DataFrame methods `sort` and `select` plus Python's `sorted()` function.

- **[scope] Q:** Which file is modified by this commit, and what kind of change does it contain?
  **A:** Only `README.md` is modified, and the change is documentation: a new section titled 'Order independent DataFrame comparisons' with code examples.

- **[usage] Q:** How would a developer use the documented approach to compare two DataFrames while ignoring row order?
  **A:** They would call `assert_df_equality(df1, df2, transforms=[lambda df: df.sort(df.columns)])`, which sorts each DataFrame by all of its columns before comparison.

- **[edge_case] Q:** What default or edge-case handling is implied for DataFrames whose row or column order should not matter?
  **A:** The README shows that order sensitivity can be neutralized by transforming the DataFrames first: use `df.sort(df.columns)` for row-order differences and `df.select(sorted(df.columns))` for column-order differences.

- **[risk] Q:** What potential compatibility concern could arise from the documented order-independent comparison approach?
  **A:** Because the approach depends on transforming DataFrames before equality checks, users must ensure the chosen transform preserves the intended data semantics; for example, sorting rows by all columns or reordering columns with `sorted(df.columns)` may not match every DataFrame shape or comparison expectation.

### MrPowers/chispa @ commit #21 `e793de98b3` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the DataFrame comparison utilities?
  **A:** It adds optional NaN-aware equality handling to DataFrame and Row comparison, so `assert_df_equality` can treat matching `NaN` values as equal when `allow_nan_equality=True`.

- **[new_api] Q:** What new helper function is introduced in `chispa/number_helpers.py`, and what does it do?
  **A:** The new function `nan_safe_equality(x, y) -> bool` returns `True` when `x == y` or when both values are NaN, using the helper `isnan(x)` to detect NaN values.

- **[signature_change] Q:** How did the signature of `assert_df_equality` change in `chispa/dataframe_comparer.py`?
  **A:** `assert_df_equality` changed from `def assert_df_equality(df1, df2, ignore_nullable=False, transforms=[])` to `def assert_df_equality(df1, df2, ignore_nullable=False, transforms=[], allow_nan_equality=False)`, adding the `allow_nan_equality` boolean parameter.

- **[behavior_change] Q:** How does `assert_df_equality` behave differently when `allow_nan_equality` is enabled?
  **A:** When `allow_nan_equality=True`, it compares collected rows with `are_rows_equal_enhanced(r1, r2, True)` and considers rows equal if corresponding NaN values match, instead of relying on plain `rows1 != rows2` equality.

- **[refactor] Q:** Why was `are_rows_equal_enhanced` added alongside the existing row comparison function?
  **A:** It separates the new NaN-aware comparison logic from the original `are_rows_equal(r1, r2)` behavior, letting callers choose between standard row equality and an enhanced comparison that treats NaN values as equal.

- **[dependency] Q:** What import changes were made in `chispa/dataframe_comparer.py` and `chispa/row_comparer.py`?
  **A:** `chispa/dataframe_comparer.py` changed `from chispa.row_comparer import are_rows_approx_equal` to `from chispa.row_comparer import *`, and `chispa/row_comparer.py` now imports `isnan` and `nan_safe_equality` from the new `chispa.number_helpers` module.

- **[scope] Q:** Which files were touched by this commit, and what kind of change did each file receive?
  **A:** `chispa/dataframe_comparer.py` gained the `allow_nan_equality` path in `assert_df_equality`; `chispa/row_comparer.py` added `are_rows_equal_enhanced`; `chispa/number_helpers.py` was added with NaN helpers; and `chispa/schema_comparer.py` changed its nullable-assertion diff formatting to use `six.moves.zip_longest`.

- **[usage] Q:** How would a developer call the updated DataFrame comparison function to enable NaN-aware equality?
  **A:** They would call `assert_df_equality(df1, df2, allow_nan_equality=True)`; the new flag enables the comparison path that treats matching NaN values as equal.

- **[edge_case] Q:** What edge-case handling was added for `None` and NaN values in row comparison?
  **A:** `are_rows_equal_enhanced` returns `True` when both rows are `None`, returns `False` when only one row is `None`, and with `allow_nan_equality=True` it uses `nan_safe_equality` so matching NaN values in overlapping row fields are treated as equal.

- **[risk] Q:** What compatibility or regression risk is introduced by the new equality path?
  **A:** The new `allow_nan_equality` branch changes how row differences are detected and reported, so callers that enable it may get different pass/fail results than before for data containing NaN values, and the use of `from chispa.row_comparer import *` also broadens the imported surface in `dataframe_comparer.py`.

### MrPowers/chispa @ commit #22 `b1d5cef931` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in `chispa/dataframe_comparer.py`?
  **A:** It adds configurable DataFrame comparison options for ignoring column order and row order, and it refactors equality checks to use a generic row-comparison helper for approximate and NaN-aware comparisons.

- **[new_api] Q:** What new helper function was introduced in `chispa/dataframe_comparer.py`, and what does it do?
  **A:** The new function `assert_generic_df_equality(df1, df2, row_equality_fun, row_equality_fun_args)` compares two DataFrames by first checking schema equality, then collecting rows and comparing them with a supplied row-comparison function and its arguments, raising `DataFramesNotEqualError` when rows differ.

- **[signature_change] Q:** How did the signature of `assert_df_equality` change in `chispa/dataframe_comparer.py`?
  **A:** `assert_df_equality` changed from `assert_df_equality(df1, df2, ignore_nullable=False, transforms=[], allow_nan_equality=False)` to `assert_df_equality(df1, df2, ignore_nullable=False, transforms=None, allow_nan_equality=False, ignore_column_order=False, ignore_row_order=False)`, replacing the mutable default list and adding two new boolean parameters.

- **[behavior_change] Q:** How does DataFrame comparison behavior differ after this change when `ignore_column_order` or `ignore_row_order` is enabled?
  **A:** When `ignore_column_order` is `True`, `assert_df_equality` adds a transform that selects columns in sorted order; when `ignore_row_order` is `True`, it adds a transform that sorts rows by `df.columns` before comparison. This makes equality checks insensitive to column ordering and/or row ordering.

- **[refactor] Q:** What existing comparison logic was simplified or centralized in this commit?
  **A:** The approximate- and NaN-aware row comparison paths were centralized through `assert_generic_df_equality`. `assert_approx_df_equality` now delegates to that helper, and the `allow_nan_equality` branch in `assert_df_equality` also uses it instead of duplicating row-by-row comparison and table formatting logic.

- **[dependency] Q:** What import or dependency-related change was made in `chispa/number_helpers.py`, and why does it matter?
  **A:** The broad `except:` in `isnan` was narrowed to `except TypeError:`. This keeps `isnan` returning `False` for non-numeric values that do not support `math.isnan`, while avoiding swallowing unrelated exceptions.

- **[scope] Q:** Which files were modified in this commit, and what changed in each one?
  **A:** `chispa/dataframe_comparer.py` was updated to add `ignore_column_order` and `ignore_row_order`, introduce `assert_generic_df_equality`, and route NaN/approximate comparisons through it. `chispa/number_helpers.py` was changed to catch only `TypeError` in `isnan`.

- **[usage] Q:** How would a developer use the new comparison options added to `assert_df_equality`?
  **A:** They can call `assert_df_equality(df1, df2, ignore_column_order=True)` to ignore column ordering, `assert_df_equality(df1, df2, ignore_row_order=True)` to ignore row ordering, or set both flags to make the comparison insensitive to both dimensions.

- **[edge_case] Q:** What default or edge-case handling changed for the `transforms` parameter in `assert_df_equality`?
  **A:** `transforms` now defaults to `None` instead of an empty list, and the function initializes it to `[]` internally when needed. This avoids sharing a mutable default list across calls.

- **[risk] Q:** What compatibility or regression risk does the new row/column-order handling introduce?
  **A:** The new `ignore_row_order` behavior sorts rows with `df.sort(df.columns)`, which assumes all columns are sortable and may change how duplicate or complex rows are ordered before comparison. Also, the new `transforms=None` default changes the public signature of `assert_df_equality`, which could affect callers relying on the old parameter list.

### MrPowers/chispa @ commit #23 `4db4ddc187` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in chispa's comparison helpers?
  **A:** It refactors DataFrame and schema comparison so nullable handling is centralized, and it adds a dedicated struct-field comparison helper to compare fields while optionally ignoring nullability.

- **[new_api] Q:** What new function is introduced in `chispa/structfield_comparer.py`, and what does it do?
  **A:** The new function is `are_structfields_equal(sf1, sf2, ignore_nullability=False)`. It compares two Spark struct fields either by full equality or, when `ignore_nullability=True`, by checking that both fields exist and have the same `name` and `dataType` while ignoring nullability-related differences.

- **[signature_change] Q:** Which existing comparison functions had their signatures changed, and what new parameter was added?
  **A:** `assert_schema_equality` in `chispa/schema_comparer.py` now accepts `ignore_nullable=False`, and `assert_approx_df_equality` in `chispa/dataframe_comparer.py` now accepts `ignore_nullable=False` as well. `assert_approx_df_equality` also now calls schema equality before comparing rows.

- **[behavior_change] Q:** How does `assert_df_equality` behave differently at runtime after this change?
  **A:** `assert_df_equality` now always applies the accumulated `transforms` list to both DataFrames, then checks schema equality through `assert_schema_equality(df1.schema, df2.schema, ignore_nullable)`, and finally either compares rows with `assert_generic_rows_equality` when `allow_nan_equality` is true or uses the new `assert_basic_rows_equality` for exact row comparison.

- **[refactor] Q:** What comparison logic was split out of `assert_generic_df_equality`, and why does that matter structurally?
  **A:** The row-collection and row-diff formatting logic that used to live in `assert_generic_df_equality` was renamed to `assert_generic_rows_equality`, while exact row comparison was moved into a separate `assert_basic_rows_equality` helper. This separates generic row comparison from plain equality comparison and removes schema checks from the row-only helper.

- **[dependency] Q:** What import change was made in `chispa/dataframe_comparer.py`, and what new module does schema comparison depend on?
  **A:** `chispa/dataframe_comparer.py` stopped importing `assert_schema_equality_ignore_nullable` and `are_schemas_equal_ignore_nullable` from `chispa.schema_comparer`, importing only `assert_schema_equality` instead. In `chispa/schema_comparer.py`, a new dependency on `chispa.structfield_comparer.are_structfields_equal` was added so nullable-ignoring schema comparisons can compare struct fields consistently.

- **[scope] Q:** Which files were modified in this commit, and what kind of change did each file receive?
  **A:** `chispa/dataframe_comparer.py` was refactored to use shared schema/row helpers and to add `assert_basic_rows_equality`. `chispa/schema_comparer.py` gained the `ignore_nullable` parameter routing and uses `are_structfields_equal` for nullable-agnostic struct-field comparison. `chispa/row_comparer.py` removed the unused `isnan` import. `chispa/structfield_comparer.py` is a new file introducing struct-field equality logic.

- **[usage] Q:** How would a developer now invoke approximate DataFrame equality when they want nullable differences ignored?
  **A:** They would call `assert_approx_df_equality(df1, df2, precision, ignore_nullable=True)`. The function now accepts the `ignore_nullable` flag and uses it when checking schema equality before comparing rows approximately.

- **[edge_case] Q:** How are `None` struct fields handled by the new struct-field comparison helper when nullability is ignored?
  **A:** In `are_structfields_equal(sf1, sf2, ignore_nullability=True)`, if one field is `None` and the other is not, the function returns `False`. It only returns `True` when both fields are present and have matching `name` and `dataType`.

- **[risk] Q:** What backward-compatibility or regression risk is introduced by this refactor of row and schema comparison?
  **A:** Because `assert_df_equality` now always runs the transform pipeline before schema and row checks, and because exact row comparison moved to `assert_basic_rows_equality`, any code relying on the old direct flow or on the previously imported `assert_generic_df_equality` name may need to adapt to the renamed helper and the new `ignore_nullable` routing in schema checks.

### MrPowers/chispa @ commit #24 `16ba7c50e8` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** This commit updates the README to document additional DataFrame comparison features, including ignoring row order, ignoring column order, ignoring nullable schema differences, and allowing NaN equality, and it also makes a small logic cleanup in `chispa/structfield_comparer.py`.

- **[new_api] Q:** Which new comparison options are described in the README, and what does each one do?
  **A:** The README now describes `assert_df_equality(df1, df2, ignore_row_order=True)` to compare DataFrames without considering row order, `assert_df_equality(df1, df2, ignore_column_order=True)` to ignore column order, `assert_df_equality(df1, df2, ignore_nullable=True)` to ignore schema nullability differences, and `assert_df_equality(df1, df2, allow_nan_equality=True)` to treat two NaN values as equal.

- **[signature_change] Q:** What API call signature examples were added or changed in the documentation for `assert_df_equality`?
  **A:** The README now shows `assert_df_equality` being called with new boolean keyword arguments: `ignore_row_order=True`, `ignore_column_order=True`, `ignore_nullable=True`, and `allow_nan_equality=True`. The diff does not show a code signature change, only updated usage examples.

- **[behavior_change] Q:** How does the documented runtime behavior of DataFrame comparison differ after this change?
  **A:** The documented behavior now includes optional comparison modes that can ignore row order, ignore column order, ignore nullability differences, and consider NaN values equal. The README also states that row sorting is not performed by default because sorting slows the function.

- **[refactor] Q:** What small refactor was made in `chispa/structfield_comparer.py`, and why is it notable?
  **A:** In `are_structfields_equal(sf1, sf2, ignore_nullability=False)`, the `if` checks under `ignore_nullability` were changed to an `if`/`elif` chain. This keeps the null checks and field-attribute comparison in a single exclusive path and avoids evaluating later conditions after an earlier match.

- **[dependency] Q:** What import or dependency-related change is shown in the diff, and what was its purpose?
  **A:** The README import example for column comparison was changed from `from chispa.column_comparer import *` to `from chispa.column_comparer import assert_column_equality`, which narrows the import to the specific function being used. The diff also updates the README text about Poetry, but no code dependency list changes are shown.

- **[scope] Q:** Which files are touched in this commit, and what type of change does each file receive?
  **A:** The commit touches `README.md`, where the documentation is expanded and rewritten to explain more comparison options and project setup, and `chispa/structfield_comparer.py`, where the nullability comparison logic is rewritten from separate `if` statements to an `if`/`elif` chain.

- **[usage] Q:** How would a developer use the newly documented row-order-insensitive DataFrame comparison?
  **A:** A developer would call `assert_df_equality(df1, df2, ignore_row_order=True)` to compare two DataFrames while ignoring the order of their rows.

- **[edge_case] Q:** How does the README say NaN values are handled in DataFrame equality comparisons?
  **A:** By default, two NaN values are not considered equal, and the README shows `nan1 == nan2` evaluating to `False`. To make NaN values compare as equal in `assert_df_equality`, the developer must pass `allow_nan_equality=True`.

- **[risk] Q:** What backward-compatibility or regression concern is suggested by the new comparison options in the README?
  **A:** The new flags change how equality is evaluated only when they are explicitly enabled, but the README notes that row sorting is not done by default because it slows the function. That means comparisons that depend on row order still fail unless `ignore_row_order=True` is passed, and enabling the new options could mask differences such as row order, column order, nullability, or NaN handling.

### MrPowers/chispa @ commit #25 `4813e3c9a1` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in `chispa/schema_comparer.py` and the README links?
  **A:** The commit updates `are_schemas_equal_ignore_nullable` to compare schema field data types more carefully while ignoring nullability, including array element types, and it also changes README image URLs from the `master` branch to the `main` branch.

- **[new_api] Q:** What new helper function is introduced in `chispa/schema_comparer.py`, and what does it do?
  **A:** The new helper is `check_type_equal_ignore_nullable(sf1, sf2)`. It compares two `StructField` data types while ignoring nullability by checking `typeName()`, and for array types it also compares `elementType`.

- **[signature_change] Q:** Did any function signature or type annotation change in this commit, and if not, what new function signature was added?
  **A:** No existing function signature was changed. The new function added is `check_type_equal_ignore_nullable(sf1, sf2)`, which takes two `StructField` arguments and returns whether their data types match when nullability is ignored.

- **[behavior_change] Q:** How does the runtime behavior of schema comparison change after this update?
  **A:** `are_schemas_equal_ignore_nullable(s1, s2)` now delegates datatype comparison to `check_type_equal_ignore_nullable` instead of using direct `sf1.dataType != sf2.dataType` checks, so array fields are compared by `elementType` rather than only by the raw `dataType` object.

- **[refactor] Q:** What existing logic was restructured in `are_schemas_equal_ignore_nullable`, and why?
  **A:** The inline comparison `sf1.name != sf2.name or sf1.dataType != sf2.dataType` was split into separate `names_equal` and `types_equal` checks. This makes the code easier to read and allows datatype comparison to be handled by the new helper `check_type_equal_ignore_nullable`.

- **[dependency] Q:** What external module or dependency is involved in the schema comparison code, and did this commit change it?
  **A:** The code still uses `six.moves.zip_longest` to iterate over the two schemas, and this commit does not change that dependency. The only new logic added is the local helper `check_type_equal_ignore_nullable`.

- **[scope] Q:** Which files are modified in this commit, and what kind of changes does each file receive?
  **A:** Two files are modified: `chispa/schema_comparer.py` receives a logic change in `are_schemas_equal_ignore_nullable` plus the new helper `check_type_equal_ignore_nullable`, and `README.md` only has documentation link updates that replace `blob/master` with `blob/main` in several image URLs.

- **[usage] Q:** How would a developer use the newly added schema comparison behavior from this change?
  **A:** A developer would call `are_schemas_equal_ignore_nullable(s1, s2)` as before; the difference is internal. The function now correctly treats schemas as equal when corresponding fields have the same name and matching data types under nullability-insensitive comparison, including array element types.

- **[edge_case] Q:** What edge case does the new helper explicitly handle, especially for complex types?
  **A:** The helper explicitly handles array types by checking `dt1.typeName() == 'array'` and comparing `dt1.elementType == dt2.elementType`. For non-array types, it returns `True` when the `typeName()` values match.

- **[risk] Q:** What compatibility or regression risk is introduced by this schema comparison change?
  **A:** Because array fields are now compared by `elementType` rather than just the top-level `dataType`, schemas that previously passed or failed based on the older broad comparison may now behave differently. This could change equality results for tests that depend on the prior array-type handling.

### MrPowers/chispa @ commit #27 `500793efe1` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the README update in this commit regarding DataFrame equality checks?
  **A:** The README now documents that `assert_df_equality(..., ignore_nullable=True)` also ignores the `nullable` setting inside `ArrayType()` elements, not just the top-level column schema nullable flag.

- **[new_api] Q:** What new documented example function is introduced in the README, and what does it demonstrate?
  **A:** The README adds an example function named `ignore_nullable_property_array()` that creates two DataFrames with identical data but different `ArrayType(DoubleType(), ...)` element nullability, demonstrating the equality-check issue that `ignore_nullable=True` bypasses.

- **[signature_change] Q:** Did this commit change any function or method signature in the repository, and what API usage does the new README text refer to?
  **A:** No code signature is changed in the diff; the README refers to the existing `assert_df_equality(df1, df2, ignore_nullable=True)` call and documents that the `ignore_nullable` flag now also applies to array element nullability.

- **[behavior_change] Q:** According to the new README example, how does the equality-check behavior differ when array element nullability differs between two DataFrames?
  **A:** The documented behavior is that two DataFrames with the same values but different `ArrayType` element nullability can fail equality checking unless `ignore_nullable=True` is used, because that flag bypasses the nullability mismatch.

- **[refactor] Q:** What part of the README explanation was restructured or expanded, and why?
  **A:** The nullable-equality section was expanded with a new paragraph and code example to clarify that nullability checks include both column schema fields and `ArrayType()` element nullability, making the existing explanation more specific.

- **[dependency] Q:** Were there any import or dependency changes in this commit, and what external Spark types does the new README example mention?
  **A:** No imports or dependencies were changed in the diff. The new documentation example mentions Spark types such as `StructType`, `StructField`, `StringType`, `ArrayType`, and `DoubleType`.

- **[scope] Q:** Which file is modified in this commit, and what kind of change does it receive?
  **A:** Only `README.md` is modified, and it receives documentation changes that add text and a Python example about how `ignore_nullable=True` interacts with `ArrayType` element nullability.

- **[usage] Q:** How would a developer use the behavior described by the new README section to compare two DataFrames with differing array element nullability?
  **A:** They would call `assert_df_equality(df1, df2, ignore_nullable=True)` when comparing DataFrames whose `coords` column uses `ArrayType(DoubleType(), True)` in one DataFrame and `ArrayType(DoubleType(), False)` in the other.

- **[edge_case] Q:** What edge case does the new README content call out about nullability handling?
  **A:** It calls out the edge case where the column itself is nullable in both DataFrames, but the `ArrayType()` element nullability differs between `True` and `False`; that inner nullability is also ignored when `ignore_nullable=True`.

- **[risk] Q:** What compatibility or regression risk is implied by the new documentation around `ignore_nullable=True`?
  **A:** The documentation implies that `ignore_nullable=True` may mask differences not only in column nullability but also in array element nullability, so comparisons that previously failed on those schema differences may now pass and hide unintended schema changes.

### MrPowers/chispa @ commit #28 `19eb1410a7` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the change in this commit?
  **A:** The commit adds documentation to the README clarifying which PySpark and Python versions chispa supports, including guidance for older Python users and a note about future PySpark 2 support.

- **[new_api] Q:** What new API or callable is introduced by this change, and what does it do?
  **A:** No new function, method, or class is introduced in this commit; the only change is a new README section titled "Supported PySpark / Python versions".

- **[signature_change] Q:** Which function or method signature changes in this commit, and how does it change?
  **A:** No function, method, or type signature changes in this commit because the diff only updates README.md documentation.

- **[behavior_change] Q:** How does the runtime behavior of chispa differ after this change?
  **A:** The runtime behavior does not change; this commit only adds version-support information to the README and does not modify code paths.

- **[refactor] Q:** What existing construct was restructured or simplified by this change?
  **A:** No code construct was restructured or simplified; the change simply inserts a new documentation section in README.md after the schema comparison image section.

- **[dependency] Q:** What import or dependency change was made, and why?
  **A:** No imports or dependencies were changed. The README text only states that chispa currently supports PySpark 2.4+ and Python 3.5+.

- **[scope] Q:** Which files are touched by this commit, and what change does each file receive?
  **A:** Only README.md is touched, and it receives a new "Supported PySpark / Python versions" section that documents supported versions and upgrade guidance.

- **[usage] Q:** How should a developer use the information added by this change when choosing a chispa version?
  **A:** A developer should read the new README section to confirm compatibility: chispa supports PySpark 2.4+ and Python 3.5+, and the README says to use chispa v0.8.2 for older Python versions.

- **[edge_case] Q:** What compatibility edge case does the new README section call out for older environments?
  **A:** The README explicitly calls out older Python versions, saying to use chispa v0.8.2 if the project is on an older Python release, and it also notes that PySpark 2 support will be dropped when chispa 1.x is released.

- **[risk] Q:** What backward-compatibility concern is mentioned in the updated documentation?
  **A:** The README warns that PySpark 2 support will be dropped when chispa 1.x is released, which signals a potential compatibility break for users relying on PySpark 2.

### MrPowers/chispa @ commit #29 `8e9673406b` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the schema comparison code?
  **A:** It extends nullable-ignoring equality checks so that nested `StructType` fields are compared recursively, not just primitive types and array element types, in `chispa/schema_comparer.py` and `chispa/structfield_comparer.py`.

- **[new_api] Q:** What new recursive comparison behavior was added to `are_structfields_equal` in `chispa/structfield_comparer.py`?
  **A:** `are_structfields_equal(sf1, sf2, ignore_nullability=False)` now detects when both `sf1.dataType` and `sf2.dataType` are `StructType` instances and recursively compares their nested fields using `six.moves.zip_longest` and `are_structfields_equal(..., ignore_nullability)`.

- **[signature_change] Q:** Did the public function signatures change in this diff, and if not, what internal type handling was added?
  **A:** The function signatures did not change. `are_structfields_equal(sf1, sf2, ignore_nullability=False)` kept the same parameters, but it now imports `StructType` and `six` and adds special handling when `dataType` is a `StructType`.

- **[behavior_change] Q:** How does schema comparison behave differently after this change when nested structs are compared with nullability ignored?
  **A:** Before this change, nested `StructType` values were compared only by direct `dataType` equality inside `are_structfields_equal`, so nested differences could be missed or handled too coarsely. After the change, nested struct fields are compared field-by-field recursively, while still ignoring nullability when requested.

- **[refactor] Q:** Why was the equality check in `are_structfields_equal` restructured instead of keeping a single combined condition?
  **A:** The combined `sf1.name != sf2.name or sf1.dataType != sf2.dataType` check was split so the function can first reject mismatched names, then branch into a recursive comparison for nested `StructType` values, and otherwise fall back to direct `dataType` equality.

- **[dependency] Q:** Which new imports were added to support the nested struct comparison, and what are they used for?
  **A:** `chispa/structfield_comparer.py` now imports `six` and `StructType` from `pyspark.sql.types`. `StructType` is used to detect nested struct fields, and `six.moves.zip_longest` is used to iterate over pairs of nested fields during recursive comparison.

- **[scope] Q:** Which files were modified in this commit, and what kind of change did each file receive?
  **A:** Two files changed: `chispa/schema_comparer.py` was updated so `check_type_equal_ignore_nullable` also handles `struct` data types, and `chispa/structfield_comparer.py` was updated to recursively compare nested `StructType` fields while ignoring nullability.

- **[usage] Q:** How would a developer benefit from the updated comparison helpers when checking two Spark schemas?
  **A:** A developer can continue calling `are_structfields_equal(sf1, sf2, ignore_nullability=True)` or using `check_type_equal_ignore_nullable(sf1, sf2)`, and nested struct fields will now be compared recursively instead of only comparing the top-level `dataType` values.

- **[edge_case] Q:** What happens if two nested struct fields have different lengths or missing fields during the new recursive comparison?
  **A:** The code uses `six.moves.zip_longest(sf1.dataType, sf2.dataType)` to pair nested fields, so if one struct has extra fields or a missing field, one side of a pair will be `None`; the recursive `are_structfields_equal` call will then return `False` because it explicitly checks for `None` mismatches.

- **[risk] Q:** What compatibility risk does this change introduce for existing schema comparisons?
  **A:** Because `check_type_equal_ignore_nullable` now delegates `struct` comparison to recursive field-by-field equality, comparisons that previously passed based on shallow `dataType` equality may now fail when nested struct field names, ordering, or nested field types differ.

### MrPowers/chispa @ commit #30 `5be6f9ce15` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this change in `chispa`?
  **A:** The change adds an `ignore_schema` option to `assert_df_equality` in `chispa/dataframe_comparer.py`, allowing DataFrame equality checks to skip schema comparison while keeping the existing row-comparison behavior.

- **[new_api] Q:** What new capability does the updated `assert_df_equality` function provide, and how is it controlled?
  **A:** `assert_df_equality(df1, df2, ..., ignore_schema=False)` now supports skipping schema validation when `ignore_schema` is set to `True`. When enabled, the function does not call `assert_schema_equality(df1.schema, df2.schema, ignore_nullable)` before comparing rows.

- **[signature_change] Q:** How did the signature of `assert_df_equality` change in `chispa/dataframe_comparer.py`?
  **A:** The function signature was extended from `assert_df_equality(df1, df2, ignore_nullable=False, transforms=None, allow_nan_equality=False, ignore_column_order=False, ignore_row_order=False)` to include a new trailing parameter: `ignore_schema=False`.

- **[behavior_change] Q:** How does `assert_df_equality` behave differently at runtime after this commit when `ignore_schema` is enabled?
  **A:** If `ignore_schema` is `True`, `assert_df_equality` no longer performs `assert_schema_equality` on `df1.schema` and `df2.schema`; it proceeds directly to the row-equality checks after applying any transforms and ordering options.

- **[refactor] Q:** What existing comparison step was restructured in `assert_df_equality`, and why is that meaningful?
  **A:** The unconditional schema comparison was restructured into a conditional block. Instead of always calling `assert_schema_equality(df1.schema, df2.schema, ignore_nullable)`, the function now wraps it in `if not ignore_schema:` so schema checking can be optionally bypassed.

- **[dependency] Q:** What import change was made in `chispa/structfield_comparer.py`, and what dependency does it affect?
  **A:** The file changed from `import six` to `import chispa.six as six`, switching the module to use the vendored `chispa.six` dependency instead of importing `six` directly.

- **[scope] Q:** Which files were modified in this commit, and what changed in each one?
  **A:** `chispa/dataframe_comparer.py` was updated to add the `ignore_schema` parameter and conditionally skip schema equality assertions. `chispa/structfield_comparer.py` was updated to import `six` from `chispa.six` instead of the top-level `six` module.

- **[usage] Q:** How would a developer use the updated DataFrame comparison API to ignore schema differences?
  **A:** A developer would call `assert_df_equality` with `ignore_schema=True`, for example `assert_df_equality(df1, df2, ignore_schema=True)`, which skips the schema comparison and compares the DataFrames using the remaining comparison logic.

- **[edge_case] Q:** What happens if two DataFrames have different schemas but the caller wants to compare only their row content?
  **A:** With `ignore_schema=True`, `assert_df_equality` will not fail on schema differences, because the `assert_schema_equality` call is skipped. The function will still apply any transforms and then run the generic row-equality check.

- **[risk] Q:** What compatibility or regression concern could this change introduce for existing callers?
  **A:** The new `ignore_schema` parameter changes the `assert_df_equality` call signature, so code that passes positional arguments could be affected if it relies on the previous parameter ordering. Also, callers that expect schema mismatches to always raise will now need to avoid setting `ignore_schema=True`.

### MrPowers/chispa @ commit #31 `88cd6f33be` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** It extends DataFrame equality checks to support approximate float comparison with a `precision` argument, while also threading `allow_nan_equality` through the approximate comparison path.

- **[new_api] Q:** What new helper function is introduced in `chispa/number_helpers.py`, and what does it do?
  **A:** The new function is `nan_safe_approx_equality(x, y, precision)`, and it returns true when `abs(x - y) <= precision` or when both values are NaN.

- **[signature_change] Q:** How did the signature of `assert_approx_df_equality` change in `chispa/dataframe_comparer.py`?
  **A:** `assert_approx_df_equality` changed from `assert_approx_df_equality(df1, df2, precision, ignore_nullable=False)` to `assert_approx_df_equality(df1, df2, precision, ignore_nullable=False, transforms=None, allow_nan_equality=False, ignore_column_order=False, ignore_row_order=False, ignore_schema=False)`, and it now forwards those options to `assert_df_equality`.

- **[behavior_change] Q:** How does `assert_df_equality` behave differently at runtime after this change when a nonzero precision is provided?
  **A:** When `precision != 0`, `assert_df_equality` now compares rows using `are_rows_approx_equal` with both `precision` and `allow_nan_equality`, instead of only using exact row equality or the NaN-aware equality path.

- **[refactor] Q:** Why was `assert_approx_df_equality` restructured in `chispa/dataframe_comparer.py`?
  **A:** It was simplified into a wrapper around `assert_df_equality` so that approximate equality shares the same preprocessing and options handling as the main DataFrame equality function.

- **[dependency] Q:** What new import was added to `chispa/row_comparer.py`, and what is it used for?
  **A:** `nan_safe_approx_equality` was imported from `chispa.number_helpers` and is used to make approximate float comparison NaN-safe when `allow_nan_equality` is enabled. The file also adds `import math` for checking `math.isnan(abs(d1[key] - d2[key]))`.

- **[scope] Q:** Which files are changed in this commit, and what kind of change does each file receive?
  **A:** `README.md` updates the Poetry install command from `poetry add chispa --dev` to `poetry add chispa -G dev`; `chispa/dataframe_comparer.py` adds a `precision` parameter and routes approximate equality through `assert_df_equality`; `chispa/number_helpers.py` adds `nan_safe_approx_equality`; and `chispa/row_comparer.py` updates approximate row comparison to handle `allow_nan_equality`.

- **[usage] Q:** How would a developer use the updated approximate DataFrame equality API after this change?
  **A:** A developer would call `assert_approx_df_equality(df1, df2, precision, ...)`, and can now also pass options such as `ignore_nullable`, `transforms`, `allow_nan_equality`, `ignore_column_order`, `ignore_row_order`, and `ignore_schema`, which are forwarded to `assert_df_equality`.

- **[edge_case] Q:** How does the new approximate row comparison handle NaN values when `allow_nan_equality` is true?
  **A:** In `are_rows_approx_equal(r1, r2, precision, allow_nan_equality=False)`, if `allow_nan_equality` is true, the code uses `nan_safe_approx_equality` so that two NaN float values can compare as equal even under approximate comparison.

- **[risk] Q:** What backward-compatibility risk is introduced by the new approximate equality path in `assert_df_equality`?
  **A:** Because `assert_df_equality` now treats any nonzero `precision` as a trigger for approximate comparison, callers that start passing a precision value will get approximate row semantics instead of exact equality, which can change which DataFrames pass or fail.

### MrPowers/chispa @ commit #32 `1e646129be` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the dataframe comparison helpers?
  **A:** It separates approximate equality handling from the general `assert_df_equality` path by removing the `precision` parameter from `chispa/dataframe_comparer.py`'s `assert_df_equality` and keeping precision-based row comparison logic inside `assert_approx_df_equality`.

- **[new_api] Q:** What does the updated `assert_approx_df_equality` function in `chispa/dataframe_comparer.py` do, and how is it used for approximate comparisons?
  **A:** `assert_approx_df_equality(df1, df2, precision, ignore_nullable=False, transforms=None, allow_nan_equality=False, ignore_column_order=False, ignore_row_order=False, ignore_schema=False)` now performs the full comparison itself: it applies any transforms, optionally sorts/selects columns, checks schema unless `ignore_schema` is set, and then compares rows with `are_rows_approx_equal` when `precision != 0`.

- **[signature_change] Q:** Which function signature changed in `chispa/dataframe_comparer.py`, and what parameter was removed?
  **A:** `assert_df_equality` changed from `assert_df_equality(df1, df2, ignore_nullable=False, transforms=None, allow_nan_equality=False, ignore_column_order=False, ignore_row_order=False, ignore_schema=False, precision=0)` to the same signature without the trailing `precision=0` parameter.

- **[behavior_change] Q:** How does the runtime behavior of `assert_df_equality` differ after this commit when precision-based comparison is desired?
  **A:** `assert_df_equality` no longer performs approximate row comparison at all; it now only checks exact row equality with `assert_basic_rows_equality` unless `allow_nan_equality` is true, in which case it uses `assert_generic_rows_equality` with `are_rows_equal_enhanced`.

- **[refactor] Q:** Why was the comparison logic in `chispa/dataframe_comparer.py` restructured between `assert_df_equality` and `assert_approx_df_equality`?
  **A:** The approximate-comparison branch was duplicated out of `assert_df_equality` so that `assert_approx_df_equality` explicitly owns the precision-based workflow, while `assert_df_equality` remains a simpler exact-equality helper.

- **[dependency] Q:** What import or dependency usage changed in the row comparison code, and what was removed?
  **A:** In `chispa/row_comparer.py`, only explanatory commented-out debug lines were removed from `are_rows_approx_equal`; there was no import or external dependency change in the diff.

- **[scope] Q:** Which files were modified in this commit, and what kind of changes were made in each one?
  **A:** `README.md` was updated with new badges, `chispa/dataframe_comparer.py` was refactored to remove `precision` from `assert_df_equality` and inline the approximate-equality logic into `assert_approx_df_equality`, and `chispa/row_comparer.py` had commented-out debug lines removed from `are_rows_approx_equal`.

- **[usage] Q:** How should a developer call the comparison helpers after this change if they want approximate equality with a precision value?
  **A:** They should call `assert_approx_df_equality(df1, df2, precision, ...)` for precision-based comparisons; `assert_df_equality` no longer accepts a `precision` argument.

- **[edge_case] Q:** What happens when `precision` is set to `0` in `assert_approx_df_equality` after this change?
  **A:** If `precision == 0`, `assert_approx_df_equality` skips `are_rows_approx_equal` and falls back to `assert_basic_rows_equality` unless `allow_nan_equality` is true, in which case it uses `assert_generic_rows_equality` with `are_rows_equal_enhanced`.

- **[risk] Q:** What backward-compatibility concern does this commit introduce for callers of `assert_df_equality`?
  **A:** Callers that previously passed `precision=` to `assert_df_equality` will now break, because that parameter was removed from the function signature and approximate comparison is now only available through `assert_approx_df_equality`.

### MrPowers/chispa @ commit #33 `63e04316dd` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa package?
  **A:** It extracts row-comparison logic into a new `chispa/rows_comparer.py` module and updates dataframe equality helpers to compare collected row lists instead of Spark DataFrames directly.

- **[new_api] Q:** What new public helper is introduced in `chispa/rows_comparer.py`, and what does it do?
  **A:** The new helper `assert_basic_rows_equality(rows1, rows2)` compares two row sequences for exact equality and, if they differ, raises `chispa.DataFramesNotEqualError` with a `PrettyTable` diff of the two sequences.

- **[signature_change] Q:** How did the argument expectations for `assert_basic_rows_equality` and `assert_generic_rows_equality` change in this commit?
  **A:** Both helpers now take row collections directly (`rows1`, `rows2`) instead of DataFrame objects. `assert_generic_rows_equality` now receives the already-collected rows plus the row comparison function and its argument list.

- **[behavior_change] Q:** How does `assert_df_equality` behave differently at runtime after this change?
  **A:** It now calls `df1.collect()` and `df2.collect()` before row comparison, so the row equality helpers operate on Python lists of rows rather than on DataFrame objects.

- **[refactor] Q:** Why was the row equality logic moved out of `dataframe_comparer.py` into a separate module?
  **A:** The change separates row-level comparison code into `chispa/rows_comparer.py`, leaving `chispa/dataframe_comparer.py` focused on DataFrame-level checks such as schema comparison and collecting rows before delegating to the row helpers.

- **[dependency] Q:** Which import was added to `chispa/dataframe_comparer.py`, and what role does it serve?
  **A:** `from chispa.rows_comparer import assert_basic_rows_equality, assert_generic_rows_equality` was added so the DataFrame comparison functions can reuse the new row-based helpers from the separate module.

- **[scope] Q:** Which files are touched by this commit, and what changes does each file receive?
  **A:** `chispa/__init__.py` re-exports `assert_basic_rows_equality`; `chispa/dataframe_comparer.py` imports row helpers from `rows_comparer.py` and calls them with `df.collect()` results; `chispa/rows_comparer.py` is a new file containing `assert_basic_rows_equality` and `assert_generic_rows_equality`.

- **[usage] Q:** How should a developer use the newly exported row-equality helper after this change?
  **A:** A developer can import `assert_basic_rows_equality` from `chispa` and pass two row sequences, for example lists returned by `df.collect()`, to compare them directly.

- **[edge_case] Q:** How does the new row-comparison code handle mismatched lengths or missing rows?
  **A:** `assert_generic_rows_equality` uses `zip_longest` and explicitly treats any pair where one side is `None` and the other is not as unequal, adding the pair to the diff table and marking the rows unequal.

- **[risk] Q:** What compatibility risk does this commit introduce for callers of the internal row-comparison helpers?
  **A:** Any existing code that called `assert_basic_rows_equality` or `assert_generic_rows_equality` with DataFrames instead of row collections will now break, because those helpers expect pre-collected rows after the refactor.

### MrPowers/chispa @ commit #34 `c12a501f99` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this change in `chispa/dataframe_comparer.py`?
  **A:** It removes the `ignore_schema` option from the DataFrame equality helpers so `assert_df_equality` and `assert_approx_df_equality` always validate schema equality via `assert_schema_equality(df1.schema, df2.schema, ignore_nullable)` before comparing rows.

- **[new_api] Q:** What new API was introduced by this change, and what does it do?
  **A:** No new function, method, or class was introduced in this diff; the change only removes the `ignore_schema` parameter from `assert_df_equality` and `assert_approx_df_equality` and keeps the existing row-comparison behavior.

- **[signature_change] Q:** How did the signatures of the DataFrame comparison functions change?
  **A:** Both `assert_df_equality` and `assert_approx_df_equality` in `chispa/dataframe_comparer.py` dropped the `ignore_schema=False` parameter from their signatures; the remaining parameters are unchanged, including `ignore_nullable`, `transforms`, `allow_nan_equality`, `ignore_column_order`, and `ignore_row_order` (plus `precision` for the approximate variant).

- **[behavior_change] Q:** How does runtime behavior differ after this change when comparing two DataFrames with different schemas?
  **A:** After this change, both `assert_df_equality` and `assert_approx_df_equality` always call `assert_schema_equality` on the two schemas, so schema differences can no longer be skipped by passing `ignore_schema=True`; schema mismatches will be checked every time before row comparison.

- **[refactor] Q:** Why was the `if not ignore_schema:` block removed from the comparison helpers?
  **A:** The code was simplified so schema validation is unconditional, which eliminates the conditional branch around `assert_schema_equality` and removes the now-unused `ignore_schema` control path from both helpers.

- **[dependency] Q:** Which imports were removed from `chispa/dataframe_comparer.py`, and what does that suggest about the module after the change?
  **A:** The file removed imports of `PrettyTable` from `chispa.prettytable`, `*` from `chispa.bcolors`, and `chispa.six as six`; this suggests those dependencies were no longer needed in this module after the schema-comparison-only change.

- **[scope] Q:** Which parts of the codebase are touched by this commit, and what changes were made in each part?
  **A:** Only `chispa/dataframe_comparer.py` is modified. In that file, two comparison functions have their `ignore_schema` parameter removed, schema validation is made unconditional, and three unused imports (`PrettyTable`, `bcolors`, and `six`) are deleted.

- **[usage] Q:** How should a developer call the updated DataFrame comparison helpers after this change?
  **A:** A developer should call `assert_df_equality(df1, df2, ...)` or `assert_approx_df_equality(df1, df2, precision, ...)` without any `ignore_schema` argument, since that parameter no longer exists; schema checking now happens automatically inside both functions.

- **[edge_case] Q:** How are schema checks handled if `ignore_nullable` is set or if a caller previously relied on `ignore_schema`?
  **A:** `ignore_nullable` is still passed into `assert_schema_equality(df1.schema, df2.schema, ignore_nullable)`, so nullable differences can still be controlled there, but there is no longer any way to bypass schema checking with `ignore_schema` because that parameter was removed.

- **[risk] Q:** What compatibility risk does this commit introduce for existing callers?
  **A:** Any existing code that calls `assert_df_equality` or `assert_approx_df_equality` with `ignore_schema=...` will break because the parameter was removed, and callers that depended on skipping schema mismatches will now get schema validation every time.

### MrPowers/chispa @ commit #35 `4d7617d6b3` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in chispa's schema comparison code?
  **A:** It consolidates nullable-ignoring comparison logic into `chispa/schema_comparer.py`, deletes `chispa/structfield_comparer.py`, and makes schema/data type equality checks recurse through nested `struct` and `array` types while ignoring nullability.

- **[new_api] Q:** What new helper functions are introduced in `chispa/schema_comparer.py`, and what do they do?
  **A:** The commit adds `are_structfields_equal_ignore_nullable(sf1, sf2)`, which compares two `StructField` values while allowing nullability differences, and `are_datatypes_equal_ignore_nullable(dt1, dt2)`, which compares data types recursively for nested `struct` and `array` types while ignoring nullability.

- **[signature_change] Q:** How did the comparison helper signature change in this commit?
  **A:** The old `check_type_equal_ignore_nullable(sf1, sf2)` helper was removed, and its responsibility was replaced by `are_datatypes_equal_ignore_nullable(dt1, dt2)`, which takes data type objects directly instead of `StructField` objects.

- **[behavior_change] Q:** How does the runtime comparison behavior differ after this change when schemas contain nested arrays or structs?
  **A:** Nested `array` element types are now compared recursively with `are_datatypes_equal_ignore_nullable`, and nested `struct` types are compared via `are_schemas_equal_ignore_nullable`, so nullability differences inside nested types are ignored more thoroughly than before.

- **[refactor] Q:** Why was `are_schemas_equal_ignore_nullable` simplified in this commit?
  **A:** It was restructured to delegate each field comparison to the new `are_structfields_equal_ignore_nullable(sf1, sf2)` helper instead of inlining separate name and type checks, making the logic reusable and easier to follow.

- **[dependency] Q:** What import dependency was removed from `chispa/schema_comparer.py`, and why?
  **A:** The import `from chispa.structfield_comparer import are_structfields_equal` was removed because the struct-field comparison logic was moved into `chispa/schema_comparer.py` itself, eliminating the separate module dependency.

- **[scope] Q:** Which files are affected by this commit, and what happens in each file?
  **A:** `chispa/schema_comparer.py` is updated to add recursive nullable-ignoring comparison helpers and to use them in schema equality checks and pretty-printing; `chispa/structfield_comparer.py` is deleted entirely.

- **[usage] Q:** How would a developer use the updated code to compare two schemas while ignoring nullability?
  **A:** They would call `assert_schema_equality_ignore_nullable(s1, s2)` or `are_schemas_equal_ignore_nullable(s1, s2)` from `chispa.schema_comparer`, and the comparison will recurse through nested `struct` and `array` types while ignoring nullable differences.

- **[edge_case] Q:** How does the new comparison helper handle `None` values for struct fields?
  **A:** `are_structfields_equal_ignore_nullable(sf1, sf2)` explicitly returns `True` when both inputs are `None` and returns `False` when only one side is `None`.

- **[risk] Q:** What backward-compatibility risk is introduced by this commit?
  **A:** Code that imported `are_structfields_equal` from `chispa.structfield_comparer` will break because `chispa/structfield_comparer.py` is deleted and the comparison logic is now only available through `chispa/schema_comparer.py`.

### MrPowers/chispa @ commit #36 `04af7733e2` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of adding `chispa/structfield_comparer.py` in this commit?
  **A:** The commit adds a dedicated `are_structfields_equal` helper for comparing Spark `StructField` objects, with optional nullability-insensitive comparison via `ignore_nullability`.

- **[new_api] Q:** What new API is introduced in `chispa/structfield_comparer.py`, and what does it do?
  **A:** The new function `are_structfields_equal(sf1, sf2, ignore_nullability=False)` is introduced. It compares two StructField values and, when `ignore_nullability` is `True`, delegates to `are_structfields_equal_ignore_nullable(sf1, sf2)`; otherwise it compares them with `sf1 == sf2`.

- **[signature_change] Q:** What is the signature of the new comparison function, including its parameter default?
  **A:** The function signature is `are_structfields_equal(sf1, sf2, ignore_nullability=False)`, where `ignore_nullability` defaults to `False`.

- **[behavior_change] Q:** How does runtime behavior change when `ignore_nullability` is enabled versus disabled?
  **A:** When `ignore_nullability` is `False`, the function returns the direct equality result of `sf1 == sf2`. When `ignore_nullability` is `True`, it instead compares the StructFields using `are_structfields_equal_ignore_nullable(sf1, sf2)`, which ignores nullability differences.

- **[refactor] Q:** Why was the comparison logic split into a new module instead of being left inline elsewhere?
  **A:** The comparison logic was extracted into `chispa/structfield_comparer.py` so StructField equality can be handled by a focused helper function rather than inline code, with the nullability-specific branch isolated behind a single wrapper.

- **[dependency] Q:** What existing import does the new file add, and why is it needed?
  **A:** The new file imports `are_structfields_equal_ignore_nullable` from `chispa.schema_comparer`. That import is needed so `are_structfields_equal` can delegate nullability-insensitive comparisons to the existing schema comparison helper.

- **[scope] Q:** Which file is changed by this commit, and what change does it contain?
  **A:** Only one file is added: `chispa/structfield_comparer.py`. It contains a new import from `chispa.schema_comparer` and a new `are_structfields_equal` function with an `ignore_nullability` branch.

- **[usage] Q:** How would a developer use the new StructField comparison helper in code?
  **A:** A developer would call `are_structfields_equal(sf1, sf2)` for a normal StructField equality check, or `are_structfields_equal(sf1, sf2, ignore_nullability=True)` to compare two StructFields while ignoring nullability differences.

- **[edge_case] Q:** What happens if `ignore_nullability` is not provided or is set to `False` when comparing two StructFields?
  **A:** The function uses its default `ignore_nullability=False` and falls back to direct equality with `sf1 == sf2`, so nullability is not ignored in that case.

- **[risk] Q:** What backward-compatibility or regression risk could this new helper introduce?
  **A:** Because the new helper changes the comparison path based on `ignore_nullability`, any callers expecting direct `sf1 == sf2` semantics must pass `ignore_nullability=False`; otherwise they may get different results when nullability differs.

### MrPowers/chispa @ commit #37 `b92cf0805c` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in `chispa`'s schema/structfield comparison code?
  **A:** The commit consolidates struct field comparison into a single `are_structfields_equal` function in `chispa/schema_comparer.py`, and updates the schema comparison helpers to call it with `ignore_nullability=True` where nullable differences should be ignored.

- **[new_api] Q:** What new function is introduced in `chispa/schema_comparer.py`, and what does it do?
  **A:** `are_structfields_equal(sf1, sf2, ignore_nullability=False)` is introduced. It compares two Spark `StructField` values, either by strict equality when `ignore_nullability` is `False`, or by comparing field names and recursively comparing data types while ignoring nullability when `True`.

- **[signature_change] Q:** How did the struct field comparison API change across the touched modules?
  **A:** `chispa/structfield_comparer.py` no longer defines its own `are_structfields_equal(sf1, sf2, ignore_nullability=False)` implementation; it now imports `are_structfields_equal` from `chispa.schema_comparer`. In `chispa/schema_comparer.py`, the new function adds the `ignore_nullability=False` parameter directly to the shared implementation.

- **[behavior_change] Q:** How does runtime behavior differ when comparing struct fields after this change?
  **A:** Struct field comparison now uses one shared implementation. With `ignore_nullability=False`, it returns `sf1 == sf2` instead of separately handling name and datatype checks; with `ignore_nullability=True`, it still ignores nullability by checking for `None`, comparing names, and then delegating to `are_datatypes_equal_ignore_nullable`.

- **[refactor] Q:** Why was the struct field comparison logic restructured in this commit?
  **A:** The logic was centralized so that schema comparison and struct field comparison both use the same `are_structfields_equal` function from `chispa/schema_comparer.py`, removing the duplicate wrapper implementation that previously lived in `chispa/structfield_comparer.py`.

- **[dependency] Q:** What import change was made in `chispa/structfield_comparer.py`, and why?
  **A:** The file now imports `are_structfields_equal` from `chispa.schema_comparer` and removes the local function body, so `structfield_comparer.py` acts as a thin re-export instead of maintaining a separate implementation.

- **[scope] Q:** Which files are modified in this commit, and what does each file change?
  **A:** `chispa/schema_comparer.py` is updated to call the unified `are_structfields_equal` helper from schema comparison functions and to define the new shared helper. `chispa/structfield_comparer.py` is simplified to only import `are_structfields_equal` from `chispa.schema_comparer`.

- **[usage] Q:** How should a developer use the updated struct field comparison API after this change?
  **A:** A developer can call `are_structfields_equal(sf1, sf2)` for strict equality, or `are_structfields_equal(sf1, sf2, True)` to compare two `StructField` values while ignoring nullability. The schema comparison helpers in `schema_comparer.py` now use the `True` form internally.

- **[edge_case] Q:** How are `None` struct fields handled when nullability is ignored?
  **A:** Inside `are_structfields_equal(sf1, sf2, ignore_nullability=True)`, if either field is `None`, the function returns `True` only when both are `None`; if exactly one is `None`, it returns `False`.

- **[risk] Q:** What compatibility or regression risk does this refactor introduce for callers of struct field comparison?
  **A:** Callers that imported `are_structfields_equal` from `chispa.structfield_comparer` now depend on that module re-exporting the implementation from `chispa.schema_comparer`. Also, the strict comparison path changed from explicit name/dataType checks to `sf1 == sf2`, which could alter behavior if Spark's `StructField.__eq__` differs from the previous manual comparison.

### MrPowers/chispa @ commit #38 `0d341218cd` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of commit 0d341218cd0a7be84af5408b84f40edaa7799ec1 in the chispa repository?
  **A:** The commit adds support for underlining individual differing cells in DataFrame equality failure output, exposed through a new `underline_cells` option on `assert_df_equality`, and documents that behavior in the README.

- **[new_api] Q:** What new helper function was introduced in `chispa/bcolors.py`, and what does it return?
  **A:** The new function `underline_text(input_text: str) -> str` was added in `chispa/bcolors.py`; it wraps the input string with ANSI codes to produce white, underlined text and then resets formatting.

- **[signature_change] Q:** How did the signature of `assert_df_equality` in `chispa/dataframe_comparer.py` change?
  **A:** `assert_df_equality` gained a new trailing boolean parameter named `underline_cells=False`, so its full signature now includes `underline_cells` after `ignore_row_order`.

- **[behavior_change] Q:** How does the row comparison output change at runtime when `underline_cells=True` is passed to `assert_df_equality`?
  **A:** When rows differ and `underline_cells=True`, the failure table now highlights the differing fields inside each `Row(...)` string by underlining only the columns whose values differ, instead of printing the entire row unstyled.

- **[refactor] Q:** What internal restructuring was done in `chispa/rows_comparer.py` to support the new formatting behavior?
  **A:** The row comparison code was refactored to factor cell-formatting logic into a new private helper, `__underline_cells_in_row`, which both `assert_basic_rows_equality` and `assert_generic_rows_equality` can call when `underline_cells` is enabled.

- **[dependency] Q:** What import or dependency-related change was made in `chispa/prettytable.py`, and why is it relevant to this patch?
  **A:** The `pkg_resources` import and the commented version lookup were removed from `chispa/prettytable.py`, simplifying that module while the new underlined failure rendering is implemented elsewhere in `rows_comparer.py` and `bcolors.py`.

- **[scope] Q:** Which files were changed in this commit, and what kind of change did each file receive?
  **A:** `README.md` was updated with documentation and an example for `underline_cells`; `chispa/bcolors.py` added underline styling helpers; `chispa/dataframe_comparer.py` threaded the new `underline_cells` argument through equality assertions; `chispa/rows_comparer.py` added the underlined diff rendering logic; `chispa/prettytable.py` was cleaned up by removing `pkg_resources`; `docs/gen_ref_pages.py` was added for docs generation; and `docs/index.md` was added to include the README in the docs site.

- **[usage] Q:** How would a developer enable the new underlined diff display when checking DataFrame equality?
  **A:** They would call `assert_df_equality(df1, df2, underline_cells=True)`, which tells chispa to underline differing columns in the row-level failure output.

- **[edge_case] Q:** What default handling was added for the new underline feature when callers do not pass it explicitly?
  **A:** The new `underline_cells` parameter defaults to `False` in `assert_df_equality`, `assert_basic_rows_equality`, and `assert_generic_rows_equality`, so existing behavior remains unchanged unless the caller opts in.

- **[risk] Q:** What backward-compatibility or regression risk does this commit introduce in the row comparison path?
  **A:** The new underlined formatting path assumes `rows1[0].__fields__` exists when `underline_cells=True`; if row sequences are empty or non-Row inputs are passed, the new helper-based formatting path could fail differently than the previous plain row printing behavior.

### MrPowers/chispa @ commit #39 `fa17888e48` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the dataframe and schema comparison code?
  **A:** It adds support for ignoring schema metadata during equality checks, by threading a new `ignore_metadata` option through `assert_df_equality` and `assert_schema_equality`, and by updating struct-field comparison logic to optionally skip metadata differences.

- **[new_api] Q:** What new helper function is introduced in `chispa/schema_comparer.py`, and what does it do?
  **A:** The commit introduces `assert_schema_equality_full(s1, s2, ignore_nullable=False, ignore_metadata=False)`, which performs a full schema comparison that can ignore nullability and/or metadata, and raises `SchemasNotEqualError` with a `PrettyTable` diff when the schemas do not match.

- **[signature_change] Q:** Which function signatures changed to accept the new metadata-related option?
  **A:** `assert_df_equality` in `chispa/dataframe_comparer.py` now accepts `ignore_metadata=False`, and `assert_schema_equality` in `chispa/schema_comparer.py` now has the signature `assert_schema_equality(s1, s2, ignore_nullable=False, ignore_metadata=False)`. `are_structfields_equal` also changed from `are_structfields_equal(sf1, sf2, ignore_nullability=False)` to `are_structfields_equal(sf1, sf2, ignore_nullability=False, ignore_metadata=False)`.

- **[behavior_change] Q:** How does schema comparison behave differently after this change when metadata differs between two schemas?
  **A:** Before this commit, `assert_schema_equality` compared schemas without a metadata-specific option. After the change, callers can pass `ignore_metadata=True`, and `are_structfields_equal` will treat two `StructField`s as equal even if their `metadata` values differ, as long as the names and data types still match under the chosen nullability rules.

- **[refactor] Q:** What existing comparison path was restructured, and why was it split into a new helper?
  **A:** `assert_schema_equality` was restructured so that the simple exact-equality path remains in `assert_basic_schema_equality`, while the more flexible comparison logic moved into `assert_schema_equality_full`. This separation lets the code handle combinations of `ignore_nullable` and `ignore_metadata` instead of only the previous exact-vs-ignore-nullable split.

- **[dependency] Q:** What library usage appears in the diff for schema diff reporting, and what is it used for?
  **A:** The code continues to use `PrettyTable` and `six.moves.zip_longest` in `chispa/schema_comparer.py`. `PrettyTable` formats mismatch output, and `zip_longest` is used to pair schema fields when building the error table for unequal schemas.

- **[scope] Q:** Which files were modified, and what kind of change was made in each one?
  **A:** `chispa/dataframe_comparer.py` was updated to pass a new `ignore_metadata` argument from `assert_df_equality` into schema comparison. `chispa/schema_comparer.py` received the core comparison changes: new `ignore_metadata` support, the new `assert_schema_equality_full` helper, and updates to `are_structfields_equal`, `assert_schema_equality_ignore_nullable`, and related helpers.

- **[usage] Q:** How would a developer use the changed API to ignore metadata differences when comparing two DataFrames?
  **A:** A developer would call `assert_df_equality(df1, df2, ignore_metadata=True)` or call `assert_schema_equality(schema1, schema2, ignore_metadata=True)` directly. The new flag is wired through from `assert_df_equality` to `assert_schema_equality`.

- **[edge_case] Q:** How does the new struct-field comparison handle `None` values and metadata when optional flags are enabled?
  **A:** `are_structfields_equal` still handles `None` inputs explicitly: two `None` values compare equal, while only one `None` returns `False`. When both fields are present, it checks the field name first, then checks metadata unless `ignore_metadata` is `True`, and finally compares data types while optionally ignoring nullability.

- **[risk] Q:** What compatibility risk does this commit introduce for code calling the comparison helpers?
  **A:** The public comparison helpers now accept an extra `ignore_metadata` parameter, and the comparison semantics can change when that flag is used. Code that relied on strict schema equality, including metadata, may need to avoid passing the new flag; otherwise schemas that previously failed can now pass.

### MrPowers/chispa @ commit #40 `ba512da000` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in `MrPowers/chispa`?
  **A:** It adds terminal string formatting helpers and updates `chispa.rows_comparer.assert_basic_rows_equality` to produce more detailed, color-formatted row and cell mismatch output, with configurable default formats.

- **[new_api] Q:** What new API is introduced in `chispa/terminal_str_formatter.py`, and what does it do?
  **A:** The new module introduces `format_string(input, formats)`, which wraps a string in ANSI escape codes based on a list of format names, and `format_mismatched_cell(input_text: str, mismatched_cells) -> str`, which delegates to `format_string` for mismatched-cell formatting.

- **[signature_change] Q:** How did the signature of `assert_basic_rows_equality` in `chispa/rows_comparer.py` change?
  **A:** Its signature changed from `assert_basic_rows_equality(rows1, rows2, underline_cells=False)` to `assert_basic_rows_equality(rows1, rows2, underline_cells=False, formats=DefaultFormats())`, adding a `formats` parameter that defaults to a `DefaultFormats` instance.

- **[behavior_change] Q:** How does `assert_basic_rows_equality` now behave differently when the two row lists have different lengths?
  **A:** It now checks `len(rows1) != len(rows2)` up front and raises `chispa.DataFramesNotEqualError("\n" + "Number of rows is not equal")` immediately instead of relying only on the later row-by-row comparison output.

- **[refactor] Q:** What part of the row comparison logic was restructured in `chispa/rows_comparer.py`, and how is it represented now?
  **A:** The old underline-based helper path for mismatched rows was removed from `assert_basic_rows_equality`; mismatched rows are now built inline by iterating over field names, formatting each field/value pair, and assembling explicit `Row(...)` strings for the `PrettyTable` output.

- **[dependency] Q:** What new import or dependency was added to support the row comparison changes, and why?
  **A:** `chispa.rows_comparer` now imports `format_string` and `format_mismatched_cell` from `chispa.terminal_str_formatter`, plus `DefaultFormats` from `chispa.default_formats`, so row comparison can apply configurable ANSI formatting instead of using the previous direct color helpers.

- **[scope] Q:** Which files were added or modified in this commit, and what changed in each one?
  **A:** `chispa/default_formats.py` was added with a `DefaultFormats` dataclass holding default style lists; `chispa/terminal_str_formatter.py` was added with ANSI formatting helpers; and `chispa/rows_comparer.py` was modified to use those helpers, add a `formats` parameter, and change mismatch rendering.

- **[usage] Q:** How would a developer use the updated row comparison formatting from `assert_basic_rows_equality`?
  **A:** A developer would call `assert_basic_rows_equality(rows1, rows2, formats=DefaultFormats())` or pass a custom object with `matched_rows`, `mismatched_rows`, `matched_cells`, and `mismatched_cells` lists to control how equal and unequal rows/cells are rendered in the failure table.

- **[edge_case] Q:** What default formatting is applied when no custom formats are supplied, and how are matched versus mismatched cells handled?
  **A:** The default `DefaultFormats` values are `mismatched_rows = ["red", "bold"]`, `matched_rows = ["blue"]`, `mismatched_cells = ["white", "underline"]`, and `matched_cells = ["blue", "bold"]`; matched rows are formatted with `formats.matched_rows`, while each cell in a mismatched row is formatted with either `formats.mismatched_cells` or `formats.matched_cells` depending on whether the field values differ.

- **[risk] Q:** What backward-compatibility risk is introduced by the new comparison formatting logic?
  **A:** The mismatch output format changes substantially: `assert_basic_rows_equality` no longer uses the old `underline_cells`-driven helper path and now emits formatted `Row(field='value', ...)` strings, so any tests or callers that assert on the previous error message or table layout could break.

### MrPowers/chispa @ commit #41 `4bb7cc3c8c` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the row-comparison and formatting code?
  **A:** It simplifies how row mismatches are rendered by removing the separate `format_mismatched_cell` helper, trimming the default style lists in `chispa/default_formats.py`, and changing `assert_basic_rows_equality` in `chispa/rows_comparer.py` to handle unequal row lengths and display mismatched rows more directly.

- **[new_api] Q:** What new public function or class was introduced by this change, and what does it do?
  **A:** No new function or class is introduced in this commit; the only API change is the removal of `format_mismatched_cell` from `chispa/terminal_str_formatter.py`, while `format_string` remains the formatting helper.

- **[signature_change] Q:** Which function signature or type annotation changed in this diff?
  **A:** No function signature or type annotation was changed in the diff. The notable change is that `format_mismatched_cell(input_text: str, mismatched_cells) -> str` was deleted from `chispa/terminal_str_formatter.py`, but existing function signatures were left intact.

- **[behavior_change] Q:** How does the runtime behavior of `assert_basic_rows_equality` differ after this change when the two row lists have different lengths?
  **A:** `assert_basic_rows_equality` no longer raises immediately on length mismatch with the message `Number of rows is not equal`. Instead, it uses `six.moves.zip_longest(rows1, rows2)` to compare both lists and adds table rows where one side is `None`, then raises `DataFramesNotEqualError` with the generated table output.

- **[refactor] Q:** Why was the row-rendering logic in `chispa/rows_comparer.py` restructured?
  **A:** The row-rendering logic was simplified so mismatched rows are displayed without building separate `Row(...)` strings and without using the removed `format_mismatched_cell` helper. The code now directly appends the formatted field strings and joins them with `", "` for the table output.

- **[dependency] Q:** What import dependency was removed from `chispa/rows_comparer.py`, and what code path no longer needs it?
  **A:** The import of `format_mismatched_cell` from `chispa.terminal_str_formatter` was removed. It is no longer needed because `assert_basic_rows_equality` now formats mismatched row content directly with `format_string` and no longer calls the removed helper.

- **[scope] Q:** Which files are touched in this commit, and what kind of change does each file receive?
  **A:** Three files are modified: `chispa/default_formats.py` changes the default style lists for `mismatched_rows` and `matched_cells`; `chispa/rows_comparer.py` changes row-equality comparison and table rendering logic; and `chispa/terminal_str_formatter.py` removes the `format_mismatched_cell` helper.

- **[usage] Q:** How would a developer use the updated row-comparison behavior when comparing two lists of Spark `Row` objects?
  **A:** A developer would still call `assert_basic_rows_equality(rows1, rows2, underline_cells=False, formats=DefaultFormats())` from `chispa.rows_comparer`, but when rows differ the error table now shows formatted field strings directly, and mismatched row-length cases are handled through the same comparison path instead of a separate length check.

- **[edge_case] Q:** How are `None` or missing rows handled now when one input list is shorter than the other?
  **A:** The function explicitly checks for `r1 is None` and `r2 is not None`, or vice versa, after `zip_longest`. In those cases it adds a table row with `None` on the missing side and formats the present row using `formats.mismatched_rows`.

- **[risk] Q:** What backward-compatibility or regression risk is introduced by this formatting change?
  **A:** Because the default style lists were reduced (`mismatched_rows` loses `bold`, `matched_cells` loses `bold`) and the error-table text no longer wraps row values in `Row(...)`, any tests or consumers that assert on the exact visual output of comparison failures may need to be updated.

### MrPowers/chispa @ commit #42 `c5a16a0bb7` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the change in `chispa/rows_comparer.py`?
  **A:** The change makes `assert_basic_rows_equality` raise `chispa.DataFramesNotEqualError` only when the compared rows are actually unequal at the row or cell level, instead of always raising whenever `rows1 != rows2`.

- **[new_api] Q:** What new API or helper was introduced by this commit, and what does it do?
  **A:** No new public function, method, or class was introduced. The change adds an internal `all_rows_equal` flag inside `assert_basic_rows_equality` to track whether any row or cell mismatch was found while building the comparison table.

- **[signature_change] Q:** Did any function signature or type annotation change in this diff, and if so, what changed?
  **A:** No function signature or type annotation changed in this diff. The only modified function is `assert_basic_rows_equality(rows1, rows2, underline_cells=False, formats=DefaultFormats)` and its parameters remain the same.

- **[behavior_change] Q:** How does the runtime behavior of `assert_basic_rows_equality` differ after this change?
  **A:** After the change, the function still builds a `PrettyTable` for mismatched rows, but it now raises `chispa.DataFramesNotEqualError` only if `all_rows_equal` was set to `False` during comparison. If the initial `rows1 != rows2` condition is true but no row or cell mismatch is recorded, the final exception is no longer raised.

- **[refactor] Q:** Why was the comparison logic in `assert_basic_rows_equality` restructured?
  **A:** The logic was restructured to separate detecting differences from formatting the comparison output. The new `all_rows_equal` variable is updated inside each mismatch branch, and the final exception is emitted only after all rows are processed and the flag shows a real mismatch.

- **[dependency] Q:** What dependency-related change was made in `rows_comparer.py`, and why?
  **A:** No import or external dependency was added or removed. The file still uses `six.moves.zip_longest`, `PrettyTable`, and `chispa` as before; the change only adjusts how those existing dependencies are used to decide whether to raise an error.

- **[scope] Q:** Which file is affected by this commit, and what kind of change does it contain there?
  **A:** Only `chispa/rows_comparer.py` is modified. The change is limited to `assert_basic_rows_equality`, where mismatch tracking is added and the final `DataFramesNotEqualError` is made conditional on the new flag.

- **[usage] Q:** How would a developer use the changed code path to compare rows after this commit?
  **A:** A developer would still call `assert_basic_rows_equality(rows1, rows2, underline_cells=False, formats=DefaultFormats)` from `chispa/rows_comparer.py`. The difference is that the function now inspects each paired row and cell, builds a formatted table of mismatches, and raises `chispa.DataFramesNotEqualError` only when an actual mismatch is detected.

- **[edge_case] Q:** What edge case does the new `all_rows_equal` check address in `assert_basic_rows_equality`?
  **A:** It addresses the case where `rows1 != rows2` is true but the comparison loop does not encounter any mismatched rows or cells. The new `all_rows_equal` flag starts as `True` and is flipped to `False` only when a missing row or differing cell is found; the exception is raised only if that flag is false.

- **[risk] Q:** What compatibility or regression risk is introduced by this change?
  **A:** The main risk is that cases which previously triggered `chispa.DataFramesNotEqualError` solely because `rows1 != rows2` may no longer raise if no mismatch is recorded inside the loop. This changes the error-raising condition in `assert_basic_rows_equality` and could alter existing tests or callers that depended on the earlier behavior.

### MrPowers/chispa @ commit #43 `591e58f888` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the change in `chispa/terminal_str_formatter.py`?
  **A:** It expands the ANSI formatting options supported by `format_string()` by adding more named terminal color/style codes and by renaming the existing inverse/hidden style keys to shorter names.

- **[new_api] Q:** What new formatting names does `format_string(input, formats)` accept after this change, and what do they represent?
  **A:** The formatter now accepts `invert` for ANSI inverse (`\033[7m`) and `hide` for ANSI hidden (`\033[8m`), along with new color names such as `black`, `green`, `yellow`, `purple`, `cyan`, `light_gray`, `dark_gray`, `light_red`, `light_green`, `light_yellow`, `light_blue`, `light_purple`, and `light_cyan`.

- **[signature_change] Q:** Did the signature of `format_string()` change in this commit, and if so, how?
  **A:** No. The function signature remains `format_string(input, formats)`. The change is in the internal `ansi_codes` mapping, not in the function parameters or type annotations.

- **[behavior_change] Q:** How does the runtime behavior of `format_string()` differ after this update?
  **A:** At runtime, `format_string()` can now produce ANSI escape sequences for additional color names and the renamed `invert`/`hide` keys. Inputs using those new keys will now be formatted, whereas they would not have been recognized before.

- **[refactor] Q:** Why were the existing `inverted` and `hidden` entries restructured in the formatter mapping?
  **A:** They were simplified by renaming the keys to shorter forms, `invert` and `hide`, while keeping the same ANSI codes (`\033[7m` and `\033[8m`). This makes the mapping consistent with the newly added style names.

- **[dependency] Q:** Did this commit add or remove any external dependencies or imports in `chispa/terminal_str_formatter.py`?
  **A:** No external dependencies or imports were changed. The update only modifies the local `ansi_codes` dictionary inside `format_string()`.

- **[scope] Q:** Which file is modified in this commit, and what kind of change does it contain?
  **A:** Only `chispa/terminal_str_formatter.py` is modified, and the change is limited to the `ansi_codes` dictionary used by `format_string()` to map format names to ANSI escape codes.

- **[usage] Q:** How would a developer use the updated formatter to apply one of the newly added styles?
  **A:** They would call `format_string()` with the desired key in the `formats` list, for example `format_string('text', ['light_yellow'])` or `format_string('text', ['invert'])`, and the function would wrap the string with the corresponding ANSI code.

- **[edge_case] Q:** What happens if a caller passes a format name that is not present in the updated mapping?
  **A:** The diff does not change the handling of unknown formats; `format_string()` still iterates through the provided `formats` list and only applies codes for keys present in `ansi_codes`, so unsupported names remain unformatted.

- **[risk] Q:** What compatibility risk does this change introduce for existing callers of `format_string()`?
  **A:** Callers that still use the նախկին keys `inverted` and `hidden` may no longer get the intended ANSI formatting, because those keys were replaced with `invert` and `hide` in the mapping.

### MrPowers/chispa @ commit #44 `957febe988` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the row comparison code?
  **A:** It changes the formatting of row-comparison failures in `chispa/rows_comparer.py` so mismatched cells are highlighted with the configured `DefaultFormats`, and it updates the default mismatched-cell color in `chispa/default_formats.py` from `"white"` to `"red"`.

- **[new_api] Q:** What new parameter was added to `assert_generic_rows_equality`, and what does it control?
  **A:** `assert_generic_rows_equality(rows1, rows2, row_equality_fun, row_equality_fun_args, underline_cells=False, formats=DefaultFormats())` now accepts a `formats` parameter. It controls the styling used by `format_string` for mismatched rows, matched rows, mismatched cells, and matched cells in the PrettyTable output.

- **[signature_change] Q:** How did the signature of `assert_generic_rows_equality` change in `chispa/rows_comparer.py`?
  **A:** The function signature changed from `assert_generic_rows_equality(rows1, rows2, row_equality_fun, row_equality_fun_args, underline_cells=False)` to `assert_generic_rows_equality(rows1, rows2, row_equality_fun, row_equality_fun_args, underline_cells=False, formats=DefaultFormats())`, adding a new `formats` argument with a `DefaultFormats()` default.

- **[behavior_change] Q:** How does the runtime output of row comparison failures differ after this change?
  **A:** When rows differ, `assert_generic_rows_equality` now formats mismatched rows with `formats.mismatched_rows`, matched rows with `formats.matched_rows`, and individual differing fields with `formats.mismatched_cells`; matching fields are formatted with `formats.matched_cells`. The old underline-based cell highlighting helper was removed, so the failure text is now built from per-field formatted strings instead of `Row(...)` strings with underlined differences.

- **[refactor] Q:** Why was the old `__underline_cells_in_row` helper removed and the comparison loop restructured?
  **A:** The logic was simplified to build each row’s display string directly inside `assert_generic_rows_equality`, using `zip_longest` over `r1.__fields__` and `r2.__fields__` and applying `format_string` per field. This removes the separate underline-specific helper and makes formatting consistent through the `formats` configuration.

- **[dependency] Q:** What existing import or dependency is now used differently in `chispa/rows_comparer.py`, and why?
  **A:** `DefaultFormats` is now passed as the default value for the new `formats` parameter in `assert_generic_rows_equality`, so the function can use the shared formatting configuration. The code also relies on `format_string` for all row and cell styling instead of manual `bcolors` concatenation in the generic comparison path.

- **[scope] Q:** Which files are modified in this commit, and what kind of change does each file contain?
  **A:** `chispa/default_formats.py` changes the default color for `mismatched_cells` from `"white"` to `"red"`. `chispa/rows_comparer.py` changes `assert_generic_rows_equality` to accept a `formats` argument, rewrites how matched and mismatched rows are rendered, and removes the `__underline_cells_in_row` helper.

- **[usage] Q:** How would a developer use the updated generic row comparison function to customize formatting?
  **A:** A developer would call `assert_generic_rows_equality(rows1, rows2, row_equality_fun, row_equality_fun_args, formats=custom_formats)` and pass a `DefaultFormats`-like object with fields such as `mismatched_rows`, `matched_rows`, `mismatched_cells`, and `matched_cells` to control the PrettyTable styling.

- **[edge_case] Q:** How does the new implementation handle rows with different numbers of fields or values?
  **A:** It uses `six.moves.zip_longest(r1.__fields__, r2.__fields__)` to iterate over field names from both rows, and it compares each pair of fields while building the output strings. For missing rows at the top level, it formats the present row and the `None` side with `formats.mismatched_rows` before raising `chispa.DataFramesNotEqualError`.

- **[risk] Q:** What compatibility risk does this formatting change introduce for existing tests or consumers?
  **A:** Any tests or consumers that assert on the exact failure text may break, because the default mismatched-cell color changed from `"white"` to `"red"` and the generic row output format changed from the old `Row(...)`/underline style to per-field formatted strings built with `format_string`.

### MrPowers/chispa @ commit #45 `0d982532dc` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in chispa?
  **A:** It updates the row-comparison error formatting so values are rendered without surrounding quotes and adds README documentation showing how to customize the printed formats for row equality assertions.

- **[new_api] Q:** What new customization pattern does the README introduce for row comparison output, and what does it control?
  **A:** The README introduces a user-defined formats class, such as `MyFormats`, that is passed as the `formats` argument to `assert_basic_rows_equality`. It controls the styling used for `mismatched_rows`, `matched_rows`, `mismatched_cells`, and `matched_cells` in the printed assertion output.

- **[signature_change] Q:** Which function signatures or type annotations changed in the diff?
  **A:** No function signature or type annotation was changed in `rows_comparer.py`; `assert_basic_rows_equality(rows1, rows2, underline_cells=False, formats=DefaultFormats)` and `assert_generic_rows_equality(rows1, rows2, row_equality_fun, row_equality_fun_kwargs={}, formats=DefaultFormats)` keep the same parameters. The change only alters how field values are formatted inside those functions.

- **[behavior_change] Q:** How does the runtime output of the row comparison helpers differ after this commit?
  **A:** When `assert_basic_rows_equality` or `assert_generic_rows_equality` highlights matched or mismatched cells, the field values are now formatted as `field=value` instead of `field='value'`. This means the generated error messages no longer wrap displayed values in single quotes.

- **[refactor] Q:** What existing formatting logic was simplified in `rows_comparer.py`?
  **A:** The code that builds cell strings in both row comparison helpers was simplified by removing the extra single-quote characters from the formatted values. The repeated expressions changed from `f"{field}='{value}'"` to `f"{field}={value}"` for both matched and mismatched cells.

- **[dependency] Q:** What dependency or import-related change is reflected in the diff?
  **A:** The README now demonstrates using `@dataclass` and `@pytest.fixture()` to define and inject a custom formats object, so the documented usage depends on Python dataclasses and pytest fixtures for the example setup. The production code diff itself does not show an import change in `rows_comparer.py`.

- **[scope] Q:** Which files were modified, and what kind of change did each file receive?
  **A:** Two files were changed: `README.md` was updated with new documentation for custom formatting and revised NaN wording, and `chispa/rows_comparer.py` was updated in two functions to stop quoting field values in formatted cell output.

- **[usage] Q:** How would a developer use the new formatting customization shown in the README?
  **A:** A developer would define a formats class, such as `MyFormats`, set attributes like `mismatched_rows`, `matched_rows`, `mismatched_cells`, and `matched_cells`, and pass an instance to `assert_basic_rows_equality(df1.collect(), df2.collect(), formats=MyFormats())`. The README also shows defining that fixture in `conftest.py` and injecting it into a test.

- **[edge_case] Q:** What NaN-handling guidance is documented in the README after this change?
  **A:** The README states that pandas considers NaN values equal by default, but chispa requires the caller to pass `allow_nan_equality=True` to `assert_df_equality(df1, df2, allow_nan_equality=True)` if two NaN values should be treated as equal.

- **[risk] Q:** What backward-compatibility risk could result from the row formatting change?
  **A:** Any tests or tooling that assert against the exact string form of chispa's row-comparison error messages may fail because cell values are now rendered without single quotes. The semantic comparison behavior is unchanged, but the textual output format has changed.

### MrPowers/chispa @ commit #46 `bd822a4fa0` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa package?
  **A:** It adds a configurable formatting object, `DefaultFormats`, and threads it through DataFrame equality assertions so `assert_df_equality` and `assert_approx_df_equality` can use custom output formatting instead of only the built-in defaults.

- **[new_api] Q:** What new API is introduced in `chispa/__init__.py`, and what does its `assert_df_equality` method do?
  **A:** A new `Chispa` class is introduced in `chispa/__init__.py`. Its `assert_df_equality(self, df1, df2, ignore_nullable=False, transforms=None, allow_nan_equality=False, ignore_column_order=False, ignore_row_order=False, underline_cells=False, ignore_metadata=False)` method delegates to the module-level `assert_df_equality` function and passes along `self.formats`.

- **[signature_change] Q:** How did the signature of `assert_df_equality` in `chispa/dataframe_comparer.py` change?
  **A:** `assert_df_equality` gained a new trailing parameter, `formats=DefaultFormats()`, after `ignore_metadata=False`. The same file also changed `assert_approx_df_equality` to accept `formats=DefaultFormats()` as an added trailing parameter.

- **[behavior_change] Q:** How does row-comparison behavior differ after this change when assertions fail or format output is generated?
  **A:** Both `assert_df_equality` and `assert_approx_df_equality` now pass a `formats` object into `assert_generic_rows_equality` and `assert_basic_rows_equality`, so row comparison and any rendered output can use the supplied formatting configuration instead of relying solely on implicit defaults.

- **[refactor] Q:** What existing import structure in `chispa/dataframe_comparer.py` was simplified or restructured?
  **A:** The file stopped using `from chispa.row_comparer import *` and now explicitly imports only `are_rows_equal_enhanced` and `are_rows_approx_equal` from `chispa.row_comparer`. It also adds `DefaultFormats` from `chispa.default_formats` to support the new formatting parameter.

- **[dependency] Q:** What new dependency or import was added to support the change, and where is it used?
  **A:** `chispa.default_formats.DefaultFormats` was imported in both `chispa/__init__.py` and `chispa/dataframe_comparer.py`. It is used as the default value for the new `formats` parameter and stored on the `Chispa` class instance as `self.formats`.

- **[scope] Q:** Which files are touched by this commit, and what does each one change?
  **A:** `chispa/__init__.py` adds the `Chispa` class and re-exports `DefaultFormats`. `chispa/dataframe_comparer.py` adds `DefaultFormats` imports, changes the signatures of `assert_df_equality` and `assert_approx_df_equality`, and passes `formats` into row assertion helpers.

- **[usage] Q:** How would a developer use the new `Chispa` class to run a DataFrame equality assertion with custom formats?
  **A:** A developer would instantiate `Chispa` with a formats object, for example `chispa = Chispa(formats=DefaultFormats())`, and then call `chispa.assert_df_equality(df1, df2, ...)`. The method forwards the stored `formats` value to the module-level `assert_df_equality` function.

- **[edge_case] Q:** What default handling is added for the new formatting configuration, especially when callers do not provide one?
  **A:** Both the new `Chispa` constructor and the updated comparison functions default `formats` to `DefaultFormats()`. That means callers who do not pass a formats object still get a concrete default formatting configuration.

- **[risk] Q:** What backward-compatibility or regression risk does this change introduce for callers of the comparison helpers?
  **A:** The comparison helpers now rely on an extra `formats` argument being threaded through to `assert_generic_rows_equality` and `assert_basic_rows_equality`. If any external code depends on the previous exact signatures or on the older implicit formatting behavior, it may need to be updated to account for the new `formats` parameter and `DefaultFormats` default.

### MrPowers/chispa @ commit #47 `39c3bb18c7` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** The commit modernizes and cleans up chispa’s public code and project metadata by reformatting Python modules, switching the PrettyTable import to the external `prettytable` package, tightening some `None` checks, and adding GitHub issue and pull request templates.

- **[new_api] Q:** What new project file was introduced for feature requests, and what does it contain?
  **A:** A new `.github/ISSUE_TEMPLATE/feature_request.md` file was added. It defines a GitHub issue template for feature requests with front matter (`name: Feature request`, `about: Suggest a new feature`, `labels: "enhancement"`) and sections for describing the problem, the desired solution, additional context, and whether the reporter plans to create a PR.

- **[signature_change] Q:** How did the `chispa.Chispa.assert_df_equality` method signature change in `chispa/__init__.py`?
  **A:** `Chispa.assert_df_equality` was reformatted onto multiple lines, but its parameters stayed the same: `df1, df2, ignore_nullable=False, transforms=None, allow_nan_equality=False, ignore_column_order=False, ignore_row_order=False, underline_cells=False, ignore_metadata=False`. The method still forwards those arguments to `assert_df_equality` along with `self.formats`.

- **[behavior_change] Q:** How does `assert_approx_column_equality` now treat `None` values differently at runtime?
  **A:** In `chispa/column_comparer.py`, the comparisons now use identity checks (`is None` / `is not None`) instead of equality checks (`== None` / `!= None`). As a result, the function’s handling of missing values is based on actual `None` identity rather than general equality semantics.

- **[refactor] Q:** What code structure was simplified or reformatted in `chispa/__init__.py` and `chispa/dataframe_comparer.py`?
  **A:** Several imports and function definitions were reformatted into parenthesized multi-line blocks for readability, and the `Chispa` class declaration was simplified from `class Chispa():` to `class Chispa:`. The exception classes in `column_comparer.py` and `dataframe_comparer.py` were also reindented consistently.

- **[dependency] Q:** What import change was made for PrettyTable, and why is it significant from the diff?
  **A:** `chispa/column_comparer.py` changed from `from chispa.prettytable import PrettyTable` to `from prettytable import PrettyTable`. This removes the vendored in-repo dependency and uses the external `prettytable` package instead, matching the README’s deletion of the vendored-dependencies section.

- **[scope] Q:** Which files were changed in this commit, and what kind of change did each receive?
  **A:** The commit adds `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`, and `.github/pull_request_template.md` as new GitHub templates. It removes the vendored-dependencies section from `README.md`, updates `chispa/__init__.py`, `chispa/bcolors.py`, `chispa/column_comparer.py`, `chispa/dataframe_comparer.py`, and `chispa/rows_comparer.py` with formatting and small code changes, and it does not add new library APIs beyond those existing modules.

- **[usage] Q:** How would a developer now use the updated PrettyTable-backed column comparison code?
  **A:** A developer would still call `assert_approx_column_equality(df, col_name1, col_name2, precision)` from `chispa.column_comparer`. The function now creates its table with `PrettyTable` imported from the external `prettytable` package, and it raises `ColumnsNotEqualError` with the formatted table string when the columns do not match within the given precision.

- **[edge_case] Q:** What edge-case handling around `None` values is present in the updated approximate column comparison logic?
  **A:** In `assert_approx_column_equality`, if one element is `None` and the other is not, the rows are marked unequal; if both elements are `None`, the pair is treated as equal and both values are added to the table using colored placeholders. This logic now relies on `is None` identity checks.

- **[risk] Q:** What compatibility or regression risk could come from this commit’s dependency and comparison changes?
  **A:** The switch from `chispa.prettytable` to `prettytable` may break environments that relied on the vendored module being present, and the new `is None` checks in `assert_approx_column_equality` slightly change comparison semantics compared with `== None` / `!= None`. Both changes could affect existing users who depended on the previous packaging or behavior.

### MrPowers/chispa @ commit #48 `1d983b78b0` (in_repo_split=val)

- **[intent] Q:** What is the main purpose of this commit in the row and column comparison helpers?
  **A:** The change is a small style/safety cleanup in `chispa/column_comparer.py` and `chispa/rows_comparer.py`: it replaces boolean comparisons written as `== False` with identity checks using `is False` before raising mismatch errors.

- **[new_api] Q:** Did this commit introduce any new public function, method, or class in the comparison API?
  **A:** No new API was added. The existing functions `assert_approx_column_equality`, `assert_basic_rows_equality`, and `assert_generic_rows_equality` were unchanged in name and purpose; only their final boolean checks were rewritten.

- **[signature_change] Q:** Did any function signature or type annotation change in the updated comparer functions?
  **A:** No. The signatures of `assert_approx_column_equality(df, col_name1, col_name2, precision)`, `assert_basic_rows_equality(rows1, rows2, underline_cells=False, formats=DefaultFormats)`, and `assert_generic_rows_equality(...)` were not modified.

- **[behavior_change] Q:** How does runtime behavior differ after this change when the comparison flag is false?
  **A:** At runtime, the code still raises `ColumnsNotEqualError` or `chispa.DataFramesNotEqualError` when `all_rows_equal` is false; the only difference is that the condition is now checked with `is False` instead of `== False`, which narrows the check to the literal `False` value.

- **[refactor] Q:** Why were the final mismatch checks in the comparer functions rewritten from `== False` to `is False`?
  **A:** They were simplified to use Python’s preferred identity check for booleans. This makes the intent explicit: the exception is raised only when `all_rows_equal` is exactly `False`.

- **[dependency] Q:** Were any imports or external dependencies changed by this commit?
  **A:** No imports or dependencies were added, removed, or updated. The diff only changes boolean comparison expressions inside `chispa/column_comparer.py` and `chispa/rows_comparer.py`.

- **[scope] Q:** Which files are touched by this commit, and what changes were made in each?
  **A:** Two files were modified: `chispa/column_comparer.py`, where `assert_approx_column_equality` now uses `if all_rows_equal is False:`, and `chispa/rows_comparer.py`, where both `assert_basic_rows_equality` and `assert_generic_rows_equality` now use `if all_rows_equal is False:`.

- **[usage] Q:** How would a developer observe the effect of this change when calling the comparison helpers?
  **A:** A developer would use the same functions as before, such as `assert_basic_rows_equality(...)` or `assert_approx_column_equality(...)`; if the rows or columns do not match, the functions still raise the same mismatch exceptions, but the internal false-check now relies on `is False`.

- **[edge_case] Q:** What edge case does the new `is False` check cover differently from `== False` in these helpers?
  **A:** The new check only triggers when `all_rows_equal` is exactly the boolean `False`, not just any value that compares equal to `False`. This affects the final guard in `assert_approx_column_equality`, `assert_basic_rows_equality`, and `assert_generic_rows_equality`.

- **[risk] Q:** What compatibility or regression risk could this boolean-comparison cleanup introduce?
  **A:** The main risk is a behavioral narrowing: if `all_rows_equal` were ever a non-boolean falsey value that previously satisfied `== False`, the new `is False` check would not raise. In the current code, though, the change is limited to the final mismatch guards in the two comparer modules.

### MrPowers/chispa @ commit #49 `34b37b11e1` (in_repo_split=val)

- **[intent] Q:** What is the main purpose of this commit in the chispa package?
  **A:** This commit cleans up the package’s public exports and import usage: it adds an explicit `__all__` in `chispa/__init__.py`, and replaces wildcard color imports with specific names in the comparer modules to make the exposed API and imports more explicit.

- **[new_api] Q:** What new package-level API definition was introduced in `chispa/__init__.py`, and what does it expose?
  **A:** A new `__all__` tuple was added to `chispa/__init__.py` that explicitly exposes `DataFramesNotEqualError`, `assert_df_equality`, `assert_approx_df_equality`, `ColumnsNotEqualError`, `assert_column_equality`, `assert_approx_column_equality`, `assert_basic_rows_equality`, `DefaultFormats`, and `Chispa`.

- **[signature_change] Q:** Did any function or class signatures change in this commit, and if not, what type-related change was made instead?
  **A:** No function or class signatures were changed in the diff. The only type-related adjustment was changing `__all__` in `chispa/structfield_comparer.py` from a string to a one-item tuple: `("are_structfields_equal",)`.

- **[behavior_change] Q:** How does runtime behavior change when importing symbols from `chispa` after this commit?
  **A:** Importing from `chispa` becomes more controlled because `chispa/__init__.py` now defines `__all__`, so wildcard imports from the package are limited to the listed names instead of relying on implicit export behavior.

- **[refactor] Q:** Why were the wildcard color imports in the comparer modules restructured?
  **A:** The wildcard imports from `chispa.bcolors` were replaced with explicit imports such as `bcolors` and `blue` to reduce namespace pollution and make it clearer which color constants each module actually uses.

- **[dependency] Q:** What import change was made in `chispa/__init__.py` related to `pyspark`, and why was it likely added?
  **A:** The line `from pyspark import context` was updated to `from pyspark import context  # noqa: F401`, which keeps the import while silencing an unused-import lint warning.

- **[scope] Q:** Which files were touched by this commit, and what kind of change happened in each one?
  **A:** Five files were changed: `chispa/__init__.py` added an explicit `__all__` and a lint-suppression comment on the `pyspark` import; `chispa/column_comparer.py` replaced `from chispa.bcolors import *` with `from chispa.bcolors import bcolors`; `chispa/rows_comparer.py` removed the wildcard `bcolors` import; `chispa/schema_comparer.py` replaced `from chispa.bcolors import *` with `from chispa.bcolors import blue`; and `chispa/structfield_comparer.py` changed `__all__` from a string to a tuple.

- **[usage] Q:** How would a developer now rely on the package’s exported names when importing from `chispa`?
  **A:** A developer can use the names listed in `chispa.__all__` for wildcard imports, for example `from chispa import *`, and receive the explicitly exported items such as `assert_df_equality`, `assert_column_equality`, `DefaultFormats`, and `Chispa`.

- **[edge_case] Q:** What edge-case or tooling-related handling was added around the `pyspark` import in `chispa/__init__.py`?
  **A:** The `pyspark` import in `chispa/__init__.py` is kept but marked with `# noqa: F401`, which handles the case where the import is needed for side effects or availability checks even though the imported `context` name is not otherwise referenced in that file.

- **[risk] Q:** What compatibility risk could the new explicit `__all__` introduce for code that imports from `chispa`?
  **A:** Code that depended on wildcard-importing names not included in the new `chispa.__all__` could stop receiving those symbols from `from chispa import *`, because the package now publishes only the explicitly listed exports.

### MrPowers/chispa @ commit #51 `3f6d88ae07` (in_repo_split=val)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** The commit is a cleanup-oriented refactor that mostly reorganizes imports across the package modules (`chispa/__init__.py`, `column_comparer.py`, `dataframe_comparer.py`, `row_comparer.py`, `rows_comparer.py`, and `schema_comparer.py`) without changing the public API or adding new comparison logic.

- **[new_api] Q:** What new function, method, or class is introduced by this change, and what does it do?
  **A:** No new public function, method, or class is introduced in this diff. The only visible change in exported symbols is import reordering in `chispa/__init__.py`, such as `DefaultFormats`, `assert_df_equality`, `assert_approx_df_equality`, `assert_column_equality`, and related error classes.

- **[signature_change] Q:** Did this commit change any function or method signatures or type annotations?
  **A:** No function or method signatures and no type annotations were changed in this diff. The edits are limited to import ordering and import-source adjustments, such as moving `DefaultFormats` and reordering imports in `rows_comparer.py` and `row_comparer.py`.

- **[behavior_change] Q:** How does runtime behavior differ after this change when the package modules are imported or executed?
  **A:** The runtime behavior shown by the diff does not change in terms of comparison logic; the code still exposes the same comparison helpers and error classes. The only observable effect is that imports are reorganized, which can affect module loading order but not the comparison algorithms themselves.

- **[refactor] Q:** Why were existing imports and module references restructured in this commit?
  **A:** They were restructured to make the import layout cleaner and more consistent, for example by grouping standard-library imports (`os`, `sys`, `glob`, `reduce`, `zip_longest`) separately from third-party and local `chispa` imports, and by reorganizing the order of imported helpers in `__init__.py` and the comparer modules.

- **[dependency] Q:** What import or dependency changes were made, and what do they indicate?
  **A:** The diff only rearranges existing imports; it does not add or remove dependencies. For example, `chispa.dataframe_comparer` now imports `reduce` before local modules, and `chispa.row_comparer` and `chispa.schema_comparer` have their imports reordered, but the same modules (`prettytable`, `pyspark.sql.Row`, and internal `chispa` helpers) are still used.

- **[scope] Q:** Which files are touched by this commit, and what kind of change does each file receive?
  **A:** Six files are touched: `chispa/__init__.py` reorders top-level exports/imports; `chispa/column_comparer.py` reorders imports; `chispa/dataframe_comparer.py` reorders imports and keeps the same helper imports; `chispa/row_comparer.py` reorders `math`, `Row`, and `number_helpers` imports; `chispa/rows_comparer.py` reorders `PrettyTable`, `chispa`, `DefaultFormats`, and `format_string` imports; and `chispa/schema_comparer.py` reorders `zip_longest` and `PrettyTable`/`blue` imports.

- **[usage] Q:** How would a developer use the changed code from this commit?
  **A:** A developer would still import the same public helpers from `chispa`, such as `DefaultFormats`, `assert_df_equality`, `assert_approx_df_equality`, `assert_column_equality`, `assert_approx_column_equality`, and `assert_basic_rows_equality`. The commit does not introduce a new calling pattern; it only changes how those names are ordered in the package initializer.

- **[edge_case] Q:** What default value or empty-input handling is visible in the changed code?
  **A:** The diff does not change any edge-case handling logic, but it shows that `assert_basic_rows_equality` still defaults its `formats` parameter to `DefaultFormats()`. That default remains in `chispa/rows_comparer.py` and is also re-exported from `chispa/__init__.py`.

- **[risk] Q:** What backward-compatibility or regression risk could this refactor introduce?
  **A:** The main risk is limited to import-time behavior: because the commit rearranges top-level imports in `chispa/__init__.py` and several comparer modules, any latent circular-import or import-order dependency could surface differently. The diff itself does not show a functional API change, so backward-compatibility risk appears low and centered on module initialization order.

### MrPowers/chispa @ commit #52 `580631aa33` (in_repo_split=val)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** It introduces a new formatting API based on `FormattingConfig`, `Color`, `Style`, and `Format`, and updates the library to accept those formatting objects instead of only the older `DefaultFormats` pattern.

- **[new_api] Q:** What new formatting-related API is exposed from `chispa/__init__.py`, and what does it represent?
  **A:** `chispa/__init__.py` now exports `Style`, `Color`, `FormattingConfig`, and `Format` alongside `DefaultFormats`. These names represent the new configurable formatting types used for error-message styling.

- **[signature_change] Q:** How did the `Chispa.__init__` signature change, and what type is now expected for the `formats` parameter?
  **A:** `Chispa.__init__` changed from `def __init__(self, formats=DefaultFormats(), default_output=None)` to `def __init__(self, formats: FormattingConfig | None = None, default_output=None)`. The `formats` argument is now annotated as `FormattingConfig | None`.

- **[behavior_change] Q:** How does runtime handling of the `formats` argument differ after this change?
  **A:** When `formats` is omitted or falsy, `Chispa.__init__` and `assert_df_equality` now create a new `FormattingConfig()` instance. If `formats` is already a `FormattingConfig`, it is used directly; otherwise it is converted with `FormattingConfig._from_arbitrary_dataclass(formats)`.

- **[refactor] Q:** Why was the old custom-formats example in `README.md` restructured?
  **A:** The example was rewritten to use `FormattingConfig` objects instead of a custom `@dataclass` named `MyFormats`, simplifying the documented API and matching the new formatting classes exposed by the package.

- **[dependency] Q:** What import changes were made to support the new formatting system?
  **A:** Several modules now import from `chispa.formatting`, especially `Color`, `Format`, `FormattingConfig`, and `Style` in `chispa/__init__.py`, and `FormattingConfig` in `chispa/dataframe_comparer.py`. The old direct dependency on `DefaultFormats` as the default `formats` value was removed from `dataframe_comparer.py`.

- **[scope] Q:** Which files were touched in this commit, and what kind of change did each receive?
  **A:** `.github/ISSUE_TEMPLATE/bug_report.md` and `.github/ISSUE_TEMPLATE/feature_request.md` only gained trailing newlines; `NOTICE-binary.md` was added as a new license notice file; `README.md` was updated to document `FormattingConfig`; `chispa/__init__.py`, `chispa/bcolors.py`, `chispa/column_comparer.py`, and `chispa/dataframe_comparer.py` were updated to use future annotations and the new formatting API.

- **[usage] Q:** How should a developer now pass custom row and cell colors when calling `assert_basic_rows_equality`?
  **A:** They should construct a `FormattingConfig`, for example `formats = FormattingConfig(mismatched_rows={"color": "light_yellow"}, matched_rows={"color": "cyan", "style": "bold"}, mismatched_cells={"color": "purple"}, matched_cells={"color": "blue"})`, and then call `assert_basic_rows_equality(df1.collect(), df2.collect(), formats=formats)`.

- **[edge_case] Q:** What happens if the `formats` value is `None` or otherwise falsy when creating a `Chispa` instance?
  **A:** `Chispa.__init__` treats falsy `formats` values as missing and replaces them with a fresh `FormattingConfig()` instance before storing them on `self.formats`.

- **[risk] Q:** What compatibility risk does the new formatting handling introduce for existing callers?
  **A:** Callers that previously passed an arbitrary dataclass instance as `formats` now rely on `FormattingConfig._from_arbitrary_dataclass(formats)` to adapt it, so any dataclass shape that does not match the expected fields may no longer behave the same way as before.

### MrPowers/chispa @ commit #53 `aaf0005dc8` (in_repo_split=val)

- **[intent] Q:** What is the main purpose of this commit in `chispa/schema_comparer.py`?
  **A:** It extends the schema-comparison helpers so callers can optionally ignore `metadata` as well as nullability when comparing schemas, struct fields, and nested datatypes.

- **[new_api] Q:** What new capability does the updated `are_schemas_equal_ignore_nullable` function provide, and how is it exposed?
  **A:** `are_schemas_equal_ignore_nullable` now accepts an `ignore_metadata=False` parameter, allowing it to compare two schemas while optionally skipping metadata differences in addition to ignoring nullability.

- **[signature_change] Q:** Which function signatures were changed in this diff, and what new parameter was added?
  **A:** `are_schemas_equal_ignore_nullable(s1, s2)` became `are_schemas_equal_ignore_nullable(s1, s2, ignore_metadata=False)`, and `are_datatypes_equal_ignore_nullable(dt1, dt2)` became `are_datatypes_equal_ignore_nullable(dt1, dt2, ignore_metadata=False)`.

- **[behavior_change] Q:** How does runtime comparison behavior change after this patch when `ignore_metadata` is enabled?
  **A:** When `ignore_metadata` is passed through, schema and datatype comparisons will treat differing `metadata` as acceptable instead of returning `False`, including for nested structs reached through datatype comparison.

- **[refactor] Q:** Why was the comparison logic in `schema_comparer.py` restructured in this change?
  **A:** The logic was restructured so the `ignore_metadata` flag is propagated consistently through the helper chain, instead of being hardcoded in `assert_schema_equality_full` and omitted from deeper equality helpers.

- **[dependency] Q:** What internal dependency or call-chain change was made between comparison helpers, and why?
  **A:** `are_structfields_equal` now calls `are_datatypes_equal_ignore_nullable(sf1.dataType, sf2.dataType, ignore_metadata)`, and `are_datatypes_equal_ignore_nullable` now calls `are_schemas_equal_ignore_nullable(dt1, dt2, ignore_metadata)`, so the metadata-ignore setting is preserved across nested comparisons.

- **[scope] Q:** Which functions in `chispa/schema_comparer.py` were touched, and what kind of change did each receive?
  **A:** `assert_schema_equality_full` was updated to pass `ignore_nullable` and `ignore_metadata` into `are_structfields_equal`; `are_schemas_equal_ignore_nullable` gained an `ignore_metadata` parameter; `are_structfields_equal` now forwards `ignore_metadata` to datatype comparison; and `are_datatypes_equal_ignore_nullable` gained an `ignore_metadata` parameter and forwards it to nested schema comparison.

- **[usage] Q:** How would a developer use the updated helpers to ignore metadata differences during schema comparison?
  **A:** They would call `are_schemas_equal_ignore_nullable(s1, s2, ignore_metadata=True)` or use a path that passes `ignore_metadata=True` into `assert_schema_equality_full`, so comparisons ignore both nullability and metadata differences.

- **[edge_case] Q:** How does the change affect comparison of nested struct fields when their metadata differs but their nested types match?
  **A:** With `ignore_metadata=True`, nested structs are compared using the same metadata-ignoring behavior recursively, so matching nested types no longer fail solely because their metadata differs.

- **[risk] Q:** What backward-compatibility or regression risk does this commit introduce for existing callers?
  **A:** Because `ignore_metadata` now defaults to `False` on the updated helpers, existing callers keep the prior metadata-sensitive behavior unless they opt in, but any code that relied on the old fixed behavior of `assert_schema_equality_full` always passing `True` into `are_structfields_equal` could see different results if it now expects the flag to control metadata handling.

### MrPowers/chispa @ commit #54 `0e513eda31` (in_repo_split=test)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** The commit moves color-formatting helpers into the `chispa.formatting` package, deprecates the old `chispa.bcolors` helpers, and updates comparers to use the new formatting API while also fixing README image links and adding a CONTRIBUTING guide.

- **[new_api] Q:** What new formatting helper is introduced in `chispa/formatting/format_string.py`, and what does it return?
  **A:** A new `blue(string: str)` function is added to `chispa/formatting/format_string.py`; it returns the input string wrapped with `Color.LIGHT_BLUE` at the start and `Color.LIGHT_RED` at the end.

- **[signature_change] Q:** How did the `assert_schema_equality_full` call to `are_structfields_equal` change in `chispa/schema_comparer.py`?
  **A:** Inside `assert_schema_equality_full`, the code now calls `are_structfields_equal(sf1, sf2, True)` instead of passing through `ignore_nullable` and `ignore_metadata`, so the comparison is now hardcoded to ignore nullable values in that branch.

- **[behavior_change] Q:** How does the runtime behavior of the old `bcolors`, `blue`, and `underline_text` helpers change after this commit?
  **A:** Instantiating `bcolors` now emits a `DeprecationWarning`, and calling `chispa.bcolors.blue()` or `chispa.bcolors.underline_text()` also emits a `DeprecationWarning` before returning the formatted string.

- **[refactor] Q:** Why were the repeated ANSI color concatenations in the comparer modules restructured?
  **A:** The repeated `bcolors.LightBlue + ... + bcolors.LightRed` string construction was replaced with the shared `chispa.formatting.blue()` helper to centralize the formatting logic and avoid duplicating the same color-wrapping code in multiple modules.

- **[dependency] Q:** What import dependency changes were made to support the new formatting helper, and where were they applied?
  **A:** `chispa/column_comparer.py` and `chispa/schema_comparer.py` switched from importing `blue` from `chispa.bcolors` to importing it from `chispa.formatting`, and `chispa/formatting/__init__.py` now re-exports `blue` from `chispa.formatting.format_string`.

- **[scope] Q:** Which files were changed in this commit, and what kind of change did each file receive?
  **A:** `CONTRIBUTING.md` was added with release instructions; `README.md` had several image links updated from `github.com/blob/...` to `raw.githubusercontent.com/...`; `chispa/bcolors.py` gained deprecation warnings; `chispa/column_comparer.py` and `chispa/schema_comparer.py` were updated to use the new formatting helper; `chispa/formatting/__init__.py` started exporting `blue`; and `chispa/formatting/format_string.py` added the new `blue` function and imported `Color`.

- **[usage] Q:** How would a developer now use the new shared blue-formatting helper from the public formatting package?
  **A:** A developer can import it with `from chispa.formatting import blue` and call `blue("text")` to get the same light-blue/light-red wrapped string used by the comparers.

- **[edge_case] Q:** What default or special handling is now applied in the full schema equality path regarding nullability?
  **A:** In `assert_schema_equality_full`, the comparison path now always passes `True` as the third argument to `are_structfields_equal`, so the helper ignores nullable differences in that row-highlighting branch.

- **[risk] Q:** What compatibility risk does this commit introduce for code that relied on the old color helpers?
  **A:** Code that instantiates `chispa.bcolors.bcolors` or calls `chispa.bcolors.blue` / `chispa.bcolors.underline_text` will still work for now, but it will start emitting `DeprecationWarning`s and may break later when those APIs are removed.

### MrPowers/chispa @ commit #55 `25791ff43c` (in_repo_split=test)

- **[intent] Q:** What is the main purpose of this commit in the `MrPowers/chispa` repository?
  **A:** The commit adds and tightens type annotations across the public comparison/formatting API, while also updating a few helper behaviors such as how `assert_df_equality` and `assert_approx_df_equality` pass named arguments internally and how `Format.from_dict` validates its inputs.

- **[new_api] Q:** What new typed API surface is introduced on the `Chispa` class in `chispa/__init__.py`?
  **A:** The `Chispa.assert_df_equality` method is now declared as `assert_df_equality(self, df1: DataFrame, df2: DataFrame, ignore_nullable: bool = False, transforms: list[Callable] | None = None, allow_nan_equality: bool = False, ignore_column_order: bool = False, ignore_row_order: bool = False, underline_cells: bool = False, ignore_metadata: bool = False) -> None`, making the DataFrame arguments and optional `transforms` parameter explicit.

- **[signature_change] Q:** Which function signature was changed in `chispa/dataframe_comparer.py`, and what are the new parameter types or return annotations?
  **A:** `assert_df_equality` now takes `df1: DataFrame` and `df2: DataFrame`, annotates `ignore_nullable` and the boolean flags as `bool`, changes `transforms` to `list[Callable] | None`, and returns `None`; `assert_approx_df_equality` received the same style of typing plus `precision: float`; and `are_dfs_equal` is now annotated as `def are_dfs_equal(df1: DataFrame, df2: DataFrame) -> bool`.

- **[behavior_change] Q:** How does the runtime behavior of approximate DataFrame equality checks change after this commit?
  **A:** `assert_approx_df_equality` now passes keyword-style arguments into the row comparers instead of positional lists: it calls `assert_generic_rows_equality(..., are_rows_approx_equal, {

- **[refactor] Q:** What existing code was simplified or restructured in the column comparison helpers, and why does it look cleaner now?
  **A:** In `chispa/column_comparer.py`, the temporary variable names were simplified from `elements`, `colName1Elements`, and `colName2Elements` to `rows`, `col_name_1_elements`, and `col_name_2_elements`, and list comprehensions replaced `map(lambda ...)`, making the extraction of selected column values more readable.

- **[dependency] Q:** What dependency- or import-related adjustments were made, and what do they support?
  **A:** Several modules now import `DataFrame` from `pyspark.sql` and `Callable` from `typing` so their public function signatures can be explicitly typed. Also, the `findspark` import in `chispa/__init__.py` is marked with `# type: ignore[import-untyped]` to silence type-checker complaints about that untyped dependency.

- **[scope] Q:** Which files are touched by this diff, and what kind of change does each file receive?
  **A:** The diff touches `chispa/__init__.py` (typed `Chispa` APIs and constructor cleanup), `chispa/bcolors.py` and `chispa/default_formats.py` (annotated `__init__`/`__post_init__` return types), `chispa/column_comparer.py` (typed column comparison helpers and small refactors), `chispa/dataframe_comparer.py` (typed equality helpers and keyword argument changes), `chispa/formatting/format_string.py` (typed `blue` return value), `chispa/formatting/formats.py` and `chispa/formatting/formatting_config.py` (stricter format typing and validation), and `chispa/number_helpers.py` (typed numeric helpers and `Decimal` support).

- **[usage] Q:** How would a developer call one of the newly typed comparison methods after this change?
  **A:** A developer would still call `Chispa.assert_df_equality(df1, df2, ...)` or the module-level `assert_df_equality(df1, df2, ...)`, but the expected inputs are now `pyspark.sql.DataFrame` objects, and the optional `transforms` argument is documented by type as `list[Callable] | None`.

- **[edge_case] Q:** What edge case handling was added to `Format.from_dict` and to the approximate column equality logic?
  **A:** `Format.from_dict` now raises `TypeError("The value for key 'color' should be a string, not a list!")` if the `color` entry is a list, and `assert_approx_column_equality` now treats the one-`None`/one-non-`None` case with the condition `(elements[0] is None) != (elements[1] is None)` so mixed null pairs are explicitly marked unequal.

- **[risk] Q:** What compatibility or regression risk does this commit introduce for existing callers?
  **A:** The stricter type annotations and the new `Format.from_dict` validation may surface previously unchecked inputs as type errors, and the internal switch from positional lists to keyword dictionaries for row comparers could affect any code relying on the old calling convention or on permissive handling of list-valued `color` fields.

### MrPowers/chispa @ commit #56 `50c2411e88` (in_repo_split=test)

- **[risk] Q:** What compatibility or regression risk is introduced by the new schema diff handling?
  **A:** The error message format for schema mismatches is no longer always the same PrettyTable output; it can now switch to a colored tree representation for wide or nested schemas, which may affect tests or tooling that compare exact exception text. Also, `print_schema_diff()` now raises `ValueError` for any `output_format` other than `"table"` or `"tree"`.

### MrPowers/chispa @ commit #57 `202ab2aa82` (in_repo_split=test)

- **[intent] Q:** What is the main purpose of this commit in chispa?
  **A:** This commit adds support for comparing Spark DataFrames while ignoring one or more specified columns, by introducing an `ignore_columns` option to the DataFrame equality helpers and documenting its use in `README.md`.

- **[new_api] Q:** What new API option was introduced on the DataFrame equality helpers, and what does it do?
  **A:** A new `ignore_columns: list[str] | None = None` parameter was added to `assert_df_equality` and `assert_approx_df_equality`. When provided, each listed column is dropped from both DataFrames before schema and row equality checks are performed.

- **[signature_change] Q:** How did the function signatures change for the equality assertion helpers?
  **A:** Both `chispa/dataframe_comparer.py::assert_df_equality` and `chispa/dataframe_comparer.py::assert_approx_df_equality` gained a new optional parameter `ignore_columns: list[str] | None = None`. The `Chispa.assert_df_equality` wrapper in `chispa/__init__.py` was also updated to accept and forward the same parameter.

- **[behavior_change] Q:** How does runtime comparison behavior differ after this change when `ignore_columns` is set?
  **A:** If `ignore_columns` is non-empty, the code now appends a transform that calls `df.drop(*ignore_columns)` on both DataFrames before `assert_schema_equality` and row comparison. This means differences in those columns no longer affect equality results.

- **[refactor] Q:** Why was the transformation pipeline in the comparer updated instead of handling ignored columns separately later?
  **A:** The existing comparison flow already builds a list of DataFrame transforms and applies them with `reduce`. Adding `df.drop(*ignore_columns)` to that pipeline keeps ignored-column handling consistent with the existing column-order and row-order transforms and ensures both schema and row checks operate on the same reduced DataFrames.

- **[dependency] Q:** What dependency or import-related change was made to support the new ignored-columns feature?
  **A:** No new external dependency was added. The change reuses existing Spark DataFrame operations by adding a `drop(*ignore_columns)` transform inside `chispa/dataframe_comparer.py`, and the public `Chispa` wrapper in `chispa/__init__.py` simply forwards the new argument to `assert_df_equality`.

- **[scope] Q:** Which files were changed, and what kind of updates did each file receive?
  **A:** Three files were touched: `README.md` gained a new documentation section showing how to ignore specific columns; `chispa/dataframe_comparer.py` added the `ignore_columns` parameter and the logic to drop those columns before comparison; and `chispa/__init__.py` updated the `Chispa.assert_df_equality` wrapper to accept and pass through `ignore_columns`.

- **[usage] Q:** How would a developer use the new functionality in code?
  **A:** A developer can call `assert_df_equality(df1, df2, ignore_columns=["clean_name"])` to compare two DataFrames while ignoring the `clean_name` column. The same keyword can be passed through the `Chispa.assert_df_equality` method as well.

- **[edge_case] Q:** What happens if `ignore_columns` is not provided or is empty?
  **A:** The new transform is only added when `ignore_columns` is truthy, so `None` or an empty list leaves the comparison behavior unchanged. In that case, no columns are dropped before schema and row equality checks.

- **[risk] Q:** What compatibility or regression risk does this change introduce?
  **A:** Because ignored columns are dropped before `assert_schema_equality`, any code that previously relied on schema mismatches in those columns will now pass when `ignore_columns` is set. There is also a behavioral change if a listed column is missing or if dropping columns alters the remaining schema in ways that affect existing assertions.

### MrPowers/chispa @ commit #58 `20b05945b9` (in_repo_split=test)

- **[intent] Q:** What is the main purpose of the change in `chispa/schema_comparer.py`?
  **A:** It makes `are_datatypes_equal_ignore_nullable` compare additional datatype attributes for non-null, non-struct Spark types instead of treating every such pair as equal.

- **[new_api] Q:** Which existing function is effectively extended by this change, and what does the new logic do for non-struct datatypes?
  **A:** The existing function is `are_datatypes_equal_ignore_nullable(dt1, dt2, ignore_metadata: bool = False)`. For datatypes that are not nullable-equal structs, it now compares `vars(dt1)` and `vars(dt2)` so attributes like Decimal `precision` and `scale` must match.

- **[signature_change] Q:** Did the patch change any function signature or type annotation in `chispa/schema_comparer.py`?
  **A:** No signature or type annotation changed; the modification is entirely inside `are_datatypes_equal_ignore_nullable`, where the fallback return value was replaced with a comparison of the datatype instance attributes.

- **[behavior_change] Q:** How does runtime behavior change for non-struct Spark datatypes after this commit?
  **A:** Previously, any non-struct datatypes that reached the fallback branch were considered equal. After this change, they are only considered equal if their instance dictionaries match, so differences such as Decimal precision or scale now cause inequality.

- **[refactor] Q:** Why was the fallback branch in `are_datatypes_equal_ignore_nullable` restructured?
  **A:** It was restructured to avoid an overly permissive blanket `True` result and to make the equality check account for datatype-specific attributes that are stored on the objects, such as Decimal metadata.

- **[dependency] Q:** Did this change add or remove any imports or external dependencies in `chispa/schema_comparer.py`?
  **A:** No imports or external dependencies were changed; the patch only adds an inline comment and replaces `return True` with `return vars(dt1) == vars(dt2)` in the existing function.

- **[scope] Q:** Which file is touched by this commit, and what kind of change does it contain?
  **A:** Only `chispa/schema_comparer.py` is modified, and the change is a small logic update inside `are_datatypes_equal_ignore_nullable` plus an explanatory comment about checking datatype attributes.

- **[usage] Q:** How would a developer observe the effect of this change when using the schema comparison utilities?
  **A:** When calling schema comparison code that relies on `are_datatypes_equal_ignore_nullable`, two non-struct datatypes with different internal attributes, such as Decimal types with different `precision` or `scale`, will now compare as unequal instead of equal.

- **[edge_case] Q:** What edge case does the new comment and comparison logic explicitly address?
  **A:** It addresses Spark datatypes that carry extra attributes beyond their type name, with Decimal given as an example; those attributes are now part of equality checking via `vars(dt1) == vars(dt2)`.

- **[risk] Q:** What backward-compatibility risk does this change introduce for schema comparisons?
  **A:** Code that previously relied on all non-struct datatypes being treated as equal may now fail comparisons when datatype instance attributes differ, which can surface new test failures or stricter validation results.

### MrPowers/chispa @ commit #59 `ab283fff36` (in_repo_split=test)

- **[intent] Q:** What is the main purpose of this change in `chispa/dataframe_comparer.py`?
  **A:** It reorders the dataframe transformation pipeline in both `assert_df_equality` and `assert_approx_df_equality` so that `ignore_columns` is applied before `ignore_row_order`, instead of sorting rows before dropping columns.

- **[new_api] Q:** What new function, method, or class was introduced by this commit, and what does it do?
  **A:** No new API was introduced. The commit only changes the order of existing transformation steps inside `assert_df_equality` and `assert_approx_df_equality` in `chispa/dataframe_comparer.py`.

- **[signature_change] Q:** Did any function signature or type annotation change in this diff, and if so, what is the change?
  **A:** No function signatures or type annotations changed. The definitions of `assert_df_equality` and `assert_approx_df_equality` remain the same; only the order of `transforms.append(...)` calls changed.

- **[behavior_change] Q:** How does the runtime behavior of the dataframe comparison logic differ after this change?
  **A:** When both `ignore_columns` and `ignore_row_order` are enabled, the code now drops the listed columns before sorting rows. Before this change, it sorted by `df.columns` first and then dropped columns; now the reduce pipeline applies `df.drop(*ignore_columns)` before `df.sort(df.columns)`.

- **[refactor] Q:** Why was the transformation logic in `assert_df_equality` and `assert_approx_df_equality` restructured?
  **A:** The transformation list was reordered so that column dropping happens earlier in the pipeline. This simplifies the sequence of operations by ensuring `ignore_columns` is handled before row-order normalization in both comparison functions.

- **[dependency] Q:** What import or dependency usage changed in this diff, and what was the reason?
  **A:** No imports or external dependencies changed. The only affected code is the local transformation ordering in `chispa/dataframe_comparer.py`, using the existing `reduce`-based pipeline.

- **[scope] Q:** Which files or modules are touched by this commit, and what changes are made in each?
  **A:** Only `chispa/dataframe_comparer.py` is modified. In both `assert_df_equality` and `assert_approx_df_equality`, the `ignore_columns` transform is moved ahead of the `ignore_row_order` transform in the `transforms` list.

- **[usage] Q:** How would a developer use the changed comparison functions to take advantage of this update?
  **A:** A developer would call `assert_df_equality(...)` or `assert_approx_df_equality(...)` as before, optionally passing both `ignore_columns` and `ignore_row_order`. The functions now apply `df.drop(*ignore_columns)` before `df.sort(df.columns)` when those options are used together.

- **[edge_case] Q:** What edge case involving comparison options is affected by this change?
  **A:** The edge case is when both `ignore_columns` and `ignore_row_order` are set. In that case, the dataframe is now first reduced by dropping the specified columns and only then sorted by the remaining columns, instead of sorting before the drop.

- **[risk] Q:** What compatibility or regression risk does this transformation reordering introduce?
  **A:** The main risk is behavioral change for callers that rely on the previous ordering of operations when both `ignore_columns` and `ignore_row_order` are enabled. Any comparisons whose outcome depended on sorting before dropping columns could now behave differently because the transformations are applied in the opposite order.

### MrPowers/chispa @ commit #60 `3d7915663e` (in_repo_split=test)

- **[intent] Q:** What is the main purpose of this commit in the chispa package?
  **A:** The commit centralizes schema/data type string literals into enums and updates public exports/import handling, so schema comparison code uses typed constants like OutputFormat.TABLE/TREE and TypeName.STRUCT/ARRAY instead of raw strings.

- **[new_api] Q:** What new API is introduced in chispa/common_enums.py, and what does it represent?
  **A:** A new module, chispa/common_enums.py, introduces two string-valued enums: OutputFormat with TABLE="table" and TREE="tree", and TypeName with ARRAY="array" and STRUCT="struct". These enums provide named constants for schema output modes and Spark type names.

- **[signature_change] Q:** Which function signature was changed to use a new type annotation, and what is the new annotation?
  **A:** chispa.schema_comparer.print_schema_diff now annotates its output_format parameter as OutputFormat instead of str, with a default value of OutputFormat.TABLE.

- **[behavior_change] Q:** How does the runtime behavior of schema diff printing change after this commit?
  **A:** print_schema_diff now compares output_format against OutputFormat.TABLE and OutputFormat.TREE rather than bare string literals, and its error message now reports the allowed values via OutputFormat.__members__ instead of a hard-coded list.

- **[refactor] Q:** Why were string comparisons in schema comparison code restructured in this change?
  **A:** The code in chispa.schema_comparer was refactored to replace repeated raw string checks such as "struct" and "array" with the TypeName enum, making the type checks centralized and less error-prone.

- **[dependency] Q:** What import changes were made in the package and utility modules, and why?
  **A:** Several modules switched from typing.Callable to collections.abc.Callable: chispa/__init__.py, chispa/dataframe_comparer.py, and chispa/rows_comparer.py. This updates the Callable import to the standard collections.abc location.

- **[scope] Q:** Which files are touched by this commit, and what kind of change does each file receive?
  **A:** chispa/__init__.py removes the Spark import/bootstrap path logic, switches Callable imports, and rewrites __all__. chispa/common_enums.py is added with OutputFormat and TypeName enums. chispa/dataframe_comparer.py and chispa/rows_comparer.py update Callable imports. chispa/formatting/__init__.py only reorders __all__. chispa/schema_comparer.py adds enum imports and replaces string literals with enum constants.

- **[usage] Q:** How should a developer call the updated schema diff printer after this change?
  **A:** A developer should pass an OutputFormat enum value, for example print_schema_diff(s1, s2, ignore_nullable=False, ignore_metadata=False, output_format=OutputFormat.TABLE) or output_format=OutputFormat.TREE, rather than passing the raw strings "table" or "tree".

- **[edge_case] Q:** What default or fallback behavior exists for the schema diff output format parameter?
  **A:** print_schema_diff defaults output_format to OutputFormat.TABLE, so if no value is provided it prints the PrettyTable-based schema comparison. If a value other than OutputFormat.TABLE or OutputFormat.TREE is supplied, it raises a ValueError.

- **[risk] Q:** What compatibility risk does the new output_format typing introduce for existing callers?
  **A:** Existing callers that passed plain strings like "table" or "tree" may no longer match the intended enum-based API, since print_schema_diff now expects OutputFormat values. That can cause backward-compatibility issues if external code relies on string arguments or string-based comparisons.

