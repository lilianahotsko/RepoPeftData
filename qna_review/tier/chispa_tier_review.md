# Generated QA dataset review

## split: cr_test  (20 commits)

### MrPowers/chispa @ commit #0 `a75eeb69b0` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** It initializes the project with a minimal README, creates the chispa package, and adds basic Spark-related code including a local SparkSession and a column comparison helper.

- **[new_api] Q:** What new function is introduced in chispa/column_comparer.py, and what does it do?
  **A:** The new function is `assert_column_equality(df, col_name1, col_name2)`. It selects the two named columns from the DataFrame, collects them, compares the extracted value lists, and prints each pair of values, marking differing pairs with `MISMATCH:` when the two columns do not match.

- **[signature_change] Q:** What is the signature of the new column comparison function added by this change?
  **A:** The new function is defined as `assert_column_equality(df, col_name1, col_name2)`, taking three positional arguments: a DataFrame-like `df` and two column-name arguments, `col_name1` and `col_name2`.

- **[behavior_change] Q:** How does the runtime behavior of the column comparison logic work after this change?
  **A:** When the two collected column value lists are not equal, the function now iterates through the paired values and prints either `value | value` for matching pairs or `MISMATCH: value | value` for differing pairs. If the lists are equal, it does not print anything from this branch.

- **[scope] Q:** Which files were added in this commit, and what change does each file introduce?
  **A:** `README.md` was added with a one-line project title `# chispa`; `chispa/__init__.py` was added as an empty package initializer; `chispa/column_comparer.py` was added with the `assert_column_equality` function; and `chispa/spark.py` was added to create a SparkSession configured with `.master("local")` and `.appName("chispa")`.

### MrPowers/chispa @ commit #3 `02558e588b` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** It refactors the comparison helpers to produce richer, table-based error messages for DataFrame and column equality checks, while also extracting ANSI color codes into a separate `chispa.bcolors` module.

- **[new_api] Q:** What new API is introduced in `chispa/dataframe_comparer.py`, and what does it do?
  **A:** The new `assert_df_equality(df1, df2)` function compares two Spark DataFrames by schema and collected rows, and raises `SchemasNotEqualError` when the schemas differ or `DataFramesNotEqualError` with a `PrettyTable` diff when the rows differ. The module also adds `are_dfs_equal(df1, df2)`, which returns a boolean instead of raising.

- **[signature_change] Q:** Which comparison function signature or return type was changed to support the new DataFrame equality checks?
  **A:** A new boolean-style helper, `are_dfs_equal(df1, df2)`, was added in `chispa/dataframe_comparer.py` with the same two-DataFrame parameters as `assert_df_equality`, but it returns `True` or `False` instead of raising exceptions.

- **[behavior_change] Q:** How does the runtime behavior of failed column equality checks differ after this change?
  **A:** `assert_column_equality(df, col_name1, col_name2)` now raises `ColumnsNotEqualError` containing a `PrettyTable` string prefixed with a newline, instead of building a manually formatted line-by-line message. Matching values are also rendered with `bcolors.LightBlue` and `bcolors.LightRed` coloring in the table output.

- **[refactor] Q:** Why was the color-code definition in `chispa/column_comparer.py` restructured?
  **A:** The inline `bcolors` class was removed from `chispa/column_comparer.py` and replaced with an import from `chispa.bcolors`, separating ANSI color constants into their own module for reuse across comparison helpers.

- **[dependency] Q:** What new third-party import was added, and why is it needed?
  **A:** `PrettyTable` was added via `from prettytable import PrettyTable` in both `chispa/column_comparer.py` and `chispa/dataframe_comparer.py` so the comparison errors can render tabular diffs of mismatched values.

- **[scope] Q:** Which files are added, modified, or removed in this commit, and what does each change contain?
  **A:** `chispa/bcolors.py` is a new file defining ANSI color constants; `chispa/column_comparer.py` is modified to import `bcolors` and `PrettyTable` and to format column mismatch errors as tables; `chispa/dataframe_comparer.py` is a new module containing `DataFramesNotEqualError`, `SchemasNotEqualError`, `assert_df_equality`, and `are_dfs_equal`; `chispa/row_comparer.py` is added with an incomplete `are_rows_equal` stub; and `setup.py` is deleted.

- **[usage] Q:** How would a developer use the new DataFrame comparison helper added by this commit?
  **A:** A developer would call `assert_df_equality(df1, df2)` from `chispa.dataframe_comparer` to validate two DataFrames. If the schemas differ, it raises `SchemasNotEqualError`; if the collected rows differ, it raises `DataFramesNotEqualError` with a pretty-printed table of `df1` and `df2` values. For a non-raising check, they can call `are_dfs_equal(df1, df2)` and inspect the returned boolean.

### MrPowers/chispa @ commit #7 `d94ff473d4` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in `chispa/column_comparer.py`?
  **A:** The commit refactors `assert_approx_column_equality` so its handling of `None` values and approximate numeric comparisons is explicit and easier to read, with separate branches for `None` vs. non-`None`, `None` vs. `None`, and values whose difference is below `precision`.

- **[new_api] Q:** Which comparison function is updated in this change, and what cases does it now handle explicitly?
  **A:** `assert_approx_column_equality(df, col_name1, col_name2, precision)` in `chispa/column_comparer.py` now explicitly handles three cases: one value is `None` and the other is not, both values are `None`, and the absolute difference between the two values is less than `precision`.

- **[behavior_change] Q:** How does the runtime behavior of `assert_approx_column_equality` differ after this commit when comparing rows containing `None` or near-equal numbers?
  **A:** The comparison logic is unchanged in outcome, but it is now split into separate branches: rows are marked unequal when exactly one side is `None`, rows with both sides `None` are treated as equal, and rows with `abs(elements[0] - elements[1]) < precision` are treated as approximately equal.

- **[scope] Q:** Which files are touched by this commit, and what change is made in each one?
  **A:** `chispa/column_comparer.py` is modified to rewrite the conditional logic inside `assert_approx_column_equality` and add explanatory comments for each case. `chispa/row_comparer.py` is deleted entirely; the file previously contained only an incomplete stub, `def are_rows_equal:` followed by a blank line.

### MrPowers/chispa @ commit #10 `b413aa2cf8` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the change to `chispa/__init__.py` in this commit?
  **A:** The commit makes the `chispa` package expose its main comparison APIs at the top level by importing dataframe and column comparer functions and error classes into `chispa.__init__`.

- **[behavior_change] Q:** How does importing from `chispa` behave differently after this change?
  **A:** After this change, code can import `DataFramesNotEqualError`, `SchemasNotEqualError`, `assert_schema_equality`, `assert_df_equality`, `assert_approx_df_equality`, `ColumnsNotEqualError`, `assert_column_equality`, and `assert_approx_column_equality` directly from `chispa` instead of importing them from the underlying `dataframe_comparer` or `column_comparer` modules.

- **[scope] Q:** Which file is modified in this commit, and what does that file change contain?
  **A:** Only `chispa/__init__.py` is changed, and it is populated with two import statements: one re-exporting dataframe comparison errors and assertion helpers from `dataframe_comparer`, and another re-exporting column comparison errors and assertion helpers from `column_comparer`.

### MrPowers/chispa @ commit #14 `ca279766f4` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the changes in `chispa/dataframe_comparer.py`?
  **A:** The change updates DataFrame comparison logic to use `six.moves.zip_longest` instead of plain `zip`, so comparisons account for rows that exist in one DataFrame but not the other, especially in `assert_df_equality` and `assert_approx_df_equality`.

- **[new_api] Q:** What new module import is added in `chispa/dataframe_comparer.py`, and what is it used for?
  **A:** The file adds `import chispa.six as six`, and it is used to call `six.moves.zip_longest(rows1, rows2)` and `six.moves.zip_longest(df1_rows, df2_rows)` when pairing rows from two DataFrames for comparison.

- **[behavior_change] Q:** How does `assert_approx_df_equality` behave differently after this commit when the two DataFrames have different numbers of rows?
  **A:** `assert_approx_df_equality` now detects unmatched rows explicitly: if one row is `None` and the other is not, it marks the rows as unequal, sets `allRowsEqual` to `False`, and adds the pair to the pretty table. Before this change, it used plain `zip`, which would stop at the shorter collection and could miss extra rows.

- **[scope] Q:** Which file is changed by this commit, and what kinds of edits does it contain?
  **A:** Only `chispa/dataframe_comparer.py` is modified. The edits add an import of `chispa.six as six`, switch row pairing from `zip` to `six.moves.zip_longest` in both `assert_df_equality` and `assert_approx_df_equality`, and add special handling in `assert_approx_df_equality` for `None` versus non-`None` rows.

### MrPowers/chispa @ commit #18 `e2469ef4ba` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the changes made in `chispa/__init__.py`?
  **A:** The change adds startup logic to make PySpark importable when it is not already on the Python path, by trying `findspark` first and then falling back to a manual `SPARK_HOME`-based path setup.

- **[new_api] Q:** What new import-time behavior is introduced in `chispa.__init__`, and what does it do?
  **A:** `chispa/__init__.py` now includes import-time PySpark path bootstrapping: it tries `from pyspark import context`, then calls `findspark.init()` if `pyspark` is missing, and if `findspark` is unavailable it appends `SPARK_HOME/python` and the newest `py4j-*-src.zip` from `SPARK_HOME/python/lib` to `sys.path`.

- **[behavior_change] Q:** How does importing `chispa` behave differently after this commit when PySpark is not installed on the current path?
  **A:** Before the change, `chispa` only exposed its comparer functions; after the change, importing `chispa` first attempts to locate PySpark automatically. If `pyspark` cannot be imported, it tries `findspark`, then manually configures `sys.path` from `SPARK_HOME`, and if `SPARK_HOME` is missing it prints an error and exits with status `-1`.

- **[scope] Q:** Which file is modified by this commit, and what kind of change does it contain?
  **A:** Only `chispa/__init__.py` is modified. The file gains new import-time setup code for locating PySpark and `py4j`, while the existing exports of `DataFramesNotEqualError`, `assert_df_equality`, `assert_approx_df_equality`, `ColumnsNotEqualError`, `assert_column_equality`, and `assert_approx_column_equality` remain in place.

### MrPowers/chispa @ commit #22 `b1d5cef931` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in `chispa/dataframe_comparer.py`?
  **A:** It extends DataFrame equality assertions to support optional column-order and row-order ignoring, and it refactors the approximate-equality path into a reusable generic comparison helper.

- **[new_api] Q:** What new helper function is introduced in `chispa/dataframe_comparer.py`, and what does it do?
  **A:** `assert_generic_df_equality(df1, df2, row_equality_fun, row_equality_fun_args)` is added. It performs the shared schema and row comparison logic for DataFrames using a caller-provided row comparison function and argument list, and raises `DataFramesNotEqualError` when rows do not match.

- **[signature_change] Q:** How did the signature of `assert_df_equality` change in `chispa/dataframe_comparer.py`?
  **A:** `assert_df_equality` changed from `assert_df_equality(df1, df2, ignore_nullable=False, transforms=[], allow_nan_equality=False)` to `assert_df_equality(df1, df2, ignore_nullable=False, transforms=None, allow_nan_equality=False, ignore_column_order=False, ignore_row_order=False)`. The `transforms` default was changed from an empty list to `None`, and two new boolean parameters were added: `ignore_column_order` and `ignore_row_order`.

- **[behavior_change] Q:** How does row and column ordering affect equality checks after this change?
  **A:** `assert_df_equality` can now ignore column order by appending a transform that selects columns in sorted order, and ignore row order by appending a transform that sorts each DataFrame by `df.columns`. When `ignore_row_order` or `ignore_column_order` is enabled, those transforms are applied before the schema and row comparisons.

- **[scope] Q:** Which files were modified in this commit, and what changed in each one?
  **A:** Two files were changed: `chispa/dataframe_comparer.py`, where DataFrame equality logic was extended with new ordering options and a generic row-comparison helper, and `chispa/number_helpers.py`, where `isnan` was narrowed to catch only `TypeError` instead of catching all exceptions.

- **[usage] Q:** How would a developer use the new DataFrame equality options or helper introduced by this change?
  **A:** A developer can call `assert_df_equality(df1, df2, ignore_column_order=True)` to compare DataFrames while ignoring column order, or `assert_df_equality(df1, df2, ignore_row_order=True)` to ignore row order. For approximate comparisons, `assert_approx_df_equality(df1, df2, precision)` now delegates to `assert_generic_df_equality(df1, df2, are_rows_approx_equal, [precision])`, and that generic helper can also be called directly with a custom row comparison function and its arguments.

### MrPowers/chispa @ commit #27 `500793efe1` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of the README change in this commit?
  **A:** The README is updated to document that `assert_df_equality(..., ignore_nullable=True)` also ignores the nullable flag on elements inside an `ArrayType()`, not just the nullable flag on the top-level column schema.

- **[new_api] Q:** What does the new `ignore_nullable_property_array()` example function in the README demonstrate?
  **A:** The `ignore_nullable_property_array()` example shows two Spark DataFrames built with the same data but different `ArrayType(DoubleType(), ...)` nullability for the `coords` field, and it demonstrates the equality failure that `assert_df_equality(df1, df2)` can encounter without `ignore_nullable=True`.

- **[behavior_change] Q:** How does the documented runtime behavior of `assert_df_equality` change after this commit when `ignore_nullable=True` is used?
  **A:** After this change, `ignore_nullable=True` is documented to bypass nullability differences both for column fields and for the nullability of `ArrayType` elements, so DataFrames with the same values but different array-element nullability are treated as equal.

- **[scope] Q:** Which file is modified, and what kind of change is made there?
  **A:** Only `README.md` is modified, and the change is purely documentation: it adds a new explanation plus a Python example illustrating how `ignore_nullable=True` handles `ArrayType` element nullability.

### MrPowers/chispa @ commit #30 `5be6f9ce15` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this change to `chispa`'s DataFrame comparison logic?
  **A:** The change adds an `ignore_schema` option to `assert_df_equality` in `chispa/dataframe_comparer.py`, allowing callers to compare two DataFrames without running `assert_schema_equality` on their schemas.

- **[new_api] Q:** What new argument was added to `assert_df_equality`, and how does it affect the function?
  **A:** `assert_df_equality(df1, df2, ..., ignore_column_order=False, ignore_row_order=False, ignore_schema=False)` now includes a new boolean parameter named `ignore_schema`. When `ignore_schema` is `True`, the function skips the `assert_schema_equality(df1.schema, df2.schema, ignore_nullable)` check before comparing rows.

- **[behavior_change] Q:** How does DataFrame equality checking behave differently after this commit when schema differences are present?
  **A:** Before this commit, `assert_df_equality` always called `assert_schema_equality` and would fail on schema mismatches. After this commit, passing `ignore_schema=True` lets the comparison proceed without checking schema equality, so DataFrames can be compared even if their schemas differ.

- **[scope] Q:** Which files were modified, and what kind of change was made in each file?
  **A:** Two files were changed: `chispa/dataframe_comparer.py` was updated to add the `ignore_schema` parameter to `assert_df_equality` and conditionally skip schema validation, and `chispa/structfield_comparer.py` changed its import from `import six` to `import chispa.six as six`.

### MrPowers/chispa @ commit #33 `63e04316dd` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** This commit moves row-comparison logic into a new `chispa.rows_comparer` module and updates dataframe equality helpers to compare collected row lists explicitly, while also exporting `assert_basic_rows_equality` from the package root.

- **[new_api] Q:** What new function is introduced in `chispa/rows_comparer.py`, and what does it do?
  **A:** The new function `assert_basic_rows_equality(rows1, rows2)` compares two row sequences directly; if they differ, it builds a `PrettyTable` showing the mismatched rows and raises `chispa.DataFramesNotEqualError` with that table.

- **[signature_change] Q:** How did the call signatures or expected inputs for the row comparison helpers change in `chispa/dataframe_comparer.py`?
  **A:** `assert_generic_rows_equality` and `assert_basic_rows_equality` are now called with `df1.collect()` and `df2.collect()` instead of full DataFrame objects, so the row-comparison helpers in `chispa.rows_comparer` operate on collected row lists rather than DataFrames.

- **[behavior_change] Q:** How does runtime behavior differ in dataframe equality checks after this change?
  **A:** `assert_df_equality` and `assert_approx_df_equality` now collect both DataFrames before row comparison, so row comparison functions receive lists of rows directly; the mismatch reporting logic is unchanged, but the comparison no longer depends on the helpers calling `.collect()` internally.

- **[refactor] Q:** Why was the row comparison code restructured into a separate module?
  **A:** The row comparison implementations were extracted from `chispa/dataframe_comparer.py` into `chispa/rows_comparer.py` so that basic and generic row equality logic is centralized and reusable, while `dataframe_comparer.py` focuses on DataFrame-level orchestration.

- **[dependency] Q:** What import changes were made, and what was the reason implied by the diff?
  **A:** `chispa/dataframe_comparer.py` now imports `assert_basic_rows_equality` and `assert_generic_rows_equality` from `chispa.rows_comparer`, and `chispa/__init__.py` re-exports `assert_basic_rows_equality`; this reflects the move of row-comparison code into the new module.

- **[scope] Q:** Which files are touched by this commit, and what does each file change?
  **A:** `chispa/__init__.py` adds a package export for `assert_basic_rows_equality`; `chispa/dataframe_comparer.py` changes row comparisons to use collected rows and imports the moved helpers; `chispa/rows_comparer.py` is a new file containing `assert_basic_rows_equality` and `assert_generic_rows_equality`.

- **[usage] Q:** How would a developer use the newly exposed row-equality helper after this change?
  **A:** A developer can import `assert_basic_rows_equality` from `chispa` or `chispa.rows_comparer` and call it with two row sequences, for example `assert_basic_rows_equality(df1.collect(), df2.collect())`.

- **[edge_case] Q:** How does `assert_basic_rows_equality` handle unequal row lists when one side has extra rows or missing rows?
  **A:** It uses `six.moves.zip_longest(rows1, rows2)` to align the two sequences, so if one list is longer or shorter, the unmatched positions are included in the error table and a `chispa.DataFramesNotEqualError` is raised.

### MrPowers/chispa @ commit #37 `b92cf0805c` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the schema/structfield comparison code?
  **A:** The commit consolidates struct field equality logic into a single `are_structfields_equal(sf1, sf2, ignore_nullability=False)` function in `chispa/schema_comparer.py`, and updates the schema comparison helpers to call it with `True` when nullable differences should be ignored.

- **[new_api] Q:** What new comparison function is introduced in `chispa/schema_comparer.py`, and how does it behave?
  **A:** `are_structfields_equal(sf1, sf2, ignore_nullability=False)` is introduced. When `ignore_nullability` is `True`, it handles `None` values, compares `sf1.name` and `sf2.name`, and then delegates to `are_datatypes_equal_ignore_nullable(sf1.dataType, sf2.dataType)`; when `ignore_nullability` is `False`, it falls back to `sf1 == sf2`.

- **[signature_change] Q:** How did the struct field equality API change across the affected modules?
  **A:** The standalone helper `are_structfields_equal_ignore_nullable(sf1, sf2)` was removed from `chispa/schema_comparer.py`, and `chispa/structfield_comparer.py` now re-exports `are_structfields_equal` from `chispa.schema_comparer`. The new function signature is `are_structfields_equal(sf1, sf2, ignore_nullability=False)`, adding the `ignore_nullability` parameter.

- **[behavior_change] Q:** How does the runtime behavior of struct field comparisons change after this commit?
  **A:** Struct field comparisons no longer use a separate ignore-nullable helper. Calls that previously used `are_structfields_equal_ignore_nullable` now invoke `are_structfields_equal(..., True)`, so the ignore-nullability path is handled by the unified function, while the default path now returns `sf1 == sf2` when `ignore_nullability` is `False`.

- **[scope] Q:** Which files are modified in this commit, and what changes are made in each one?
  **A:** `chispa/schema_comparer.py` is updated to replace calls to `are_structfields_equal_ignore_nullable` with `are_structfields_equal(..., True)` and to define the new unified `are_structfields_equal` implementation. `chispa/structfield_comparer.py` is simplified to import and re-export `are_structfields_equal` from `chispa.schema_comparer`, removing its own wrapper function.

### MrPowers/chispa @ commit #40 `ba512da000` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the row-comparison code?
  **A:** The commit introduces configurable terminal formatting for row comparison output by extracting color/style handling into reusable helpers and a `DefaultFormats` container, then updating `assert_basic_rows_equality` to render matched and mismatched rows/cells with those formats.

- **[new_api] Q:** What new helper functions or classes were added, and what do they do?
  **A:** A new `DefaultFormats` class was added in `chispa/default_formats.py` with four format lists: `mismatched_rows`, `matched_rows`, `mismatched_cells`, and `matched_cells`. A new `format_string(input, formats)` helper was added in `chispa/terminal_str_formatter.py` to wrap text with ANSI codes, and `format_mismatched_cell(input_text: str, mismatched_cells) -> str` simply delegates to `format_string` for mismatched-cell styling.

- **[signature_change] Q:** How did the signature of `assert_basic_rows_equality` change in `chispa/rows_comparer.py`?
  **A:** `assert_basic_rows_equality` changed from `def assert_basic_rows_equality(rows1, rows2, underline_cells=False)` to `def assert_basic_rows_equality(rows1, rows2, underline_cells=False, formats=DefaultFormats())`, adding a new `formats` parameter that defaults to a `DefaultFormats()` instance.

- **[behavior_change] Q:** How does the runtime behavior of `assert_basic_rows_equality` differ after this change?
  **A:** The function now raises `chispa.DataFramesNotEqualError("\n" + "Number of rows is not equal")` immediately when `len(rows1) != len(rows2)`. When rows differ, it no longer uses the old underline-only branch; instead it builds `Row(...)` strings with ANSI formatting for each field, marking equal cells with `formats.matched_cells` and unequal cells with `formats.mismatched_cells`, and it formats equal rows with `formats.matched_rows`.

- **[refactor] Q:** What existing row-rendering logic was restructured, and what was the effect of that restructuring?
  **A:** The inline formatting logic that previously used `blue(r1)` / `blue(r2)` for equal rows and the `__underline_cells_in_row(...)` helper for mismatched rows was replaced with a generic `format_string`-based approach. This centralizes ANSI styling in `chispa/terminal_str_formatter.py` and makes the row rendering driven by the configurable `formats` object instead of hardcoded color helpers.

- **[scope] Q:** Which files were added or modified in this commit, and what does each change do?
  **A:** Three files are involved: `chispa/default_formats.py` was added to define the default row and cell formatting lists; `chispa/terminal_str_formatter.py` was added to provide ANSI formatting helpers; and `chispa/rows_comparer.py` was modified to import those helpers, accept a `formats` argument, check for unequal row counts, and format matched/mismatched rows and cells using the new helpers.

- **[usage] Q:** How would a developer use the updated row comparison code to control styling of mismatch output?
  **A:** A developer would call `assert_basic_rows_equality(rows1, rows2, formats=...)` and pass an object with the same four attributes as `DefaultFormats` (`mismatched_rows`, `matched_rows`, `mismatched_cells`, `matched_cells`). The function will use those lists when building the `PrettyTable` output for matched rows and mismatched cells.

### MrPowers/chispa @ commit #44 `957febe988` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit across the changed files?
  **A:** It changes row-comparison formatting so mismatched cells are highlighted in red instead of white, and it refactors `assert_generic_rows_equality` to build PrettyTable output using configurable `DefaultFormats`-based styling rather than hardcoded color strings and a separate underline helper.

- **[new_api] Q:** What new behavior does the `formats` parameter added to `assert_generic_rows_equality` provide?
  **A:** The `formats` parameter lets `assert_generic_rows_equality(rows1, rows2, row_equality_fun, row_equality_fun_args, underline_cells=False, formats=DefaultFormats())` use a `DefaultFormats` instance to style matched rows, mismatched rows, matched cells, and mismatched cells when building the error table.

- **[signature_change] Q:** How did the signature of `assert_generic_rows_equality` change in `chispa/rows_comparer.py`?
  **A:** Its signature changed from `assert_generic_rows_equality(rows1, rows2, row_equality_fun, row_equality_fun_args, underline_cells=False)` to `assert_generic_rows_equality(rows1, rows2, row_equality_fun, row_equality_fun_args, underline_cells=False, formats=DefaultFormats())`, adding a new optional `formats` argument with a `DefaultFormats()` default value.

- **[behavior_change] Q:** How does the runtime output for mismatched cells differ after this change?
  **A:** When two rows differ at a field, the cell text added to the PrettyTable is now wrapped with `format_string(..., formats.mismatched_cells)` instead of being underlined, and `DefaultFormats.mismatched_cells` now defaults to `['red', 'underline']` rather than `['white', 'underline']`.

- **[refactor] Q:** What existing helper or logic was removed from `chispa/rows_comparer.py`, and what replaced it?
  **A:** The private helper `__underline_cells_in_row` was removed, along with the special `underline_cells`-driven branch that built underlined row strings. It was replaced by inline per-field row formatting that iterates over `r1.__fields__` and `r2.__fields__`, compares each field, and formats matched or mismatched cells directly into strings for PrettyTable.

- **[dependency] Q:** What formatting-related dependency shift is visible in `rows_comparer.py`, and why was it introduced?
  **A:** The comparison code no longer builds output with hardcoded `bcolors.LightBlue` and `bcolors.LightRed` concatenation; instead it uses `format_string(...)` together with a `DefaultFormats` instance. This was introduced so row and cell styling comes from configurable format lists rather than fixed color constants.

- **[scope] Q:** Which files are modified in this commit, and what kind of change does each file receive?
  **A:** `chispa/default_formats.py` changes one default style value, updating `DefaultFormats.mismatched_cells` from `['white', 'underline']` to `['red', 'underline']`. `chispa/rows_comparer.py` refactors `assert_generic_rows_equality`, adds the `formats` parameter, removes `__underline_cells_in_row`, and changes how matching and mismatching rows/cells are rendered in the error table.

- **[usage] Q:** How would a developer use the updated row-comparison code to customize table styling?
  **A:** A developer can call `assert_generic_rows_equality(..., formats=some_formats)` and pass a `DefaultFormats`-like object to control the styling lists used for matched rows, mismatched rows, matched cells, and mismatched cells when the comparison raises `chispa.DataFramesNotEqualError`.

### MrPowers/chispa @ commit #47 `39c3bb18c7` (in_repo_split=train)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** It removes vendored third-party code and switches chispa to use external dependencies, while also adding GitHub issue and pull request templates and applying small code cleanup/formatting updates.

- **[new_api] Q:** What new repository workflow templates were introduced by this change, and what are they for?
  **A:** The commit adds `.github/ISSUE_TEMPLATE/bug_report.md` for bug reports, `.github/ISSUE_TEMPLATE/feature_request.md` for feature requests, and `.github/pull_request_template.md` for PR submissions with a checklist and description prompt.

- **[signature_change] Q:** Which public Python API signature was rewritten in this commit, and what are its parameters?
  **A:** `chispa.__init__.Chispa.assert_df_equality` was reformatted into a multi-line signature, but it keeps the same parameters: `df1`, `df2`, `ignore_nullable=False`, `transforms=None`, `allow_nan_equality=False`, `ignore_column_order=False`, `ignore_row_order=False`, `underline_cells=False`, and `ignore_metadata=False`.

- **[behavior_change] Q:** How does runtime behavior change in `chispa/column_comparer.py` when comparing values that may be `None`?
  **A:** `assert_approx_column_equality` now uses identity checks (`is None` / `is not None`) instead of equality comparisons (`== None` / `!= None`) when deciding whether one or both elements are `None`.

- **[refactor] Q:** Why was the `Chispa` class definition in `chispa/__init__.py` restructured during this commit?
  **A:** The class definition and the `assert_df_equality` method were reformatted for readability, with the method arguments and the call to `assert_df_equality` expanded across multiple lines and parentheses.

- **[dependency] Q:** What dependency import changed in `chispa/column_comparer.py`, and what does that imply about the project layout?
  **A:** `PrettyTable` is now imported from the external `prettytable` package with `from prettytable import PrettyTable` instead of `from chispa.prettytable import PrettyTable`, which indicates chispa no longer vendors PrettyTable in its own package.

- **[scope] Q:** Which files or modules were touched by this commit, and what kind of change did each receive?
  **A:** `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`, and `.github/pull_request_template.md` were added as new repository templates; `README.md` had the vendored-dependencies section removed; `chispa/__init__.py`, `chispa/bcolors.py`, `chispa/column_comparer.py`, and `chispa/dataframe_comparer.py` were updated with formatting, import, and comparison-logic cleanup.

- **[usage] Q:** How would a developer use the changed `Chispa.assert_df_equality` method after this commit?
  **A:** A developer would instantiate `Chispa` and call `assert_df_equality(df1, df2, ...)` on the instance; the method still forwards the DataFrames plus options like `ignore_nullable`, `transforms`, `allow_nan_equality`, `ignore_column_order`, `ignore_row_order`, `underline_cells`, and `ignore_metadata` to the module-level `assert_df_equality` function using `self.formats`.

- **[edge_case] Q:** How does `assert_approx_column_equality` handle the case where both compared column values are `None`?
  **A:** When both `elements[0]` and `elements[1]` are `None`, the code treats them as equal for that row and adds the formatted pair to the table without marking the rows unequal.

- **[risk] Q:** What backward-compatibility risk does this commit introduce by changing how `PrettyTable` is imported?
  **A:** Because `chispa/column_comparer.py` now depends on the external `prettytable` package instead of `chispa.prettytable`, environments that previously relied on the vendored module without installing the package may fail at import time unless `prettytable` is available.

### MrPowers/chispa @ commit #48 `1d983b78b0` (in_repo_split=val)

- **[intent] Q:** What is the main purpose of this commit across the comparer functions in chispa?
  **A:** The commit makes the equality checks in `chispa/column_comparer.py` and `chispa/rows_comparer.py` use identity comparison against `False` (`all_rows_equal is False`) instead of equality comparison (`all_rows_equal == False`) before raising the relevant mismatch exceptions.

- **[new_api] Q:** What new function, method, or class was introduced by this change, and what does it do?
  **A:** No new function, method, or class was introduced. The change only updates conditional checks inside existing functions: `assert_approx_column_equality` in `chispa/column_comparer.py`, and `assert_basic_rows_equality` and `assert_generic_rows_equality` in `chispa/rows_comparer.py`.

- **[behavior_change] Q:** How does the runtime behavior change after this commit when the row or column comparison result is false?
  **A:** When `all_rows_equal` is false in the affected functions, the code now tests with `is False` rather than `== False` before raising `ColumnsNotEqualError` or `chispa.DataFramesNotEqualError`. The error-raising path and messages remain the same; only the boolean check syntax changes.

- **[scope] Q:** Which files are modified in this commit, and what kind of change does each file receive?
  **A:** Two files are modified: `chispa/column_comparer.py`, where `assert_approx_column_equality` changes its final condition from `if all_rows_equal == False:` to `if all_rows_equal is False:`, and `chispa/rows_comparer.py`, where both `assert_basic_rows_equality` and `assert_generic_rows_equality` make the same `== False` to `is False` update.

### MrPowers/chispa @ commit #51 `3f6d88ae07` (in_repo_split=val)

- **[intent] Q:** What is the main purpose of this commit in the chispa package?
  **A:** The commit is a cleanup/refactor that only reorders and groups imports across the package modules; it does not change the comparison logic or add new runtime behavior.

- **[new_api] Q:** What top-level package symbol is now imported in chispa/__init__.py, and what is it used for?
  **A:** `DefaultFormats` is imported from `chispa.default_formats` at the package level; it provides the default formatting configuration used by chispa's comparison helpers.

- **[signature_change] Q:** Which function or method signature changed in this commit?
  **A:** No function or method signatures changed in this diff; the changes are limited to import ordering and import grouping.

- **[behavior_change] Q:** How does the runtime behavior of chispa change after this commit?
  **A:** The runtime comparison behavior does not change. The diff only rearranges imports, so helpers such as `assert_df_equality`, `assert_column_equality`, and the row/schema comparers keep the same behavior.

- **[refactor] Q:** Why were the import statements in several chispa modules restructured?
  **A:** They were rearranged to group related imports together and place them in a cleaner order, such as standard-library imports first and chispa/third-party imports grouped consistently.

- **[scope] Q:** Which files are touched by this commit, and what kind of change does each one receive?
  **A:** Six files are modified: `chispa/__init__.py` reorders and re-groups its exports/imports, `chispa/column_comparer.py` reorders imports, `chispa/dataframe_comparer.py` reorders imports, `chispa/row_comparer.py` reorders imports, `chispa/rows_comparer.py` reorders imports, and `chispa/schema_comparer.py` reorders imports.

- **[usage] Q:** How would a developer import the `DefaultFormats` class after this change?
  **A:** A developer can import it directly from the package initializer with `from chispa import DefaultFormats`.

### MrPowers/chispa @ commit #53 `aaf0005dc8` (in_repo_split=val)

- **[intent] Q:** What is the main purpose of this change in `chispa/schema_comparer.py`?
  **A:** The change extends the schema/data type comparison helpers so they can also ignore field metadata, not just nullability, by threading an `ignore_metadata` flag through the comparison functions.

- **[new_api] Q:** What new capability does `are_schemas_equal_ignore_nullable` provide after this commit, and how is it exposed in the function definition?
  **A:** `are_schemas_equal_ignore_nullable` now accepts an optional `ignore_metadata=False` parameter, allowing callers to compare two schemas while ignoring metadata differences in addition to nullability differences.

- **[signature_change] Q:** Which function signatures were changed in `chispa/schema_comparer.py`, and what new parameter was added?
  **A:** `are_schemas_equal_ignore_nullable(s1, s2)` was changed to `are_schemas_equal_ignore_nullable(s1, s2, ignore_metadata=False)`, and `are_datatypes_equal_ignore_nullable(dt1, dt2)` was changed to `are_datatypes_equal_ignore_nullable(dt1, dt2, ignore_metadata=False)`.

- **[behavior_change] Q:** How does the runtime comparison behavior differ after this commit when metadata differs between schemas or nested structs?
  **A:** Before this change, the nullability-ignoring comparison path always treated metadata as significant because it hardcoded the field comparison to `are_structfields_equal(..., True)`. After the change, `ignore_metadata` is passed through `assert_schema_equality_full`, `are_schemas_equal_ignore_nullable`, and `are_datatypes_equal_ignore_nullable`, so metadata differences can be ignored when that flag is set, including inside nested struct comparisons.

- **[scope] Q:** Which functions in `chispa/schema_comparer.py` were touched, and what kind of changes were made to each?
  **A:** Three comparison paths were updated in `chispa/schema_comparer.py`: `assert_schema_equality_full` now forwards `ignore_nullable` and `ignore_metadata` to `are_structfields_equal`; `are_schemas_equal_ignore_nullable` now takes an `ignore_metadata` parameter and passes it through; and `are_datatypes_equal_ignore_nullable` now takes `ignore_metadata=False` and forwards it into recursive struct comparisons via `are_schemas_equal_ignore_nullable`.

### MrPowers/chispa @ commit #54 `0e513eda31` (in_repo_split=test)

- **[intent] Q:** What is the main purpose of this commit in the chispa repository?
  **A:** This commit centralizes color-formatting helpers under `chispa.formatting`, deprecates the old `chispa.bcolors` helpers, and updates column/schema comparison code to use the new formatting helper while also fixing README image links to use `raw.githubusercontent.com` URLs.

- **[new_api] Q:** What new helper function was added in `chispa/formatting/format_string.py`, and what does it return?
  **A:** A new `blue(string: str)` function was added to `chispa/formatting/format_string.py`. It returns `Color.LIGHT_BLUE + string + Color.LIGHT_RED`, producing the same blue-highlighted text style used by the comparison output.

- **[signature_change] Q:** Which function signatures or type annotations changed in `chispa/formatting/format_string.py`?
  **A:** The file now imports `Color` in addition to `RESET` and `Format`, and it adds a new `blue(string: str)` function without an explicit return type annotation. The existing `format_string(input_string: str, format: Format) -> str` signature is unchanged.

- **[behavior_change] Q:** How does the runtime behavior of the schema comparison output change after this commit?
  **A:** `assert_schema_equality_full`, `assert_schema_equality_ignore_nullable`, and `assert_basic_schema_equality` now call `blue(str(sf1))` and `blue(str(sf2))` when rows are considered equal, so the pretty-printed schema rows are formatted through the new helper using stringified values instead of passing the raw struct fields directly.

- **[refactor] Q:** Why were the column and schema comparison modules restructured to use `chispa.formatting.blue` instead of `chispa.bcolors.blue`?
  **A:** The change moves formatting logic away from `chispa.bcolors` and into the `chispa.formatting` package, which makes `column_comparer.py` and `schema_comparer.py` depend on the newer formatting module instead of the deprecated `bcolors` helper.

- **[dependency] Q:** What import/dependency change was made in `chispa/formatting/__init__.py`, and what does it expose?
  **A:** `chispa/formatting/__init__.py` now imports `blue` from `chispa.formatting.format_string` and adds `blue` to `__all__`, so callers can import `blue` directly from `chispa.formatting`.

- **[scope] Q:** Which files were touched in this commit, and what kind of change was made in each one?
  **A:** `CONTRIBUTING.md` was added as a new contributor/release guide; `README.md` was updated to use `raw.githubusercontent.com` image links; `chispa/bcolors.py` was marked with deprecation warnings; `chispa/column_comparer.py` and `chispa/schema_comparer.py` were switched to `chispa.formatting.blue`; `chispa/formatting/__init__.py` re-exported `blue`; and `chispa/formatting/format_string.py` added the new `blue` helper.

- **[usage] Q:** How would a developer use the newly re-exported formatting helper after this change?
  **A:** A developer can import it from the package namespace with `from chispa.formatting import blue` and then call `blue("text")` to wrap the string in the light-blue/light-red ANSI color sequence.

- **[edge_case] Q:** How is `None` or non-string input handled by the new formatting helper and the updated call sites?
  **A:** The new `blue(string: str)` helper in `chispa.formatting.format_string` concatenates the input directly with color codes, while the updated call sites in `column_comparer.py` and `schema_comparer.py` explicitly pass `str(...)` values into `blue`, which ensures the helper receives a string representation even for schema field objects.

- **[risk] Q:** What backward-compatibility or regression risk does this change introduce around the old color helpers?
  **A:** `chispa.bcolors.bcolors`, `blue`, and `underline_text` now emit `DeprecationWarning`s, and the comparison code no longer imports `blue` from `chispa.bcolors`. Existing code that depended on the old module path may still work, but it is now explicitly deprecated and may need to switch to `chispa.formatting.blue`.

### MrPowers/chispa @ commit #57 `202ab2aa82` (in_repo_split=test)

- **[intent] Q:** What is the main purpose of this change in chispa?
  **A:** This change adds support for comparing two Spark DataFrames while ignoring one or more named columns, so callers can exclude columns like `clean_name` from equality checks in `assert_df_equality` and `assert_approx_df_equality`.

- **[new_api] Q:** What new API parameter was introduced for DataFrame equality checks, and what does it do?
  **A:** A new optional parameter named `ignore_columns: list[str] | None = None` was added to `assert_df_equality`, `assert_approx_df_equality`, and the `Chispa.assert_df_equality` method. When provided, it causes those columns to be dropped from both DataFrames before schema and row comparison.

- **[signature_change] Q:** How did the function signatures change for the equality assertion helpers?
  **A:** `assert_df_equality` now accepts `ignore_columns: list[str] | None = None` before the `formats` argument, and `assert_approx_df_equality` also now accepts `ignore_columns: list[str] | None = None` before `formats`. The `Chispa.assert_df_equality` method in `chispa/__init__.py` was updated with the same new optional parameter and forwards it to `assert_df_equality`.

- **[behavior_change] Q:** How does runtime behavior differ after this change when `ignore_columns` is supplied?
  **A:** If `ignore_columns` is non-empty, both DataFrames are transformed with `df.drop(*ignore_columns)` before any schema or row equality checks run. This means the listed columns are excluded from the comparison, so differences in those columns no longer cause `assert_df_equality` or `assert_approx_df_equality` to fail.

- **[refactor] Q:** Why was the equality-check pipeline restructured in `dataframe_comparer.py`?
  **A:** The existing transform pipeline was extended to include an extra step for dropping ignored columns, using the same `transforms` list and `reduce`-based application pattern already used for column ordering and row ordering. This keeps the comparison flow consistent while adding column exclusion as another preprocessing step.

- **[scope] Q:** Which files were touched, and what kind of change was made in each one?
  **A:** Three files were modified: `README.md` adds documentation and an example for ignoring specific columns; `chispa/dataframe_comparer.py` adds the `ignore_columns` parameter and drops those columns before schema/row comparison in both equality functions; and `chispa/__init__.py` updates the `Chispa.assert_df_equality` wrapper to accept and pass through `ignore_columns`.

- **[usage] Q:** How would a developer use the new column-ignoring behavior when comparing two DataFrames?
  **A:** A developer would call `assert_df_equality(df1, df2, ignore_columns=["clean_name"])` to compare the two DataFrames while excluding the `clean_name` column from the equality check. The same `ignore_columns` argument can be passed through the `Chispa.assert_df_equality` method as well.

### MrPowers/chispa @ commit #60 `3d7915663e` (in_repo_split=test)

- **[intent] Q:** What is the main purpose of this commit across the package API and schema-comparison code?
  **A:** The change standardizes internal and public type handling by introducing shared enum constants for output and Spark type names, while also cleaning up imports and exported symbols in `chispa/__init__.py` and related modules.

- **[new_api] Q:** What new definitions are introduced in `chispa/common_enums.py`, and what values do they provide?
  **A:** The new `chispa/common_enums.py` module defines two string enums: `OutputFormat` with members `TABLE = "table"` and `TREE = "tree"`, and `TypeName` with members `ARRAY = "array"` and `STRUCT = "struct"`.

- **[signature_change] Q:** How did the `print_schema_diff` function signature change in `chispa/schema_comparer.py`?
  **A:** `print_schema_diff` now annotates its `output_format` parameter as `OutputFormat` instead of `str`, and its default value changed from the string `"table"` to `OutputFormat.TABLE`.

- **[behavior_change] Q:** How does `print_schema_diff` behave differently at runtime after this change when an invalid output format is passed?
  **A:** The function still raises `ValueError`, but the error message now lists `OutputFormat.__members__` instead of the hardcoded list `['table', 'tree']`, and valid comparisons are done against the enum values `OutputFormat.TABLE` and `OutputFormat.TREE`.

- **[refactor] Q:** Which repeated string literals were replaced with named constants in schema comparison logic, and why is that a refactor?
  **A:** String literals like `"struct"` and `"array"` were replaced with `TypeName.STRUCT` and `TypeName.ARRAY` in `schema_comparer.py`, which centralizes Spark type-name checks and avoids repeating raw strings.

- **[dependency] Q:** What import changes were made in the package modules, and what do they indicate about dependency handling?
  **A:** `chispa/__init__.py`, `chispa/dataframe_comparer.py`, and `chispa/rows_comparer.py` switched from `typing.Callable` to `collections.abc.Callable`, and `chispa/__init__.py` removed all the PySpark path-bootstrapping logic that tried `findspark` and `SPARK_HOME`.

- **[scope] Q:** Which files were touched in this commit, and what kind of changes were made in each one?
  **A:** `chispa/__init__.py` removed Spark-path setup and reordered exports; `chispa/common_enums.py` was added with new enums; `chispa/dataframe_comparer.py` and `chispa/rows_comparer.py` changed `Callable` imports; `chispa/formatting/__init__.py` reordered `__all__`; and `chispa/schema_comparer.py` adopted the new enums for output format and Spark type-name checks.

- **[usage] Q:** How would a developer now call `print_schema_diff` to request tree output?
  **A:** They would pass `OutputFormat.TREE` as the `output_format` argument, for example: `print_schema_diff(s1, s2, ignore_nullable, ignore_metadata, output_format=OutputFormat.TREE)`.

- **[edge_case] Q:** What default or edge-case handling changed for schema diff output formats?
  **A:** The default output format is now the enum value `OutputFormat.TABLE`, and the function only accepts the two enum members `OutputFormat.TABLE` and `OutputFormat.TREE`; any other value triggers a `ValueError`.

- **[risk] Q:** What backward-compatibility risk does this commit introduce for callers of schema diff APIs?
  **A:** Callers that previously passed raw strings like `"table"` or `"tree"` to `print_schema_diff` may be affected if they are not compatible with the new `OutputFormat` enum-based annotation and comparisons, even though the underlying enum values still stringify to those same words.

