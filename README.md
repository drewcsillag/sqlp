# sqlp

Sqlp is an alternate sqlite prompt that's meant for ad-hoc reporting and aggregation. It's designed for cases where you have csv datasets, logs, json entries, etc. that you want to process. I use it when tools like `sed`, `grep`, etc. get cumbersome to work with when doing various forms of analysis.

# Features
It uses readline for editing, unlike sqlite's normal interactive commandline.

## Data Loading
* Load CSV into a newly created table -- assumes first row is a header row -- if you specify a file ending in `.csv` sqlp on the commandline, it'll create a new db file with the same name, but with a `.db` extension and import the csv file into a table named `csv`.
* Load TSV into a newly created table -- assumes first row is a header row
* Explode a table with a json-containing column into a new table with the object attributes as columns. Assumes the column contents are a single JSON object literal.
* Load a log file into a table of: file, data -- the table's rowid will give you ordering.
* Load a file prefixed with `hostname: ` into a table of: hostname, file, data -- the table's rowid will give you ordering.

## Output Formatting
* repr - outputs rows using Python's `repr` function
* jq - can pipe a single output column to jq for processing
* gron - pipe a single output column to jq for processing
* line - returns results like this
```
a = 1
b = 1
c = 1

a = 2
b = 2
c = 2
```
* list - returns result rows with columns delimited by `|` characters
* csv - returns result rows in csv form (without header row)
* column - returns a more visually pleasing tabular form, e.g.
```SQLP> select * from foo;
+---+---+---+
| a | b | c |
+---+---+---+
| 1 | 1 | 1 |
| 2 | 2 | 2 |
+---+---+---+
2 row(s) in set
```
* json - returns result rows as JSON objects with the column names as the object keys.

# See Also
https://litecli.com/ is one that has better interaction capabilities, but doesn't have the same collection of data loading and output tooling. Remember, you don't have to pick just one tool! Mix and match and use each for what they're best for!
