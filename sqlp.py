#!/usr/bin/env python3
import csv
import itertools
import json
import os
import re
import readline
import sqlite3
import string
import sys
import traceback

from ast import literal_eval
from sqlite3 import Connection, Cursor
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union

HISTORY_FILE = os.environ["HOME"] + "/.sqlp_history"
DISPLAY_MODE = ["column", ()]  # type: Sequence[Any]
DICT_TYPE = type({})  # type: Any
LIST_TYPE = type([])  # type: Any

COLUMN_SAFE = string.digits + string.ascii_letters + "_"


def despecial(word: str) -> str:
    if word in ["index"]:
        return "_" + word
    w = []
    for i in word:
        if i in COLUMN_SAFE:
            w.append(i)
        else:
            w.append("_")
    return "".join(w)


def create_table_if_not_exists(table: str, cols: List[str], conn: Connection) -> bool:
    create_stmt = "CREATE TABLE %s (%s)" % (
        table,
        ", ".join(["%s TEXT" % despecial(i) for i in cols]),
    )
    stmt = "SELECT name, sql FROM sqlite_master WHERE name = '%s'" % (table,)
    cur = conn.cursor()
    print("executing " + stmt)
    cur.execute(stmt)
    results = cur.fetchall()
    if len(results) == 0:
        print("table %s doesn't exist yet, creating" % (table,))
    elif create_stmt != results[0][1]:
        print("table %s exists, but with different schema!" % (table,))
        print(">" + results[0][1])
        print("<" + create_stmt)
        return False
    else:
        print("table %s already exists, using existing table" % (table,))
        return True

    cur.execute(create_stmt + ";")
    return True


def get_create_stmt(conn: Connection, table: str) -> str:
    stmt = "SELECT sql from sqlite_master where name = '%s';" % table
    cur = conn.cursor()
    cur.execute(stmt)
    desc = cur.fetchone()[0]
    cur.close()
    return desc


def describe(conn: Connection, table: str) -> None:
    print(get_create_stmt(conn, table))


def show_tables(conn: Connection) -> None:
    stmt = "SELECT tbl_name from sqlite_master where type = 'table';"
    cur = conn.cursor()
    cur.execute(stmt)
    tables = [i[0] for i in cur.fetchall()]
    cur.close()
    for t in tables:
        print(t)


def csv_import(conn: Connection, file: str, table: str) -> None:
    r = csv.reader(open(file))
    cols = r.__next__()
    if not create_table_if_not_exists(table, cols, conn):
        print("Failed creating table")
        return

    stmt = "INSERT INTO %s VALUES (%s)" % (table, ", ".join(["?" for i in cols]))
    cur = conn.cursor()
    cur.executemany(stmt, r)
    conn.commit()
    cur.close()


def tsv_import(conn: Connection, file: str, table: str) -> None:
    r = csv.reader(open(file), delimiter="\t")
    cols = r.__next__()
    if not create_table_if_not_exists(table, cols, conn):
        print("Failed creating table")
        return

    stmt = "INSERT INTO %s VALUES (%s)" % (table, ", ".join(["?" for i in cols]))
    cur = conn.cursor()
    cur.executemany(stmt, r)
    conn.commit()
    cur.close()


def quit(conn: Connection) -> None:
    conn.close()
    sys.exit()


def load_log(conn: Connection, file: str, table: str) -> None:
    if not create_table_if_not_exists(table, ["file", "line"], conn):
        print("Failed creating table")
        return
    stmt = "INSERT INTO %s VALUES (?, ?)" % table
    cur = conn.cursor()
    fileRepeat = itertools.repeat(file)
    o = open(file)
    lines = o.readlines()
    llines = [line.strip() for line in lines]
    print("Inserting %d rows" % len(llines))
    cur.executemany(stmt, zip(fileRepeat, llines))
    conn.commit()


def gsh_line_split(line: str) -> List[str]:
    try:
        host, res = line.split(": ", 1)
    except:
        return [line.strip(), "NULL"]
    res = res.strip()
    return [host, res]


def load_gsh_log(conn: Connection, file: str, table: str) -> None:
    if not create_table_if_not_exists(table, ["file", "host", "line"], conn):
        print("Failed creating table")
        return
    stmt = "INSERT INTO %s VALUES (?, ?, ?)" % table
    cur = conn.cursor()
    o = open(file)
    lines = o.readlines()
    llines = [tuple([file] + gsh_line_split(line)) for line in lines]
    flines = [l for l in llines if l[1]]
    cur.executemany(stmt, flines)


def _js_keys(line: str) -> str:
    return repr(set(json.loads(line).keys()))


def explode_json(conn: Connection, old_table: str, column: str, new_table: str) -> None:
    keys = set()  # type: Set[str]
    stmt = "SELECT _js_keys(%s) from %s where jsvalid(%s)" % (column, old_table, column)
    cur = conn.cursor()
    cur.execute(stmt)

    while True:
        row = cur.fetchone()
        if row is None:
            break
        s = literal_eval(row[0])
        keys = keys.union(s)

    keys_list = list(keys)
    keys_list.sort()
    # possibly create if not exists
    stmt = "CREATE TABLE %s (src_rowid TEXT, %s);" % (
        new_table,
        ", ".join(["`%s` TEXT" % despecial(k) for k in keys_list]),
    )
    print("STMT: " + stmt)
    cur.execute(stmt)

    extracts = ["jsextract('[\"%s\"]', %s)" % (key, column) for key in keys_list]
    stmt = """INSERT INTO %s SELECT rowid, %s FROM %s where jsvalid(%s);""" % (
        new_table,
        ", ".join(extracts),
        old_table,
        column,
    )
    # print("STMT: " + stmt)

    cur.execute(stmt)
    conn.commit()


def cursor_iter(cur: Cursor, fn: Callable[[List[str]], Any]) -> None:
    while True:
        row = cur.fetchone()
        if row is None:
            break
        fn(row)


def safe_len(r: Any) -> int:
    if r is None:
        return 4
    if type(r) != type(""):
        r = str(r)

    return len(r)


def sub_row_None_to_NULL(row: List[str]) -> Tuple[Any, ...]:
    r = []
    for i in row:
        if i is None:
            r.append("NULL")
        else:
            r.append(i)
    return tuple(r)


def display_column_results(cur: Cursor) -> None:
    results = cur.fetchall()
    numCols = len(cur.description)
    maxes = [0] * numCols

    for row in results:
        lens = [safe_len(i) for i in row]
        for i in range(len(maxes)):
            maxes[i] = max(maxes[i], lens[i])
    lens = [len(i[0]) for i in cur.description]
    for i in range(len(maxes)):
        maxes[i] = max(maxes[i], lens[i])

    format = "|" + "|".join([" %-0" + str(i) + "s " for i in maxes]) + "|"

    horizLine = "+" + ("+".join(["-" * (i + 2) for i in maxes])) + "+"
    print(horizLine)
    print(format % tuple([i[0] for i in cur.description]))
    print(horizLine)

    for row in results:
        print(format % sub_row_None_to_NULL(row))

    print(horizLine)
    print("%d row(s) in set" % len(results))


def display_gron_results(cur: Cursor) -> None:
    if len(cur.description) == 1:
        if len(DISPLAY_MODE[1]) == 0:
            cmd = "gron"
        else:
            cmd = "gron " + " ".join(DISPLAY_MODE[1])

        p = os.popen(cmd, "w")
        cursor_iter(cur, lambda row: p.write(row[0] + "\n"))
        p.close()
    else:
        print("cannot use .mode gron with multicolumn results")

def display_jq_results(cur: Cursor) -> None:
    if len(cur.description) == 1:
        if len(DISPLAY_MODE[1]) == 0:
            cmd = "jq ."
        else:
            cmd = "jq " + " ".join(DISPLAY_MODE[1])

        p = os.popen(cmd, "w")
        cursor_iter(cur, lambda row: p.write(row[0] + "\n"))
        p.close()
    else:
        print("cannot use .mode jq with multicolumn results")


def display_line_results(cur: Cursor) -> None:
    maxc = max([len(i[0]) for i in cur.description])
    fmtString = "%0" + str(maxc) + "s = %s"
    cols = [i[0] for i in cur.description]

    def line_iter(row: List[str]) -> None:
        print("")
        for n, v in zip(cols, sub_row_None_to_NULL(row)):
            print(fmtString % (n, v))

    cursor_iter(cur, line_iter)


def display_json_results(cur: Cursor) -> None:
    cols = [i[0] for i in cur.description]

    def line_iter(row: List[str]) -> None:
        m = {}
        for n, v in zip(cols, row):
            m[n] = v
        print(json.dumps(m))

    cursor_iter(cur, line_iter)


def display_results(cur: Cursor) -> None:
    if DISPLAY_MODE[0] == "repr":
        cursor_iter(cur, lambda row: print(row))
 
    elif DISPLAY_MODE[0] == "gron":
        display_gron_results(cur)

    elif DISPLAY_MODE[0] == "jq":
        display_jq_results(cur)

    elif DISPLAY_MODE[0] == "line":
        display_line_results(cur)

    elif DISPLAY_MODE[0] == "list":
        cursor_iter(
            cur,
            lambda row: print("|".join([str(i) for i in sub_row_None_to_NULL(row)])),
        )

    elif DISPLAY_MODE[0] == "csv":
        out = csv.writer(sys.stdout)
        cursor_iter(cur, lambda row: out.writerow(row))

    elif DISPLAY_MODE[0] == "column":
        display_column_results(cur)

    elif DISPLAY_MODE[0] == "json":
        display_json_results(cur)


def get_cols_for_table(conn: Connection, table: str) -> List[str]:
    # Admittedly hokey, but should be good enough
    create_stmt = get_create_stmt(conn, table)
    open_paren = create_stmt.index("(")
    close_paren = create_stmt.index(")")
    return [i.split()[0] for i in create_stmt[open_paren + 1 : close_paren].split(",")]


def extract_by_col_val(
    conn: Connection, table: str, column: str, value: str, newtable: str
) -> None:
    cols = get_cols_for_table(conn, table)
    if column not in cols:
        raise ValueError(
            "column %s is not a column in table %s, valid columns are %r"
            % (column, table, cols)
        )
    othercols = [i for i in cols if i != column]
    query = (
        "SELECT "
        + ",".join(["count(%s)" % i for i in othercols])
        + " FROM %s WHERE %s = '%s'" % (table, column, value)
    )
    cur = conn.cursor()
    cur.execute(query)
    counts = cur.fetchone()
    # filter out zero non-null columns and the src_rowid column as we add our own
    nonzero_cols = [t for t, c in zip(othercols, counts) if c > 0 and t != "src_rowid"]
    create_table_if_not_exists(newtable, ["src_rowid"] + nonzero_cols, conn)
    insert_query = "INSERT INTO %s SELECT rowid, %s FROM %s WHERE %s = '%s'" % (
        newtable,
        ",".join(nonzero_cols),
        table,
        column,
        value,
    )
    cur.execute(insert_query)


def help(conn: Connection) -> None:
    print("")
    print("For the fine manual, see drl/sqlp_manual")
    print("")
    ks = list(commands.keys())
    ks.sort()
    for k in ks:
        print(".%s - %s" % (k, commands[k].doc))

    print("extension functions:")
    print("    REGEXP sqlite3 syntax")
    print("    regex_sub     - pattern replacement value")
    print(
        "    regex_extract - regex value group_no   # group number 0 is the whole match"
    )
    print(
        '    jsextract - path content - extract JSON sub objects from a JSON blob e.g. ["foo"][1]["ba"r"] or .foo[1].bar'
    )
    print(
        "    _js_keys - content -- returns a python representation of the key set of the object - intended for internal use"
    )


def mode(conn: Connection, mode: str, *rest: Sequence[str]) -> None:
    # won't bother: ascii html insert quote tabs tcl
    valid_modes = ["jq", "repr", "line", "list", "csv", "column", "json", "gron"]
    if mode not in valid_modes:
        print("ERROR: Invalid mode, valid kinds are " + ", ".join(valid_modes))
    if mode == "column":
        print(
            "WARNING: column uses more memory because it loads all the results into memory"
        )
    global DISPLAY_MODE
    DISPLAY_MODE = [mode, rest]


def dotopen(conn: Connection, file: str) -> Connection:
    conn.close()
    return openConn(file)


def dothrow(conn: Connection) -> None:
    raise SyntaxError("FOO!")


def doread(in_conn: Connection, file: str) -> None:
    global conn, filename
    orig_conn = conn
    orig_name = filename
    orig_stdin = sys.stdin
    sys.stdin = open(file)
    try:
        run()
    except:
        pass
    conn = orig_conn
    sys.stdin = orig_stdin
    filename = orig_name


class Command(object):
    def __init__(
        self,
        numargs: int,
        fn: Callable[..., Any],
        doc: str = "TODO",
        varargs: bool = False,
    ):
        self.numargs = numargs
        self.varargs = varargs
        self.fn = fn
        self.doc = doc


commands = {
    "quit": Command(0, quit, doc="exit sqlp"),
    # adjust this to support no header cases, and what to do about that
    "csvimport": Command(
        2,
        csv_import,
        doc=".csvimport file table - import csv file into table -- assumes a header names row",
    ),
    "tsvimport": Command(
        2,
        csv_import,
        doc=".tsvimport file table - import tsv file into table -- assumes a header names row",
    ),
    "desc": Command(
        1,
        describe,
        doc=".desc table_name - show the create table command for table_name",
    ),
    "tables": Command(0, show_tables, doc="list tables in current db"),
    "loadlog": Command(2, load_log, doc=".loadlog file table - import file into table"),
    "loadgshlog": Command(
        2,
        load_gsh_log,
        doc=".loadgshlog file table - import file from gsh output (has `host:` prefix) into table",
    ),
    "jsonexplode": Command(
        3,
        explode_json,
        doc=".jsonexplode old_table, column, new_table -- explode old_table.column into new_table",
    ),
    "help": Command(0, help, doc="show this help info"),
    "mode": Command(
        1,
        mode,
        varargs=True,
        doc="change output mode, valid values are jq, repr, line, list, csv, and column",
    ),
    "open": Command(1, dotopen, doc=".open sqlitedbfile"),
    "dothrow": Command(0, dothrow),
    "extractbycolval": Command(
        4,
        extract_by_col_val,
        doc=".extractbycolval table split_column value newtable"
        + " -- create a table named <newtable> from <table> where <column> = <value> including only "
        + "columns where all the rows for that <column>=<value> are non-null. Value is assumed to be"
        + " a string. New table excludes src_rowid, if it is in the source table, and the column you"
        + " split on.",
    ),
    "read": Command(1, doread, doc=".read sql_file -- process commands from sql_file as if they were read at the console"),
}


def do_dot_command(line: str, conn: Connection) -> Connection:
    line = line.strip()
    cmd_and_args = [l.strip() for l in line.strip().split(" ")]
    cmd = cmd_and_args[0]
    if cmd[1:] not in commands:
        print("unknown dot command %s" % cmd)
        return conn

    command = commands[cmd[1:]]
    numargs = len(cmd_and_args) - 1
    if not (
        numargs == command.numargs or command.varargs and numargs >= command.numargs
    ):
        print(
            "incorrect number of arguments to %s wanted %d got %d"
            % (cmd, command.numargs, numargs)
        )
        print(command.doc)
        return conn

    args = (conn,) + tuple(cmd_and_args[1:])
    cmd_con = command.fn(*args)
    if cmd_con is not None:
        return cmd_con
    return conn


def regexp(expr: str, item: str) -> bool:
    reg = re.compile(expr)
    return reg.search(item) is not None


def regex_extract(expr: str, item: str, group: int) -> Optional[str]:
    try:
        reg = re.compile("(%s)" % expr)
    except:
        print("ERROR compiling regex %s" % expr)
        raise

    try:
        match = reg.search(item)
    except:
        print("ERROR matching re %s" % expr)
        raise

    if not match:
        return None
    return match.groups()[group]


def regex_sub(pattern: str, repl: str, target: str) -> str:
    return re.sub(pattern, repl, target)


(
    PATH_INIT,
    PATH_OPEN_BRACE,
    PATH_STRING,
    PATH_NUMBER,
    PATH_CLOSE_BRACKET,
    PATH_DOTTED_STRING,
) = (
    "INIT",
    "open square bracket",
    "a quoted string",
    "numeric",
    "brackets",
    "dotted string",
)


def parse_path(path: str) -> List[Union[str, int]]:
    indices = []  # type: List[Union[str, int]]
    state = PATH_INIT
    idx = 0

    cur_string = []
    cur_number = []  # type: List[str]

    while True:
        if idx >= len(path):
            break

        c = path[idx]
        if state == PATH_INIT:
            if c == "[":
                state = PATH_OPEN_BRACE
            elif c == ".":
                state = PATH_DOTTED_STRING
            else:
                raise SyntaxError("expected [ got " + c)

        elif state == PATH_DOTTED_STRING:
            if c != "." and c != "[":
                cur_string.append(c)
            else:
                indices.append("".join(cur_string))
                cur_string = []
                if c == "[":
                    state = PATH_OPEN_BRACE

        elif state == PATH_OPEN_BRACE:
            if c == '"':
                state = PATH_STRING
            elif c in string.digits:
                state = PATH_NUMBER
                continue  # to avoid incrementing idx
            else:
                raise SyntaxError(
                    "expected either a quoted string or a number in square brackets, got "
                    + c
                )

        elif state == PATH_STRING:
            if c == '"':
                indices.append("".join(cur_string))
                cur_string = []
                state = PATH_CLOSE_BRACKET

            elif c == "\\":
                idx += 1
                cur_string.append(path[idx])
            else:
                cur_string.append(c)

        elif state == PATH_NUMBER:
            if c == "]":
                indices.append(int("".join(cur_number)))
                cur_number = []
                state = PATH_INIT
            elif c in string.digits:
                cur_number.append(c)
            else:
                raise SyntaxError("Expected a digit, got " + c)

        elif state == PATH_CLOSE_BRACKET:
            if c == "]":
                state = PATH_INIT
            else:
                raise SyntaxError("expected close bracket, got " + c)

        idx += 1

    if state == PATH_DOTTED_STRING:
        indices.append("".join(cur_string))
        state = PATH_INIT

    if state != PATH_INIT:
        raise SyntaxError("got end of string while in " + state)

    return indices


jsonexprcache = []  # type: List[Tuple[str, List[Union[str, int]]]]


def parse_path_cached(path: str) -> List[Union[str, int]]:
    global jsonexprcache
    for i in jsonexprcache:
        if i[0] == path:
            return i[1]
    # still here
    parsed = parse_path(path)
    jsonexprcache.insert(0, (path, parsed))
    if len(jsonexprcache) > 50:  # have to pick some limit, right?
        jsonexprcache = jsonexprcache[:15]
    return parsed


def jsvalid(js: str) -> bool:
    try:
        json.loads(js)
        return True
    except:
        return False


def jsextract(path: str, js: str) -> Optional[str]:
    try:
        j = json.loads(js)
        parsed = parse_path_cached(path)
        o = j
        for idx in parsed:
            if type(o) == DICT_TYPE:
                o = o.get(idx)
            elif type(o) == LIST_TYPE:
                o = o[idx]
            if o is None:
                return None
        if type(o) == type(""):
            return o
        if o is None:
            return None
        return json.dumps(o)
    except:
        print(traceback.format_exc())
        return None


def days_to_dhms(value: str) -> str:
    days = float(value)
    r = []
    if days >= 1:
        d = int(days)
        days -= d
        r.append(f"{d}d ")
    hours = days * 24
    if r or hours >= 1:
        h = int(hours)
        hours -= h
        r.append("%02d:" % h)
    mins = hours * 60
    if r or mins >= 1:
        m = int(mins)
        mins -= m
        r.append("%02d:" % m)
    secs = mins * 60
    r.append("%05.2f" % secs)
    return "".join(r)

    
def openConn(file: str) -> Connection:
    sqlite3.enable_callback_tracebacks(True)
    conn = sqlite3.connect(file)
    print("connected to %s" % file)
    # because SQLite doesn't really come with prebaked regex functionality (though it has syntax, you have to plug in)
    conn.create_function("REGEXP", 2, regexp)
    conn.create_function("regex_extract", 3, regex_extract)
    conn.create_function("regex_sub", 3, regex_sub)
    conn.create_function("jsextract", 2, jsextract)
    conn.create_function("jsvalid", 1, jsvalid)
    conn.create_function("_js_keys", 1, _js_keys)
    conn.create_function("days_to_dhms", 1, days_to_dhms)
    return conn


QUOTES = ["`", '"', "'"]


def strip_comments(query: str) -> str:
    """Strips comments and smashes the query into one line"""
    # maybe strip leading whitespace after newline
    outl = []
    in_quotes = None
    in_comment = 0
    for c in query:
        if in_comment == 3:
            if c == "\n":
                in_comment = 0
                outl.append(c)
        elif in_comment == 1:
            if c == "-":
                in_comment = 2
            else:
                outl.append("-")
                outl.append(c)
                in_comment = 0
        elif in_comment == 2:
            if c == " ":
                in_comment = 3
            else:
                outl.append("--")
                outl.append(c)
                in_comment = 0
        elif in_quotes:
            if c == in_quotes:
                in_quotes = None
            outl.append(c)
        elif c in QUOTES:
            in_quotes = c
            outl.append(c)
        elif c == "-":
            in_comment = 1
        elif c == "\n":
            outl.append(" ")
        else:
            outl.append(c)

    return "".join(outl).strip()


def run():
    global filename, conn
    if len(sys.argv) != 2:
        print("usage: sqlp dbfile")
        print(
            "The db file need not previously exist, it will be created if not exists."
        )

    print("Drew's fancy shmancy sqlite prompt")
    print("SQLite version %s" % sqlite3.sqlite_version)
    print('Enter ".help" for usage hints.')

    filename = sys.argv[1]
    if filename[-4:] == '.csv':
        dbname = filename[:-4] + '.db'
        conn = openConn(dbname)
        csv_import(conn, filename, "csv")
    else:
        conn = openConn(filename)

    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        readline.write_history_file(HISTORY_FILE)
        pass  # ok that the history file isn't there yet

    buffer = ""
    linecount = 0
    cont = False
    while True:
        try:
            if cont:
                line = input("....> ")
            else:
                line = input("SQLP> ")

            if not line:
                continue

            if not cont and line[0] == ".":
                conn = do_dot_command(line, conn)
                readline.append_history_file(1, HISTORY_FILE)
                continue

            if not buffer:
                buffer = line
            else:
                buffer += "\n" + line

            linecount += 1

            if sqlite3.complete_statement(buffer):
                # smash multiline queries into one for the history
                for i in range(linecount):
                    readline.remove_history_item(0)
                linecount = 0
                buffer = strip_comments(buffer)
                readline.add_history(buffer)
                readline.append_history_file(1, HISTORY_FILE)
                try:
                    buffer = buffer.strip()
                    cur = conn.cursor()
                    cur.execute(buffer)

                    if buffer.lstrip().upper().startswith("SELECT"):
                        display_results(cur)
                        # cur.fetchall(), cur.description, cur.rowcount)
                    else:
                        print("rowcount: %d" % cur.rowcount)
                    cur.close()
                    conn.commit()
                except sqlite3.Error as e:
                    print("An error occurred: %r" % (e.args[0],))
                buffer = ""
                cont = False
            else:
                cont = True
        except:
            exctype = sys.exc_info()[0]
            if (
                exctype == SystemExit
                or exctype == EOFError
                or exctype == KeyboardInterrupt
            ):
                break
            buffer = ""

            print(traceback.format_exc())
    conn.close()

if __name__ == '__main__':
    run()
