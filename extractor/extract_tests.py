# Copy and modified from https://github.com/microsoft/toga/blob/main/eval/extract_tests.py
import glob
import logging
import os, re, csv, argparse
import javalang
import pandas as pd
import subprocess as sp
from tree_hugger.core import JavaParser
from collections import defaultdict
from copy import copy
from joblib import delayed, Parallel
from tqdm import tqdm

SAMPLE_PROJECTS = ('Chart', 'Cli', 'Csv', 'Gson', 'Lang')
TMP_DIR = "/tmp/extract_tests"

jp = JavaParser("lib/tree_sitter/my-languages.so")

assert_re = re.compile("assert\w*\(.*\)")
path_re = re.compile(r"\S+\/generated\/(\S*)\/evosuite\/([0-9]+)\/(\S*)_ESTest.java")
whitespace_re = re.compile(r'\s+')

test_name_re = re.compile("public void (test[0-9]*)\(\)")
extract_package_re = re.compile(r'package\s+(\S+);')
fail_catch_re = re.compile(r"fail\(.*\).*}\s*catch", re.MULTILINE | re.DOTALL)

errs = defaultdict(int)


def checkout_project(project, bug_num):
    outpath = os.path.join(TMP_DIR, "{}_{}_buggy/".format(project, bug_num))
    if os.path.isdir(outpath): return outpath

    print("checking out {} {} into {}".format(project, bug_num, outpath))
    sp.call(["defects4j", "checkout", "-p", project, "-v", bug_num + "b", "-w", outpath])

    return outpath


def get_active_bugs(d4j_project_dir):
    active_bugs_df = pd.read_csv(d4j_project_dir + '/active-bugs.csv', index_col=0)
    bug_scm_hashes = active_bugs_df[['revision.id.buggy', 'revision.id.fixed']].to_dict()
    return active_bugs_df.index.to_list(), bug_scm_hashes


def get_project_layout(d4j_project_dir):
    project_layout_df = pd.read_csv(d4j_project_dir + '/dir-layout.csv', index_col=0,
                                    names=['src_dir', 'test_dir'])
    return project_layout_df.to_dict()


def extract_focal_class(class_dec):
    return class_dec.name.strip("_ESTest")


def extract_focal_methods(class_dec, tests, all_focal_class_methods):
    focal_class_name = extract_focal_class(class_dec)
    focal_methods = []

    for test_txt in tests:
        focal_method_name = None
        try:
            try:
                tokens = javalang.tokenizer.tokenize(test_txt)
                parser = javalang.parser.Parser(tokens)
                test_obj = parser.parse_member_declaration()
            except Exception as e:
                print('ERROR parsing test:')
                print(test_txt)
                errs['unable_to_parse_test'] += 1

                focal_methods += [('', '')]
                continue

            nodes = [n for n in test_obj]

            fm_names = []
            for p, n in reversed(nodes):
                if isinstance(n, javalang.tree.MethodInvocation):
                    if n.member == 'fail' or n.member == 'verifyException' or \
                            'assert' in n.member:
                        continue
                    focal_method_name = n.member
                    fm_names += [focal_method_name]

                if isinstance(n, javalang.tree.ClassCreator):
                    focal_method_name = n.type.name
                    fm_names += [focal_method_name]

            added = False
            for focal_method_name in fm_names:
                for focal_class_methods in all_focal_class_methods:
                    for (f_method_dec, f_method_text, line_nums, docstring) in focal_class_methods:

                        if f_method_dec.name == focal_method_name:
                            focal_methods += [(f_method_text, docstring)]
                            added = True
                            break

                    if added: break
                if added: break

            if not added:
                focal_methods += [('', '')]
        except Exception as e:
            added = False
            raise e

    return focal_methods


def get_method_txt(lines, start_line):
    """
    lines: lines of file, assume each ends with \n
    start_line: first line of method decl
    """
    method_def = ''
    method_sig = ''
    depth = 0
    method_collected = False
    in_string = False
    escape = False

    line_nums = []
    for i in range(start_line, len(lines)):
        prev_char = ''
        line = lines[i]

        for col, char in enumerate(line):
            next_char = line[col + 1] if col + 1 < len(line) else ''

            # escape
            if escape:
                escape = False
            elif char == '\\':
                escape = True

            # comment
            elif char == '/' and prev_char == '/' and not in_string:
                prev_char = ''
                break

            # single chars
            elif not in_string and prev_char == "'" and next_char == "'":
                pass

            # strings, curlys
            elif char == "\"":
                in_string = not in_string
            elif char == '{' and not in_string:
                depth += 1

                if depth == 1:  # open def, grab signature
                    method_sig = method_def + line[:col].strip() + ';'
            elif char == '}' and not in_string:
                depth -= 1
                if depth == 0:  # end of method def
                    method_def += line[:col + 1]
                    line_nums += [i + 1]
                    method_collected = True
                    break

            prev_char = char
        if method_collected:
            break

        method_def += line
        line_nums += [i + 1]

    return method_sig, method_def, line_nums


def get_class_dec(test_file):
    try:
        with open(test_file) as f:
            class_txt = f.read()

        with open(test_file) as f:
            class_lines = f.readlines()

    except Exception as e:
        print('ERROR READING:', test_file)
        raise e

    try:
        tree = javalang.parse.parse(class_txt)
    except Exception as e:
        print("error parsing", test_file)
        raise e

    class_dec = tree.types[0]

    return class_dec, class_lines


def get_classes_with_inherited(full_class_path, src_path):
    ret_list = []

    while full_class_path:

        try:
            class_dec, class_lines = get_class_dec(full_class_path)
        except Exception as e:
            print('ERROR parsing', full_class_path)
            if ret_list:
                return ret_list
            else:
                raise e

        ret_list += [(class_dec, class_lines)]

        full_class_path = None

        # get import list
        imports = {}
        for line in class_lines:
            if line.strip().startswith('import'):
                imported = line.strip().strip(';').split()[-1]
                import_cls = imported.split('.')[-1]
                imports[import_cls] = imported

        if hasattr(class_dec, 'extends') and class_dec.extends and class_dec.extends.name:
            extend_cls = class_dec.extends.name
            if extend_cls in imports:
                extend_full_cls = imports[extend_cls]
                full_class_path = src_path + '/' + extend_full_cls.replace('.', '/') + '.java'

    return ret_list


def extract_all_methods(class_dec, class_lines):
    methods = []

    for method in class_dec.constructors:
        method_sig, method_def, line_nums = get_method_txt(class_lines, method.position.line - 1)
        if method_def.count("@Test") > 1:
            continue

        methods.append((method, method_def, line_nums, method.documentation))

    for method in class_dec.methods:
        method_sig, method_def, line_nums = get_method_txt(class_lines, method.position.line - 1)
        if method_def.count("@Test") > 1:
            continue

        methods.append((method, method_def, line_nums, method.documentation))

    return methods


def split_test(test_method, line_nums, assert_line_no=None):
    # split by asserts
    split_tests = []
    split_test_line_nums = []

    relevant_lines = []
    relevant_line_nums = []
    for line, line_no in zip(test_method.split('\n'), line_nums):
        if not line.strip():
            continue

        if 'assert' in line:
            if assert_line_no is not None:
                if line_no == assert_line_no:
                    relevant_lines += [line]
                    relevant_line_nums += [line_no]
                    relevant_lines += ['}']
                    split_tests += ['\n'.join(relevant_lines)]
                    split_test_line_nums += [copy(relevant_line_nums)]
                    break

            else:  # no assert_line specified, keep all asserts
                next_test = '\n'.join(relevant_lines + [line, '}'])
                next_test_lines = copy(relevant_line_nums + [line_no])
                split_tests += [next_test]
                split_test_line_nums += [next_test_lines]

        else:  # non assert line
            relevant_lines += [line]
            relevant_line_nums += [line_no]

    split_tests += ['\n'.join(relevant_lines)]
    split_test_line_nums += [relevant_line_nums]

    return split_tests, split_test_line_nums


def get_bug_id(project, bug_num):
    return f"{project}{bug_num}"


def checkout_bug_repos(test_corpus_dir, njobs):
    bug_repos = {
    }
    task_args = []
    for tarball in glob.glob(f"{test_corpus_dir}/generated/*/*/*/*.tar.bz2"):
        items = tarball.split('/')
        project, method, test_id, tarball_name = items[-4:]
        bug_num = tarball_name.split('-')[1][:-1]
        task_args.append((project, bug_num))
    tasks = [delayed(checkout_project)(*task_arg) for task_arg in task_args]
    out_paths = Parallel(n_jobs=njobs, prefer='processes')(tqdm(tasks))
    for (project, bug_num), out_path in zip(task_args, out_paths):
        bug_repos[get_bug_id(project, bug_num)] = out_path
    return bug_repos


def extract_input_and_meta(full_fname, project, bug_num, src_dir, project_dir, full_class_path):
    try:
        class_dec, class_text = get_class_dec(full_fname)
    except Exception as e:
        errs['err_parse_test_file'] += 1
        print("ERROR:couldn't parse test_class", full_fname)
        return [], []

    try:
        src_path = os.path.join(project_dir, src_dir)
        focal_dec_text_pairs = get_classes_with_inherited(full_class_path, src_path)
    except Exception as e:
        errs['err_parse_focal_file'] += 1
        print("ERROR:couldn't parse focal class", project, bug_num, full_class_path)
        return [], []

    package = ''
    for line in class_text:
        if m := extract_package_re.match(line.strip()):
            package = m[1]
            break

    jp.parse_file(full_fname)
    class_test_methods = jp.get_all_method_bodies()

    assert len(class_test_methods) == 1

    class_name, _ = list(class_test_methods.items())[0]

    test_methods = extract_all_methods(class_dec, class_text)
    split_test_methods = []
    split_test_line_nums = []
    for obj, test_method, line_nums, _ in test_methods:
        m2 = test_name_re.search(test_method)
        if not m2:
            errs['test_name_not_matched'] += 1
            continue
        test_name = m2.group(1)
        full_test_name = package + '.' + class_name + '::' + test_name
        full_test_id = project + str(bug_num) + full_test_name

        split_tests, split_test_lines = split_test(test_method, line_nums)

        assert (split_tests)  # should always have at least one

        split_test_methods += split_tests
        split_test_line_nums += split_test_lines

    focal_class_methods = [extract_all_methods(fdec, ftxt) for fdec, ftxt in focal_dec_text_pairs]
    focal_methods = extract_focal_methods(class_dec, split_test_methods, focal_class_methods)

    assert (len(split_test_methods) == len(focal_methods))
    assert (len(split_test_methods) == len(split_test_line_nums))

    cur_meta = []
    cur_input_data = []
    for test_method, focal_method_docstring, test_lines in zip(split_test_methods, focal_methods, split_test_line_nums):
        focal_method, docstring = "", ""
        if focal_method_docstring:
            focal_method, docstring = focal_method_docstring

        assertion = ''
        try:
            m = assert_re.search(test_method)
        except Exception as e:
            print('ERROR cannot regex search test:')
            print(test_method)
            raise e
        if m:
            assertion = m[0]

        m2 = test_name_re.search(test_method)
        test_name = m2.group(1)

        full_test_name = package + '.' + class_name + '::' + test_name
        test_id = project + str(bug_num) + full_test_name

        exception_lbl = bool(fail_catch_re.search(test_method))
        assertion_lbl = assertion

        # get bug metadata
        assertion_bug = -1
        exception_bug = -1
        error = ''

        cur_meta.append(
            (project, bug_num, full_test_name, exception_bug, assertion_bug, exception_lbl, assertion_lbl, error))
        cur_input_data.append((focal_method, test_method, docstring))

    return cur_meta, cur_input_data


def main(test_corpus_dir: str,
         sample_5projects: bool,
         d4j_path: str,
         output_dir: str,
         njobs: int = None):
    input_data = []
    metadata = []

    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    logging.info("Checking out repos")
    bug_repos = checkout_bug_repos(test_corpus_dir, njobs)

    # sanity check
    for bug_id, bug_repo in bug_repos.items():
        prefix, basename = os.path.split(bug_repo)
        if not basename:
            _, basename = os.path.split(prefix)
        # print(basename)
        assert bug_id == "".join(basename.split('_')[:2])

    tasks = []
    for root, dirs, files in os.walk(test_corpus_dir):
        for f in files:
            if not f.endswith("_ESTest.java"): continue
            full_fname = os.path.join(root, f)

            match = path_re.search(full_fname)
            if not match:
                errs['file_name_not_matched'] += 1
                continue
            project = match.group(1)
            bug_num = match.group(2)
            # focal class path
            class_path = match.group(3)

            if sample_5projects and project not in SAMPLE_PROJECTS:
                continue

            d4j_project_dir = d4j_path + '/framework/projects/' + project

            project_layout = get_project_layout(d4j_project_dir)
            _, bug_scm_hashes = get_active_bugs(d4j_project_dir)

            project_dir = bug_repos[get_bug_id(project, bug_num)]

            try:
                bug_hash = bug_scm_hashes['revision.id.buggy'][int(bug_num)]
                src_dir = project_layout['src_dir'][bug_hash]
            except:
                errs['no_d4j_bug_hash'] += 1
                print('ERROR: no bug hash/dir for', bug_num, project_dir)
                # sys.exit(1)
                continue

            full_class_path = os.path.join(project_dir, src_dir, class_path) + ".java"
            if not os.path.exists(full_class_path):
                errs['cannot_find_focal_unit_file'] += 1
                print('ERROR: cannot get file:')
                print(full_class_path)
                continue

            tasks.append(
                delayed(extract_input_and_meta)(full_fname, project, bug_num, src_dir,
                                                project_dir, full_class_path))

    results = Parallel(n_jobs=njobs, prefer='processes')(tqdm(tasks))

    for cur_meta, cur_input in results:
        metadata += cur_meta
        input_data += cur_input

    print('collected inputs:', len(input_data))
    print(f'writing to {output_dir}/inputs.csv and {output_dir}/meta.csv')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + '/inputs.csv', 'w') as f1, open(output_dir + '/meta.csv', 'w') as f2:
        input_w = csv.writer(f1)
        meta_w = csv.writer(f2)

        input_w.writerow(['focal_method', 'test_prefix', 'docstring'])
        meta_w.writerow(
            'project,bug_num,test_name,exception_bug,assertion_bug,exception_lbl,assertion_lbl,assert_err'.split(','))

        for input_pair, meta in zip(input_data, metadata):
            input_w.writerow(input_pair)
            meta_w.writerow(meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_corpus_dir')
    # Since we only have the buggy version in practice, we do not know whether a bug prefix is a bug-reaching test
    # parser.add_argument('--bug_tests_only', action='store_true')
    parser.add_argument('--sample_5projects', action='store_true')
    parser.add_argument('--d4j_path', default='../defects4j/')
    parser.add_argument('-o', '--output_dir', default='.')
    parser.add_argument('-njobs', type=int, default=1)
    args = parser.parse_args()

    njobs = args.njobs
    if njobs == 1:
        njobs = None

    main(args.test_corpus_dir, args.sample_5projects, args.d4j_path, args.output_dir, njobs)
