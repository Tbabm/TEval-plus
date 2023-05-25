# encoding=utf-8

"""
Copy and modified from https://github.com/microsoft/toga:
"""

import argparse, csv, os, random
import pandas as pd
import subprocess as sp
import numpy as np

import model.exception_data as exception_data
import model.assertion_data as assertion_data
import model.ranking as ranking

pd.options.mode.chained_assignment = None

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def main():
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_data')
    parser.add_argument('metadata')
    parser.add_argument('--dry', action="store_true")
    args = parser.parse_args()

    dry_run = args.dry
    print(f"dry run: {dry_run}")
    base_dir = os.path.dirname(args.input_data)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    assert base_dir == os.path.dirname(args.metadata)

    fm_test_pairs = pd.read_csv(args.input_data).fillna('')
    metadata = pd.read_csv(args.metadata).fillna('')

    metadata['id'] = metadata.project + metadata.bug_num.astype(str) + metadata.test_name

    methods, tests, docstrings = fm_test_pairs.focal_method, fm_test_pairs.test_prefix, fm_test_pairs.docstring

    # EXCEPT INPUTS
    print('preparing exception model inputs')
    normalized_tests, kept_methods, labels, idxs = exception_data.get_model_inputs(tests, zip(methods, docstrings))

    except_data = list(zip(normalized_tests, kept_methods, labels, idxs))
    except_input_file = os.path.join(base_dir, 'except_model_inputs.csv')
    if not dry_run:
        with open(except_input_file, "w") as f:
            w = csv.writer(f)
            w.writerow(["label", "test", "fm", "docstring", "idx"])
            for test, (method, docstring), label, idx in except_data:
                w.writerow([label, test, method, docstring, idx])
        res = sp.run(f'bash ./model/exceptions/run_eval.sh {except_input_file}'.split(), env=os.environ.copy(),
                     check=True)

    except_pred_file = os.path.join(base_dir, "preds", 'exception_preds.csv')
    results = pd.read_csv(except_pred_file, index_col=False)

    exception_results = results
    exception_idxs = idxs

    except_preds = [0] * len(metadata)
    except_logit_0 = [0.0] * len(metadata)
    except_logit_1 = [0.0] * len(metadata)
    for idx, result in zip(idxs, results.itertuples()):
        # metadata.except_pred[idx] = result.pred_lbl
        except_preds[idx] = result.pred_lbl
        except_logit_0[idx] = result.logit_0
        except_logit_1[idx] = result.logit_1
    metadata['except_pred'] = except_preds
    metadata['except_logit_0'] = except_logit_0
    metadata['except_logit_1'] = except_logit_1

    # ASSERT INPUTS
    print('preparing assertion model inputs')
    vocab = np.load('data/evo_vocab.npy', allow_pickle=True).item()

    method_test_assert_data, idxs = assertion_data.get_model_inputs(tests, methods, vocab)

    assert_inputs_df = pd.DataFrame(method_test_assert_data, columns=["label", "fm", "test", "assertion"])
    assert_inputs_df['idx'] = idxs
    assert_input_file = os.path.join(base_dir, 'assert_model_inputs.csv')
    if not dry_run:
        assert_inputs_df.to_csv(assert_input_file)

        # ASSERT MODEL
        res = sp.run(f'bash ./model/assertions/run_eval.sh {assert_input_file}'.split(), env=os.environ.copy())

    model_preds = []
    assert_pred_file = os.path.join(base_dir, "preds", "assertion_preds.csv")
    with open(assert_pred_file) as f:
        reader = csv.reader(f)
        for row in reader:
            model_preds += [row]

    assertion_results = ranking.rank_assertions(model_preds[1:], idxs)

    assert_preds = [''] * len(metadata)
    assert_logit_0 = [0.0] * len(metadata)
    assert_logit_1 = [0.0] * len(metadata)
    for assertion_result in assertion_results:
        idx, pred_assert, trunc, logit_0, logit_1 = assertion_result
        assert_preds[idx] = pred_assert
        assert_logit_0[idx] = logit_0
        assert_logit_1[idx] = logit_1

    metadata['assert_pred'] = assert_preds
    metadata['assert_logit_0'] = assert_logit_0
    metadata['assert_logit_1'] = assert_logit_1

    # write oracle predictions
    pred_file = os.path.join(base_dir, 'oracle_preds.csv')

    with open(pred_file, 'w') as f:
        w = csv.writer(f)
        w.writerow(
            'project,bug_num,test_name,test_prefix,except_pred,except_logit_0,except_logit_1,assert_pred,assert_logit_0,assert_logit_1'.split(
                ','))
        for orig_test, meta in zip(tests, metadata.itertuples()):
            test_prefix = orig_test
            except_pred = meta.except_pred
            assert_pred = meta.assert_pred
            if except_pred:
                assert_pred = ''
            bug_num = meta.bug_num
            project = meta.project
            test_name = meta.test_name
            except_logit_0 = meta.except_logit_0
            except_logit_1 = meta.except_logit_1
            assert_logit_0 = meta.assert_logit_0
            assert_logit_1 = meta.assert_logit_1

            w.writerow(
                [project, bug_num, test_name, test_prefix, except_pred, except_logit_0, except_logit_1, assert_pred,
                 assert_logit_0, assert_logit_1])

    print(f'wrote oracle predictions to {pred_file}')


if __name__ == '__main__':
    main()
