import os
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="../../data/preds/2022-02-01_epochprobes_e25_4mac_newh5")
    parser.add_argument("--res_dir",default="../../data/results/2022-02-01_epochprobes_e25_4mac_newh5")
    params = parser.parse_args()
    return params

def get_performance_by_answer_type(answer_data, probetype):
    epoch = answer_data['epoch'][0]
    batchNum = answer_data['batchNum'][0]
    yes_data = answer_data.loc[answer_data['answer'] == 'yes']
    no_data = answer_data.loc[answer_data['answer'] == 'no']

    res = pd.DataFrame([[probetype, epoch, batchNum, 'yes', len(yes_data), len(yes_data.loc[yes_data['prediction'] == 'yes'])], [probetype, epoch, batchNum, 'no', len(no_data), len(no_data.loc[no_data['prediction'] == 'no'])]],
     columns=['probetype', 'epoch', 'batchNum', 'answer', 'n_total', 'n_correct'])
    res['prop_correct'] = res.n_correct/res.n_total

    return res


def get_or_inclusive_exclusive_results(answer_data, probetype):
    epoch = answer_data['epoch'][0]
    batchNum = answer_data['batchNum'][0]
    ambiguous_data = answer_data.loc[answer_data['answer'] == 'yes (inclusive) / no (exclusive)']

    res = pd.DataFrame([[probetype, epoch, batchNum, 'yes (inclusive) / no (exclusive)', len(ambiguous_data), len(ambiguous_data.loc[ambiguous_data['prediction'] == 'yes']), len(ambiguous_data.loc[ambiguous_data['prediction'] == 'no'])]],
     columns=['probetype', 'epoch', 'batchNum', 'answer', 'n_total', 'n_inclusive', 'n_exclusive'])
    res['prop_inclusive'] = res.n_inclusive/res.n_total
    res['prop_exclusive'] = res.n_exclusive/res.n_total

    return res


def analyse_results(probetype, pred_dir, res_dir):
    by_answer_type = []
    or_inclusive_exclusive = []

    for filename in os.listdir(pred_dir):
        if probetype in filename:
            answer_data = pd.read_csv(os.path.join(pred_dir, filename))
            by_answer_type.append(get_performance_by_answer_type(answer_data, probetype))
            if probetype == "probeOR" or probetype == "probeOR2":
                or_inclusive_exclusive.append(get_or_inclusive_exclusive_results(answer_data, probetype))

    res_by_answer_type = pd.concat(by_answer_type)
    res_by_answer_type.to_csv(str(os.path.join(res_dir, probetype + "_res_by_answer_type.csv")), index=False)

    if len(or_inclusive_exclusive) > 0 :
        res_or_inclusive_exclusive = pd.concat(or_inclusive_exclusive)
        res_or_inclusive_exclusive.to_csv(str(os.path.join(res_dir, probetype + "_res_or_inclusive_exclusive.csv")), index=False)

def main():
    args = get_args()
    #os.makedirs(args.res_dir, exist_ok=True)
    #analyse_results("probeAND", args.pred_dir, args.res_dir)
    #analyse_results("probeOR", args.pred_dir, args.res_dir)
    analyse_results("probeAND2", args.pred_dir, args.res_dir)
    analyse_results("probeOR2", args.pred_dir, args.res_dir)


if __name__ == "__main__":
    main()
