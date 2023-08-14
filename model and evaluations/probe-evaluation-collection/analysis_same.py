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

def analyse_results(probetype, pred_dir, res_dir):
    by_answer_type = []

    for filename in os.listdir(pred_dir):
        if probetype in filename:
            answer_data = pd.read_csv(os.path.join(pred_dir, filename))
            by_answer_type.append(get_performance_by_answer_type(answer_data, probetype))

    res_by_answer_type = pd.concat(by_answer_type)
    res_by_answer_type.to_csv(str(os.path.join(res_dir, probetype + "_res_by_answer_type.csv")), index = False)

def main():
    args = get_args()
    #os.makedirs(args.res_dir, exist_ok=True)
    analyse_results("probeSAME", args.pred_dir, args.res_dir)

if __name__ == "__main__":
    main()
