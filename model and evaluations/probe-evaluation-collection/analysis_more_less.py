import os
import pandas as pd
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="../../data/preds/2022-02-01_epochprobes_e25_4mac_newh5")
    parser.add_argument("--res_dir",default="../../data/results/2022-02-01_epochprobes_e25_4mac_newh5")
    parser.add_argument("--scene_file",default="../../data/CLEVR_v1/scenes/CLEVR_val_scenes.json")
    params = parser.parse_args()
    return params


def helper_two_object_counts(objects, params):
    o1_count = o2_count = 0
    for i, object in enumerate(objects):
        if params[0] in (object['size'], object['color'], object['material'], "") and params[1] in (object['shape'], 'thing'):
            o1_count += 1
        elif params[2] in (object['size'], object['color'], object['material'], "") and params[3] in (object['shape'], 'thing'):
            o2_count += 1
    return o1_count, o2_count

def get_num_from_scene(item, probetype):
    image_id = item["image_index"]
    question = item["question"]
    objects = g_scenes_dict[image_id]
    params = []
    if probetype == "probeMORE":
        # template : "Are there more of the [P1] [S1]s than the [P2] [S2]s?"
        question = question.replace('Are there more of the ', '')
        question = question.replace('than the ', '')
        question = question.replace('s ', ' ')
        params = question[:-2].split(' ')
    else:
        # template : "Are there fewer of the [P1] [S1] than the [P2] [S2]?"
        question = question.replace('Are there fewer of the ', '')
        question = question.replace('than the ', '')
        question = question.replace('s ', ' ')
        params = question[:-2].split(' ')
    o1_count, o2_count = helper_two_object_counts(objects, params)
    item["obj1_count"] = o1_count
    item["obj2_count"] = o2_count
    return(item)



def get_performance_by_answer_type(answer_data, probetype):
    epoch = answer_data['epoch'][0]
    batchNum = answer_data['batchNum'][0]
    yes_data = answer_data.loc[answer_data['answer'] == 'yes']
    no_data = answer_data.loc[answer_data['answer'] == 'no']

    res = pd.DataFrame([[probetype, epoch, batchNum, 'yes', len(yes_data), len(yes_data.loc[yes_data['prediction'] == 'yes'])], [probetype, epoch, batchNum, 'no', len(no_data), len(no_data.loc[no_data['prediction'] == 'no'])]],
     columns=['probetype', 'epoch', 'batchNum', 'answer', 'n_total', 'n_correct'])
    res['prop_correct'] = res.n_correct/res.n_total

    return res

def get_performance_by_num_diff(answer_data, probetype):
    epoch = answer_data['epoch'][0]
    batchNum = answer_data['batchNum'][0]

    with_counts = answer_data.apply(lambda x: get_num_from_scene(x, probetype), axis=1)
    with_counts['num_diff'] = with_counts.obj1_count - with_counts.obj2_count
    with_counts['num_sum'] = with_counts.obj1_count + with_counts.obj2_count

    all_res = with_counts.drop(['question', 'image_index', 'index', 'obj1_count', 'obj2_count', 'epoch', 'batchNum'], axis=1)
    diff_res =  pd.DataFrame(all_res.drop('num_sum', axis=1).value_counts(['answer', 'prediction', 'num_diff']))
    diff_res = diff_res.reset_index()
    diff_res.columns = ['answer', 'prediction', 'num_diff', 'counts']
    diff_res = diff_res.assign(**{'probetype': probetype, 'epoch': epoch, 'batchNum': batchNum})

    sum_res =  pd.DataFrame(all_res.drop('num_diff', axis=1).value_counts(['answer', 'prediction', 'num_sum']))
    sum_res = sum_res.reset_index()
    sum_res.columns = ['answer', 'prediction', 'num_sum', 'counts']
    sum_res = sum_res.assign(**{'probetype': probetype, 'epoch': epoch, 'batchNum': batchNum})

    return diff_res, sum_res

def analyse_results(probetype, pred_dir, res_dir):
    by_answer_type = []
    by_num_diff = []
    by_num_sum = []

    for filename in os.listdir(pred_dir):
        if probetype in filename:
            answer_data = pd.read_csv(os.path.join(pred_dir, filename))
            by_answer_type.append(get_performance_by_answer_type(answer_data, probetype))
            diff_res, sum_res = get_performance_by_num_diff(answer_data, probetype)
            by_num_diff.append(diff_res)
            by_num_sum.append(sum_res)

    res_by_answer_type = pd.concat(by_answer_type)
    res_by_answer_type.to_csv(str(os.path.join(res_dir, probetype + "_res_by_answer_type.csv")), index = False)

    res_by_num_diff = pd.concat(by_num_diff)
    res_by_num_diff.to_csv(str(os.path.join(res_dir, probetype + "_res_by_num_diff.csv")), index = False)

    res_by_num_sum = pd.concat(by_num_sum)
    res_by_num_sum.to_csv(str(os.path.join(res_dir, probetype + "_res_by_num_sum.csv")), index = False)

g_scenes_dict = dict()

def main():
    args = get_args()
    global g_scenes_dict

    # Note this will hopefully be replaced by the test image scenes once the authors get back to us
    with open(args.scene_file) as image_info:
        scenes = json.load(image_info)
    for scene in scenes['scenes']:
        g_scenes_dict[scene["image_index"]] = scene["objects"]

    #os.makedirs(args.res_dir, exist_ok=True)
    analyse_results("probeMORE", args.pred_dir, args.res_dir)
    analyse_results("probeLESS", args.pred_dir, args.res_dir)

if __name__ == "__main__":
    main()
