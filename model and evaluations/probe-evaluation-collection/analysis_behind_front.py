import os
import pandas as pd
import json
import math
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="../../data/preds/2022-02-01_epochprobes_e25_4mac_newh5")
    parser.add_argument("--res_dir",default="../../data/results/2022-02-01_epochprobes_e25_4mac_newh5")
    parser.add_argument("--scene_file",default="../../data/CLEVR_v1/scenes/CLEVR_val_scenes.json")
    params = parser.parse_args()
    return params


def helper_two_object_dist(objects, params):
    o1_coords = o2_coords = dist = None
    for i, object in enumerate(objects):
        if not o1_coords is None and not o2_coords is None:
            break
        elif params[0] in (object['size'], object['color'], object['material'], "") and params[1] in (object['shape'], 'thing'):
            o1_coords = object['3d_coords']
        elif params[2] in (object['size'], object['color'], object['material'], "") and params[3] in (object['shape'], 'thing'):
            o2_coords = object['3d_coords']
    if not o1_coords is None and not o2_coords is None:
        dist = math.sqrt((o1_coords[0] - o2_coords[0])**2 + (o1_coords[1] - o2_coords[1])**2 + (o1_coords[2] - o2_coords[2])**2)
    return dist

def get_dist_from_scene(item, probetype):
    image_id = item["image_index"]
    question = item["question"]
    objects = g_scenes_dict[image_id]
    params = []
    if probetype == "probeBEHIND":
        # template : "Is the [P1] [S1] behind the [P2] [S2]?"
        question = question.replace('Is the ', '')
        question = question.replace('behind the ', '')
        params = question[:-1].split(' ')
    else:
        # template : "Is the [P1] [S1] in front of the [P2] [S2]?"
        question = question.replace('Is the ', '')
        question = question.replace('in front of the ', '')
        params = question[:-1].split(' ')
    dist = helper_two_object_dist(objects, params)
    item["dist"] = dist
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

def get_performance_by_dist(answer_data, probetype):
    with_dist = answer_data.apply(lambda x: get_dist_from_scene(x, probetype), axis=1)

    dist_res = with_dist.drop(['question', 'image_index', 'index'], axis=1)
    dist_res['probetype'] = probetype

    dist_round = dist_res.round({'dist': 0})
    dist_round = dist_round.astype({'dist': 'int32'})
    dist_round_res = pd.DataFrame(dist_round.value_counts(['probetype', 'epoch', 'batchNum', 'answer', 'prediction', 'dist']))
    dist_round_res = dist_round_res.reset_index()
    dist_round_res.columns = ['probetype', 'epoch', 'batchNum', 'answer', 'prediction', 'dist', 'counts']

    return dist_res, dist_round_res

def analyse_results(probetype, pred_dir, res_dir):
    by_answer_type = []
    by_dist = []
    by_round_dist = []

    for filename in os.listdir(pred_dir):
        if probetype in filename:
            answer_data = pd.read_csv(os.path.join(pred_dir, filename))
            by_answer_type.append(get_performance_by_answer_type(answer_data, probetype))
            dist_res, dist_round_res = get_performance_by_dist(answer_data, probetype)
            by_dist.append(dist_res)
            by_round_dist.append(dist_round_res)

    res_by_answer_type = pd.concat(by_answer_type)
    res_by_answer_type.to_csv(str(os.path.join(res_dir, probetype + "_res_by_answer_type.csv")), index = False)

    res_by_dist = pd.concat(by_dist)
    res_by_dist.to_csv(str(os.path.join(res_dir, probetype + "_res_by_dist.csv")), index = False)

    res_by_round_dist = pd.concat(by_round_dist)
    res_by_round_dist.to_csv(str(os.path.join(res_dir, probetype + "_res_by_round_dist.csv")), index = False)

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
    analyse_results("probeBEHIND", args.pred_dir, args.res_dir)
    analyse_results("probeFRONT", args.pred_dir, args.res_dir)

if __name__ == "__main__":
    main()
