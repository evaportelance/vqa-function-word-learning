import json
import datetime
import random
from collections import namedtuple
from dataclasses import dataclass, field, asdict
from threading import Lock
from multiprocessing import Pool
from itertools import cycle
from enum import IntEnum

class ProbeType(IntEnum):
    OR = 1
    AND = 2
    MORE = 3
    LESS = 4
    BEHIND = 5
    FRONT = 6
    SAME = 7

@dataclass
class Question:
    split: str
    image_index: int
    image_filename: str
    question: str
    answer: str
    question_family_index: int

@dataclass
class Probe:
    split: str
    version: str
    license: str
    date: str = datetime.datetime.now().strftime("%x")
    questions: list = field(default_factory=list)

    def add_q(self, q):
        self.questions.append(q)


def create_question(text, params):
    question = text
    for p in params:
        #replace the param name with the word
        question = question.replace(p[0], p[2])
    return question

def test_scene_answer(scene, split, params):
    def helper_two_objects(objects):
        Object = namedtuple("Object", ["exist", "count", "index"], defaults=(False, 0, -1))
        o1_exist = o2_exist = False
        o1_count = o2_count = 0
        o1_index = o2_index = -1
        for i, object in enumerate(scene['objects']):
            if params[0][1] in (object['size'], object['color'], object['material'], "") and params[1][1] in (object['shape'], 'thing'):
                o1_exist = True
                o1_count += 1
                o1_index = i
            elif params[2][1] in (object['size'], object['color'], object['material'], "") and params[3][1] in (object['shape'], 'thing'):
                o2_exist = True
                o2_count += 1
                o2_index = i
        return Object(o1_exist, o1_count, o1_index), Object(o2_exist, o2_count, o2_index)
        # return object1_exist, object2_exist, object1_count, object2_count, object1_index, object2_index

    image_index = scene["image_index"]
    objects = scene["objects"]
    answer = None
    if split == ProbeType.OR:
        #OR "answers": [("yes", "yes"), ("no", "yes"), ("no", "no")] consider both the exclusive and inclusive interpretations
        #"Is there a <M> <N> or a <M1> <N1>?"
        #Presuppositions: None
        #Answers:
            # yes inclusive: (<M> <N>), (<M1> <N1>), (<M> <N> and <M1> <N1>)
            # yes exclusive: (<M> <N>), (<M1> <N1>),
            # no inclusive: !(<M> <N> and <M1> <N1>)
            # no exclusive: !(<M> <N> and <M1> <N1>), (<M> <N> and <M1> <N1>)
        o1, o2 = helper_two_objects(objects)

        #Answer conditions
        if o1.exist is not o2.exist:
            answer = "yes"
        elif o1.exist and o2.exist:
            answer = "yes (inclusive) / no (exclusive)"
        else:
            answer = "no"

    elif split == ProbeType.AND:
        #AND "answers" : "yes" "no"
        #"Is there a <M> <N> and a <M1> <N1>?"
        #Presuppositions: None
        #Answers:
            # yes : (<M> <N> and <M1> <N1>)
            # no : !(<M> <N>), !(<M1> <N1>)
        o1, o2 = helper_two_objects(objects)

        #Answer conditions
        answer = "yes" if o1.exist and o2.exist else "no"

    elif split == ProbeType.MORE:
        #MORE "answers" : "yes" "no" consider by relative number of each item
        #"Are there more of the <M> <N>s than the <M1> <N1>s?"
        #Presuppositions: (<M> <N> and <M1> <N1>)
        #Answers:
            # yes : count(<M> <N>) > count(<M1> <N1>)
            # no : count(<M> <N>) <= count(<M1> <N1>)
        o1, o2 = helper_two_objects(objects)

        #Presupposition
        if o1.count and o2.count:
            #Answer conditions
            more = o1.count > o2.count
            answer = "yes" if more else "no"

    elif split == ProbeType.LESS:
        #LESS "answers" : "yes" "no" consider by relative number of each item
        #"Are there fewer of the <M> <N>s than the <M1> <N1>s?"
        #Presuppositions: (<M> <N> and <M1> <N1>)
        #Answers:
            # yes : count(<M> <N>) < count(<M1> <N1>)
            # no : count(<M> <N>) >= count(<M1> <N1>)

        o1, o2 = helper_two_objects(objects)

        #Presupposition
        if o1.count and o2.count:
            #Answer conditions
            less = o1.count < o2.count
            answer = "yes" if less else "no"

    elif split == ProbeType.BEHIND:
        #BEHIND "answers" : "yes" "no" consider distance and occlusion
        #"Is the <M> <N> behind the <M1> <N1>?"
        #Presuppositions: (<M> <N> and <M1> <N1>) and uniqueness
        #Answers:
            # yes : in_relation_behind(<M> <N>)(<M1> <N1>)
            # no : !in_relation_behind(<M> <N>)(<M1> <N1>)

        o1, o2 = helper_two_objects(objects)

        #Presupposition
        if o1.count == 1 and o2.count == 1:
            #Answer conditions
            behind = o1.index in scene["relationships"]["behind"][o2.index]
            answer = "yes" if behind else "no"

    elif split == ProbeType.FRONT:
        #IN FRONT OF "answers" : "yes" "no" consider distance and occlusion
        #"Is the <M> <N> in front of the <M1> <N1>?"
        #Presuppositions: (<M> <N> and <M1> <N1>)
        #Answers:
            # yes : in_relation_front(<M> <N>)(<M1> <N1>)
            # no : !in_relation_front(<M> <N>)(<M1> <N1>)

        o1, o2 = helper_two_objects(objects)

        #Presupposition
        if o1.count == 1 and o2.count == 1:
            # Answer conditions
            front = o1.index in scene["relationships"]["front"][o2.index]
            answer = "yes" if front else "no"

    elif split == ProbeType.SAME:
        #SAME "answers" : "yes" "no" consider by feature eg what was the feature type and what was the feature value?
        #"Are the <M> <N>s the same <P>?"
        #Presuppositions: count(<M> <N>) >= 2
        #Answers:
            # yes : for all <M> <N>s same<P> == True
            # no : for all <M> <N>s same<P> == False
        property = params[2][1]
        prev_property_value = ""
        object_count = 0
        same = False
        for i, object in enumerate(scene['objects']):
            if params[0][1] in (object['size'], object['color'], object['material'], "") and params[1][1] in (object['shape'], "thing"):
                object_count += 1
                if not prev_property_value:
                    prev_property_value = object[property]
                elif prev_property_value == object[property]:
                    same = True
                else:
                    same = False
                    break
        #Presupposition
        if object_count > 1:
            #Answer conditions
            answer = "yes" if same else "no"

    return (image_index, answer)

def create_probe(template):
    id_counter = 0
    text = template["text"]
    split = template["probe_type"]
    for pt in ProbeType:
        if pt.name == split:
            probe_type = pt
            break
    probe = Probe(split, "v1.0", "CC")
    print(split)
    # If SAME template
    if probe_type == ProbeType.SAME:
        referents = g_one_referent_params
    else :
        referents = g_two_referent_params
    # for each possible param combination
    for values in referents:
        params = []
        for i in range(len(values)):
            name = template["params"][i]['name']
            value = values[i]
            word = value
            params.append((name, value, word))
        question = create_question(text, params)
        images_answers_pairs = (test_scene_answer(scene, probe_type, params) for  scene in g_scenes)
        good_images_answers = []
        for image_index, answer in images_answers_pairs:
            if answer:
                good_images_answers.append((image_index, answer))
        choice_images_answers = random.sample(good_images_answers, 10)
        for image_index, answer in choice_images_answers:
            image_filename = "CLEVR_val_"+str(image_index)+".png"
            q = Question(split, image_index, image_filename, question, answer, id_counter)
            probe.add_q(q)
        id_counter += 1
    filename = probe.split + "_val_questions.json"
    with open(filename, 'w') as f:
        probe_dict = asdict(probe)
        json.dump(probe_dict, f)
    return None



def get_probes_questions():
    # load in question templates and possible parameter values for questions
    with open("probe_templates.json") as templates:
        question_templates = json.load(templates)

    with Pool() as pool:
        pool.map(create_probe, question_templates)



g_scenes = None
g_one_referent_params = None
g_two_referent_params = None

def main():
    global g_scenes
    global g_one_referent_params
    global g_two_referent_params

    with open("one_referent_params.json") as one_referent:
        g_one_referent_params = json.load(one_referent)
    with open("two_referent_params.json") as two_referent:
        g_two_referent_params = json.load(two_referent)

    # Note this will hopefully be replaced by the test image scenes once the authors get back to us
    with open("../../data/CLEVR_v1.0/scenes/CLEVR_val_scenes.json") as image_info:
        scenes = json.load(image_info)
    g_scenes = scenes['scenes']
    get_probes_questions()
        # check out dataclasses.asdict(probe)

if __name__ == "__main__":
    main()
