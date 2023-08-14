'''
Get word counts in CLEVR corpus.
'''
import os
import sys
import argparse
import json
import csv

'''
Gets arguments from the command-line.

Returns:
    params: a dictionary of command-line arguments
'''
def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_file", default="../../data/CLEVR_v1/data/CLEVR_train_questions.json")
    params = parser.parse_args()
    return params

def get_total_word_count_and_data(question_file):
    total_word_count = 0
    data = []
    with open(question_file) as f:
        questions_data = json.load(f)
    for q in questions_data['questions']:
        line = q['question']
        line = line.strip('?').lower()
        line_as_list = line.split(" ")
        data += line_as_list
        total_word_count += len(line_as_list)
    return data, total_word_count

def get_token_counts(data):
    word_set = {"and", "or", "more", "fewer", "behind", "front", "same"}
    token_count_dict = {"and": 0, "or": 0, "more": 0, "fewer": 0, "behind": 0, "in front": 0, "same": 0}
    prev_word = ""
    for word in data:
        if word in word_set:
            if word == "front":
                if prev_word == "in":
                    token_count_dict["in front"] += 1
            else:
                token_count_dict[word] += 1
        prev_word = word
    return token_count_dict


def main():
    params = get_parameters()
    question_file = params.question_file
    print("GET DATA AND TOTAL WORD COUNT...")
    data, total = get_total_word_count_and_data(question_file)
    print("GET BY WORD COUNT...")
    token_count_dict = get_token_counts(data)
    file_name = "CLEVR_frequency_counts.csv"
    with open(file_name, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["word", "count", "total"])
        for k,v in token_count_dict.items():
            writer.writerow([k, v, total])

if __name__=="__main__":
    main()
