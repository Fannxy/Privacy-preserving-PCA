"""
Experiments - Mooc
Dataset: http://moocdata.cn/data/MOOCCube

This plain-text version is for demonstration about the data integration vertically.
Care that you should download the dataset to ./Data/Mooc, before running this demo.
"""

import numpy as np
import json
import scipy.io as scio
import random
import time

from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier

from plain_preprocess import *
from PCA_application import *
from jacobi import *

mooc_path = './Data/Mooc/' # relationships
mooc_cube = mooc_path+'MOOCCube/'
mooc_entites = mooc_cube+'entities/'
mooc_relations = mooc_cube+'relations/'

entities = {
    'user':[],
    'course':[],
    'concept':[],
    'school':[]
}

relations = {
    'user-course':[],
    'course-concept':[],
    'concept-field':[],
    'school-course':[]
}

def load_json_list(path):
    """Loading json list files
    """
    json_list = []
    with open(path) as f:
        for jsonObj in f:
            dict_item = json.loads(jsonObj)
            json_list.append(dict_item)
            
    return json_list

def load_relations(path):
    """Loading the interactions
    """
    relation_list = []
    with open(path) as f:
        for line in f.readlines():
            item = line[:-1].split('\t')
            relation_list.append(item)

    return relation_list

def pre_load():
    """loading the entities and relations
    """
    for item in entities:
        entities[item] = load_json_list(mooc_entites+item+'.json')
        print("How many items in %s <- %d" %(item, len(entities[item])))
    
    # relationship loading
    for item in relations:
        relations[item] = load_relations(mooc_relations+item+'.json')
        print("How many items in %s <- %d" %(item, len(relations[item])))


def user_field_encode(su_dict, fields_encode_map):
    """Encode each users for each school
    """
    dd = defaultdict(list)
    value_list = []
    user_list = []
    
    for item in su_dict:
        user, values = item.keys(), item.values()
        if(list(values)[0] not in fields_encode_map.keys()):
            continue
        dd[list(user)[0]].append(fields_encode_map[list(values)[0]])

    for key, value in dd.items():
        encode = np.zeros(24) # 23 fields
        encode[value] = 1
        value_list.append(encode)
        user_list.append(eval(key[2:]))
    
    index = np.argsort(user_list)
    
    return np.array(user_list)[index], np.array(value_list)[index]


def basic_load(entities, relations, K=5):
    """Load the basic dataset

    K: for extracting the most popular K schools.
    """
    fields = []
    # # entities loading
    # for item in entities:
    #     entities[item] = load_json_list(mooc_entites+item+'.json')
    #     print("How many items in %s <- %d" %(item, len(entities[item])))
    
    # # relationship loading
    # for item in relations:
    #     relations[item] = load_relations(mooc_relations+item+'.json')
    #     print("How many items in %s <- %d" %(item, len(relations[item])))
    # for item in relations['concept-field']:
    #     fields.append(item[1])

    fields = set(fields)
    fields_encode_map = dict(zip(fields, range(len(fields))))

    # map construction
    concept_field_dict ={}
    for item in relations['concept-field']:
        concept_field_dict.update({item[0]:item[1]})
    
    # course - field
    course_field = []
    course_field_dict = {}
    for item in relations['course-concept']:
        new_relation = item[0]+"*"+concept_field_dict[item[1]]
        course_field.append(new_relation)     
    course_field = [item.split('*') for item in list(set(course_field))]

    for item in course_field:
        course_field_dict.update({item[0]:item[1]})
    
    # seperate by schools
    school_list = []
    for item in entities['school']:
        school_list.append(item['id'])
    
    # find the K most popular schools
    school_course_numbers = {}
    school_now = relations['school-course'][0][0]
    item_num = 0
    for item in relations['school-course']:
        if item[0] == school_now:
            item_num += 1
        else:
            school_course_numbers.update({school_now:item_num})
            item_num = 0
            school_now = item[0]

    school_course_numbers.update({school_now:item_num})
    target_schools = [item[0] for item in sorted(school_course_numbers.items(), key = lambda item:item[1])[::-1][:K]]
    print("Target schools: ", target_schools)
    target_schools.append('others')

    # course-school map construct
    course_school_dict = {}
    for item in relations['school-course']:
        if item[0] in target_schools:
            course_school_dict.update({item[1]:item[0]})
        else:
            course_school_dict.update({item[1]:'others'})

    # find users for each schools
    school_users = {item:[] for item in target_schools} # initialize method
    for item in relations['user-course']:
        if(item[1] not in course_school_dict.keys()):
            continue;
        school = course_school_dict[item[1]]
        if(item[1] not in course_field_dict.keys()):
            user_field = {item[0]:'others'}
        else:
            user_field = {item[0]:course_field_dict[item[1]]}
        school_users[school].append(user_field)

    target_schools.pop(-1)
    school_information = {item:{'users':[], 'info':[]} for item in target_schools}
    for item in school_information:
        print(item)
        school_information[item]['users'], school_information[item]['info'] = user_field_encode(school_users[item], fields_encode_map)
    
    return school_information


def data_add(user_list, s_info, d, key='info'):
    """Add the missing information
    """
    diff = set(user_list) - set(s_info['users'])
    fit_t = np.zeros((len(diff), d))
    users_add = np.concatenate([s_info['users'], list(diff)])
    data_add = np.concatenate([s_info[key], fit_t], axis=0)
    
    index = np.argsort(users_add)
    
    return data_add[index]

def find_popular_concepts(k=100):
    """Select the most popular 100 concepts
    >> change - should not find the most popular concepts but concepts not that tightly related
    version - 02-03: random select
    """
    concept_course_map = {item['id']:[] for item in entities['concept']}
    related_courses = []

    for item in relations['course-concept']:
        concept_course_map[item[1]].append(item[0])
    n_pop = len(concept_course_map) // 5
    popular_concepts = sorted(concept_course_map.items(), key=lambda item:len(item[1]))[-n_pop:]
    popular_concepts = random.sample(popular_concepts, k)
    print(popular_concepts[1])

    return popular_concepts

def courses_encode(concepts_list):
    """find the related courses with the popular concepts
    """
    d = len(concepts_list)
    related_courses = []
    concept_map = {}
    i = 0
    for item in concepts_list:
        related_courses.extend(item[1])
        concept_map.update({item[0]:i})
        i += 1

    courses_concept_map = {item:[] for item in set(related_courses)}
    for concept, courses_ in concepts_list:
        for course in courses_:
            courses_concept_map[course].append(concept_map[concept])

    for item in courses_concept_map:
        encode_array = np.zeros(d)
        encode_index = list(set(courses_concept_map[item]))
        encode_array[encode_index] = 1
        courses_concept_map[item] = encode_array
    
    return courses_concept_map

def users_encode(courses_map, d):
    """encode users with the corresponding encode
    """
    total_users = []
    course_user_map = {item:[] for item in courses_map}
    group_info = {'users':[], 'codes':[]}
    
    for user, course in relations['user-course']:
        try:
            course_user_map[course].append(user)
        except:
            continue;
    
    for item in course_user_map:
        total_users.extend(course_user_map[item])
    total_users = list(set(total_users))
    user_encode_map = {item:np.zeros(d) for item in total_users}

    for item in course_user_map:
        for user in course_user_map[item]:
            user_encode_map[user] += courses_map[item]
    
    group_info['codes'] = np.concatenate([[user_encode_map[item]] for item in user_encode_map], axis=0)
    print(group_info['codes'].shape)
    group_info['users'] = total_users
    print(len(group_info['users']))

    return group_info

def task_construction(popular_concepts, k, user_list):
    """select the first k courses containing the related concepts
    label the users to construct the problem.
    """
    user_list = set(user_list)

    related_courses = []
    for item in popular_concepts:
        related_courses.extend(item[1])
    
    related_courses = list(set(related_courses))
    courses_popular_map = {item:[] for item in related_courses}

    for concept, item in popular_concepts:
        for course in item:
            courses_popular_map[course].append(concept)

    for item in courses_popular_map:
        courses_popular_map[item] = set(courses_popular_map[item])

    most_popular_course = sorted(courses_popular_map.items(), key=lambda item:len(item[1]))
    
    target_course = [item[0] for item in most_popular_course[-k:]]
    course_user_map = {item:[] for item in target_course}

    for user, course in relations['user-course']:
        try:
            course_user_map[course].append(user)
        except:
            continue;

    user_joint = []
    for item in target_course:
        # print(item)
        user_joint.extend(course_user_map[item])

    user_joint = set(user_joint)
    user_joint = user_joint & user_list
    diff = np.array(list(user_list - user_joint))
    user_joint = np.array(list(user_joint))

    label_joint = np.ones(len(user_joint))
    label_others = np.zeros(len(diff))

    label_index = np.argsort(np.concatenate([user_joint, diff]))
    label = np.concatenate([label_joint, label_others])[label_index]

    return label


def model_evaluation(X, y, train_index, test_index, k=16):
    """Using specified data to evaluate the model
    """
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print(">>>>>>> x.shape", X_train.shape)
    p_matrix, X_reduce = dimension_reduction(X_train, k=k)
    bdt = GradientBoostingClassifier()
    print(">>>>>>> x.shape", X_reduce.shape)
    bdt.fit(X_reduce, y_train)

    X_test = np.dot(X_test, p_matrix)
    y_pred = bdt.predict(X_test)

    print(classification_report(y_test, y_pred, digits=4, target_names=['no', 'yes']))


if __name__ == '__main__':

    L, c = 10, 200 # <- number of joint parties
    pre_load()
    popular_concepts = find_popular_concepts(k=c)
    np.random.shuffle(popular_concepts)
    d = c // L

    concept_batch_len = len(popular_concepts) // L
    split_concepts = [popular_concepts[i*concept_batch_len:(i+1)*concept_batch_len] for i in range(L)]

    data_each = {}
    i = 0
    for concept_list in split_concepts:
        d = len(concept_list)
        courses_concept_map = courses_encode(concept_list)
        user_info = users_encode(courses_concept_map, d)
        data_each.update({str(i):user_info})
        i += 1

    scio.savemat(mooc_path+'data_v5.mat', data_each)
    data_each = scio.loadmat(mooc_path+'data_v5.mat')
    # reorgnize the data_dict
    data_each_raw = {}
    for i in range(L):
        data_each_raw.update({str(i):{'users':data_each[str(i)]['users'][0][0], 'info':data_each[str(i)]['codes'][0][0]}})

    total_user_list = []
    for i in range(L):
        print("dealing users")
        print(i)
        print(len(data_each_raw[str(i)]['users']))
        print(data_each_raw[str(i)]['info'].shape)

        user_list = data_each_raw[str(i)]['users']
        total_user_list.extend(list(user_list))
        print("finish dealing users")
    total_user_list = list(set(total_user_list))

    # # construct the target problem
    # print("users:", len(total_user_list))
    # label = task_construction(popular_concepts, 400, total_user_list)
    # print("total positive samples: ", np.sum(label))

    data_parties = {}
    time_list = []
    for i in range(L):
        time_1 = time.time()
        data = data_add(total_user_list, data_each_raw[str(i)], d=d)
        time_2 = time.time()

        time_list.append(time_2 - time_1)
        np.savetxt(mooc_path+'2PC_data/parties-'+str(i)+'.txt', data, fmt='%d', delimiter=',')
        print("final shape: ", data.shape)
        data_parties.update({str(i):data})
    
    print("Time - locallt preprocess: ", np.mean(time_list))
    
    # classification problem
    # benchmark >> fused dataset
    random.seed(16)
    whole_index = range(len(total_user_list))
    sample_num = int(len(total_user_list)*0.1)
    test_index = random.sample(whole_index, sample_num)
    train_index = list(set(whole_index) - set(test_index))

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>> benchmark - fused dataset")
    X = np.concatenate([data_parties[item] for item in data_parties], axis=1)
    print("X shape: ", X.shape)
    y = label
    model_evaluation(X, y, train_index, test_index, k=7)

    # single party
    print(">>>>>>>>>>>>>>>>>>>>>")
    print("\n>>>> single parties")
    for i in [0, 1, 7]:
        print("===== party-", i)
        X = data_parties[str(i)]
        print(X.shape)
        y = label

        model_evaluation(X, y, train_index, test_index, k=7)


    # three parties
    print(">>>>>>>>>>>>>>>>>>>>>")
    print("\n>>>> three parties")
    S = 3
    parties_list = [range(i, i+S) for i in range(L - S)]
    parties_list = random.sample(parties_list, 3)
    print(parties_list)
    for i in range(3):
        print("===== party-", i)
        X = np.concatenate([data_parties[str(item)] for item in parties_list[i]], axis=1)
        y = label
        model_evaluation(X, y, train_index, test_index, k=7)
    

    # five parties
    print(">>>>>>>>>>>>>>>>>>>>>")
    print("\n>>>> five parties")
    S = 5
    parties_list = [range(i, i+S) for i in range(L - S)]
    parties_list = random.sample(parties_list, 3)
    print("len < ", len(parties_list))
    print(parties_list)
    for i in range(3):
        print("===== party-", i)
        X = np.concatenate([data_parties[str(item)] for item in parties_list[i]], axis=1)
        y = label
        model_evaluation(X, y, train_index, test_index, k=7)
    

    # ten parties
    print(">>>>>>>>>>>>>>>>>>>>>")
    print("\n>>>> seven parties")
    S = 7
    parties_list = [range(i, i+S) for i in range(L - S)]
    parties_list = random.sample(parties_list, 3)
    print(parties_list)
    for i in range(3):
        print("===== party-", i)
        X = np.concatenate([data_parties[str(item)] for item in parties_list[i]], axis=1)
        y = label
        model_evaluation(X, y, train_index, test_index, k=7)


        



    




    
