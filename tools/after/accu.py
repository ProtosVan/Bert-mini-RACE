import os
import sys
import json
import numpy as np

def accu(temp_list):
    options, answer = temp_list
    if np.argmax(options) == answer:
        return True
    else:
        return False



with open("model1_test_h.json", "r") as temp_file:
    m1h_json = json.load(temp_file)
with open("model1_test_m.json", "r") as temp_file:
    m1m_json = json.load(temp_file)
with open("model2_test_h.json", "r") as temp_file:
    m2h_json = json.load(temp_file)
with open("model2_test_m.json", "r") as temp_file:
    m2m_json = json.load(temp_file)
with open("model3_test_h.json", "r") as temp_file:
    m3h_json = json.load(temp_file)
with open("model3_test_m.json", "r") as temp_file:
    m3m_json = json.load(temp_file)

print("Model 1 High, Len:%d"%len(m1h_json))
accu1h_num = 0
for temp_list in m1h_json:
    if accu(temp_list):
        accu1h_num += 1
print(accu1h_num/len(m1h_json))

print("Model 1 Middle, Len:%d"%len(m1m_json))
accu1m_num = 0
for temp_list in m1m_json:
    if accu(temp_list):
        accu1m_num += 1
print(accu1m_num/len(m1m_json))


print((accu1h_num+accu1m_num)/(len(m1h_json)+len(m1m_json)))

print("Model 2 High, Len:%d"%len(m2h_json))
accu2h_num = 0
for temp_list in m2h_json:
    if accu(temp_list):
        accu2h_num += 1
print(accu2h_num/len(m2h_json))

print("Model 2 Middle, Len:%d"%len(m2m_json))
accu2m_num = 0
for temp_list in m2m_json:
    if accu(temp_list):
        accu2m_num += 1
print(accu2m_num/len(m2m_json))

print((accu2h_num+accu2m_num)/(len(m2h_json)+len(m2m_json)))

print("Model 3 High, Len:%d"%len(m3h_json))
accu3h_num = 0
for temp_list in m3h_json:
    if accu(temp_list):
        accu3h_num += 1
print(accu3h_num/len(m3h_json))

print("Model 3 Middle, Len:%d"%len(m3m_json))
accu3m_num = 0
for temp_list in m3m_json:
    if accu(temp_list):
        accu3m_num += 1
print(accu3m_num/len(m3m_json))

print((accu3h_num+accu3m_num)/(len(m3h_json)+len(m3m_json)))