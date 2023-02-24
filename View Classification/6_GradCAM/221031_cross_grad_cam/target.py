'''
수정 필요
'''
# 1. PA 없음: 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010', 'P011', 'P012', 'P013', 'P014', 'P015', 
#          'P016', 'P017', 'P018', 'P019', 'P020', 'P021', 'P022', 'P023', 'P024', 'P025', 'P026', 'P027', 'P028', 'P029', 'P030', 'P031'

# 2. PA 존재: 'P032', 'P033', 'P034','P035','P036'
# 3. RIGHT_DFA_LONG 없음: 'P002'
# 4. RIGHT_PTA_SHORT 없음: 'P023'
# 총 35명  train: 23  test: 12

train_patient = ['P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010', 'P011', 
                 'P012', 'P017', 'P018', 'P019', 'P020', 'P021', 'P022', 'P023', 'P030', 'P031',   
                 'P032', 'P033', 'P035'  ]  #pa 존재

test_patient = [ 'P013', 'P014', 'P015', 'P016', 'P024', 'P025', 'P026', 'P027', 'P028', 'P029', 'P034', 'P036']


# train_patient = ['P002', 'P003'  ]  #pa 존재

# test_patient = [ 'P013']




# target_name = "Main_Vessel_All"
# target_class = [['0',  '16'], 
#                 ['1',  '17'],
#                 ['2',  '18'],
#                 ['3',  '19'],
#                 ['4',  '20'],
#                 ['5',  '21'],
#                 ['6',  '22'],
#                 ['7',  '23'],
#                 ['8',  '24'], 
#                 ['9',  '25'],
#                 ['10',  '26'],
#                 ['11',  '27'],
#                 ['12',  '28'],
#                 ['13',  '29'],
#                 ['14',  '30'],
#                 ['15',  '31'],
#                 ]
# target_class_names = [
#     'CFA_LONG',
#     'SFA_LONG',
#     'DFA_LONG',
#     'POPA_LONG',
#     'ATA_LONG',
#     'PTA_LONG',
#     'PA_LONG',
#     'DORSALIS_LONG',
#     'CFA_SHORT',
#     'SFA_SHORT',
#     'DFA_SHORT',
#     'POPA_SHORT',
#     'ATA_SHORT',
#     'PTA_SHORT',
#     'PA_SHORT',
#     'DORSALIS_SHORT'    
# ]

################################# Long- Short ################################################

# target_name = "Main_Vessel_Long"

# target_class = [['0',  '16'], 
#                 ['1',  '17'],
#                 ['2',  '18'],
#                 ['3',  '19'],
#                 ['4',  '20'],
#                 ['5',  '21'],
#                 ['6',  '22'],
#                 ['7',  '23'],
# ]

# target_class_names = [
#     'CFA_LONG',
#     'SFA_LONG',
#     'DFA_LONG',
#     'POPA_LONG',
#     'ATA_LONG',
#     'PTA_LONG',
#     'PA_LONG',
#     'DORSALIS_LONG'
# ]


# target_name = "Main_Vessel_Short"
# target_class = [['8',  '24'], 
#                 ['9',  '25'],
#                 ['10',  '26'],
#                 ['11',  '27'],
#                 ['12',  '28'],
#                 ['13',  '29'],
#                 ['14',  '30'],
#                 ['15',  '31'],
#                 ]
# target_class_names = [

#     'CFA_SHORT',
#     'SFA_SHORT',
#     'DFA_SHORT',
#     'POPA_SHORT',
#     'ATA_SHORT',
#     'PTA_SHORT',
#     'PA_SHORT',
#     'DORSALIS_SHORT'    
# ]

################################# Above- Below ################################################

######1.=
# target_name  = "Main_Vessel_Long_Above"
# target_class = [
#                 ['0',  '16'], 
#                 ['1',  '17'],
#                 ['2',  '18'],
#                 ['3',  '19'],

# ]
# target_class_names = [
#     'CFA_LONG',
#     'SFA_LONG',
#     'DFA_LONG',
#     'POPA_LONG'
# ]

######2.

target_name = "Main_Vessel_Long_Below"
target_class = [
                ['4',  '20'],
                ['5',  '21'],
                ['6',  '22'],
                ['7',  '23'],
]

target_class_names = [
    'ATA_LONG',
    'PTA_LONG',
    'PA_LONG',
    'DORSALIS_LONG'
]



######3.


# target_name = "Main_Vessel_Short_Above"
# target_class = [['8',  '24'], 
#                 ['9',  '25'],
#                 ['10',  '26'],
#                 ['11',  '27'],

#                 ]
# target_class_names = [

#     'CFA_SHORT',
#     'SFA_SHORT',
#     'DFA_SHORT',
#     'POPA_SHORT',

# ]


######4.


# target_name = "Main_Vessel_Short_Below"
# target_class = [
#                 ['12',  '28'],
#                 ['13',  '29'],
#                 ['14',  '30'],
#                 ['15',  '31'],
#                 ]
# target_class_names = [

#     'ATA_SHORT',
#     'PTA_SHORT',
#     'PA_SHORT',
#     'DORSALIS_SHORT'    
# ]