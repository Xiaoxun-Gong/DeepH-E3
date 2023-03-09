#!/usr/bin/env python
import os
from deephe3 import testResultAnalyzer

prompt = '''
------------------------------------------------

# ============================================ #
#    Analyzing Tool for DeepH-E3 Test Result    #
# ============================================ #

Type in the numbers to choose an action:

0) Exit
1) Change the directory of data to be analyzed: (default to current working directory)
   Should contain: test_result.h5, src/train.ini, src/dataset_info.json, src/target_blocks.json
2) Change the output directory (default to <current_working_directory>/analyze_result)
3) View names of structures stored in test_result.h5
4) Get prediction error of selected structure (csv file and corresponding plot)
5) MSE and MAE analysis
6) Get predicted hamiltonian

... (to be developed)

-------------------------------------------------
'''

data_dir = os.getcwd()
output_dir = os.path.join(os.getcwd(), 'analyze_result')

print_prompt = True
while True:
    if print_prompt:
        print(prompt, end='')
    else:
        print('Secelct action (0-6):')
    
    option = input()
    ret = 0
    if option == '0':
        exit()
    elif option == '1':
        print('Data dir:')
        data_dir = input()
    elif option == '2':
        print('Output dir:')
        output_dir = input()
    elif option == '3':
        kernel = testResultAnalyzer(data_dir, output_dir)
        ret = kernel.view_structures()
    elif option == '4':
        # print('Input structure name:')
        # stru_name = input()
        print('Error mode: (mae/mse/maxe/rmse)')
        mode = input()
        print('Structures to include:')
        print(' - Separate structure names by space; support regular expressions; leave empty to include all')
        include = input().split()
        print('Structures to exclude:')
        print(' - Leave empty to exclude none')
        exclude = input().split()
        kernel = testResultAnalyzer(data_dir, output_dir)
        ret = kernel.error_plot(include, exclude, mode)
    elif option == '5':
        print('Structures to include:')
        print(' - Separate structure names by space; support regular expressions; leave empty to include all')
        include = input().split()
        print('Structures to exclude:')
        print(' - Leave empty to exclude none')
        exclude = input().split()
        kernel = testResultAnalyzer(data_dir, output_dir)
        ret = kernel.error_analysis(include=include, exclude=exclude)
    elif option == '6':
        print('Structure name:')
        stru_name = input()
        kernel = testResultAnalyzer(data_dir, output_dir)
        ret = kernel.get_hamiltonian(stru_name)
    else:
        print('Unknown option')
        ret = 1
    
    if ret == 0:
        print('\nACTION COMPLETE')
    else:
        print('\nACTION TERMINATED WITH ERROR')
    print_prompt = False
    print('Show prompt again? (y/n)')
    select = input()
    if select == 'y':
        print_prompt = True
