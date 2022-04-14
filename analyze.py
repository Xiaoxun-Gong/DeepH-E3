import os
from e3Aij import testResultAnalyzer

from e3Aij import e3AijKernel

prompt = '''
------------------------------------------------

# ============================================ #
#     Analyzing Tool for e3Aij Test Result     #
# ============================================ #

Type in the numbers to choose an action:

0) Exit
1) Change the directory of data to be analyzed: (default to current working directory)
   Should contain: test_result.h5, src/train.ini, src/dataset_info.json, src/target_blocks.json
2) Change the output directory (default to <current_working_directory>/analyze_result)
3) View names of structures stored in test_result.h5
4) Get prediction error (csv file and corresponding plot)

... (to be developed)

-------------------------------------------------
'''

data_dir = os.getcwd()
output_dir = os.path.join(os.getcwd(), 'analyze_result')

while True:
    print(prompt, end='')
    
    option = input()
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
        kernel.view_structures()
    elif option == '4':
        print('Input structure name:')
        stru_name = input()
        print('Error mode: (mae/mse/maxe)')
        mode = input()
        kernel = testResultAnalyzer(data_dir, output_dir)
        kernel.error_plot(stru_name, mode)
    else:
        print('Unknown option')
    
    print('\nACTION COMPLETE')