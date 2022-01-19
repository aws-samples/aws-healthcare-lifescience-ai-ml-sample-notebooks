import sys
import os
#get the requirements
def get_dependencies():
    '''Install Dependencies from requirements.txt skip ones that are installed already'''
    print("Finding Dependencies to Install")
    curr_depends=os.popen('pip list --format=freeze').readlines()
    curr_depends=[i.rstrip() for i in curr_depends]
    desired_depends=os.popen('cat requirements.txt').readlines()
    desired_depends=[i.rstrip() for i in desired_depends]
    depends_to_install=list(set(desired_depends) - set(curr_depends))
    f_out=open('temp_reqs.txt','w')
    print(f'''{str(len(depends_to_install))} dependencies to install''')
    print(*depends_to_install,end="\n",file=f_out)
    f_out.close()
    my_cmd='''cat temp_reqs.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d'|while read line; do TOREPLACE -m pip install $line;done > /dev/null ''' 
    my_cmd=my_cmd.replace('TOREPLACE',sys.executable)
    os.system(my_cmd)
    os.system('rm temp_reqs.txt')
    print("Done Getting Dependencies")
