import shlex, subprocess
import sys

PYTHON_DEFAULT = 'python'+str(sys.version_info[0])
SETTING = {
  'DOMAIN': 'wot.mecacloud.top'
}

def run_tuna(port,token):
  tokens = token.split(':')
  if len(tokens) == 2:
    frontend = " --frontend="+tokens[0]+"."+SETTING['DOMAIN']+":80"
    service_on = " --service_on=http:www."+tokens[0]+"."+SETTING['DOMAIN']+":localhost:"+str(port)+":"+tokens[1]
    command = PYTHON_DEFAULT+" ./pagekite.py --clean"+frontend+service_on
    args = shlex.split(command)
    p = subprocess.Popen(args)
    print('Initial TUNNEL =I====> [ https://'+tokens[0]+'.kb-proxy.meca.in.th ] <====I=')
  else:
    print('Invalid token')

def main():
  pass

if __name__ == '__main__':
  main()
