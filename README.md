# quantized_nn_backreach
Quantized Verification of Neural Networks using Backreachability for ACAS Xu


# tips for running on AWS remotely

use base image ubuntu 20.04 LTS, c6i.metal instance

ssh -i "sbu_laptop.pem" ubuntu@ec2-54-245-212-207.us-west-2.compute.amazonaws.com

sudo apt update
sudo apt upgrade
sudo apt install python3-pip
pip3 install --upgrade pip

git clone https://github.com/stanleybak/quantized_nn_backreach.git

cd quantized_nn_backreach
pip3 install -r requirements.txt


test if things looks like they're working
python3 backreach.py (ctrl + c to kill)

For an persistent process that will keep running when ssh closes, and then shuts down the connection use:
~> stdbuf -oL python3 backreach.py >& stdout.txt &

alternatively, if you also want to shut down the machine when things are done, try this one instead (untested, not sure if it actually stops it: 
~> ((stdbuf -oL python3 backreach.py >& stdout.txt); sudo halt) & 


(stdbuf -oL disables unnecessary buffering when redirecting stdout, stdout and stderr will be sent to the file my_stdout.txt and the '&' starts the process in the backround)
-> disown -a

(this disconnects all processes whose parent is the ssh shell, so if ssh closes the measurement process keeps running)
-> tail -f stdout.txt (print progress as it gets sent to my_stdout.txt
