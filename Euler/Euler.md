# Connecting to ETHZ Euler clusters
The wiki to connect to the Euler cluster is https://scicomp.ethz.ch/wiki/Getting_started_with_clusters.
The connection can be done with ssh and is relatively simple. Navigate down to [section 2.4: SSH](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters#SSH) to see how to set up this connection with your OS.
I also followed the steps on creating SSH keys but I'm not sure it's necessary or even worth it within our scope.

On your first login, you'll have to accept some terms and conditions, otherwise that's it.

From then on, to connect to Euler you just have to run the command 

`ssh username@hostname`

on your terminal to connect, where username is your nethz username, and hostname is euler.ethz.ch

# Using the clusters
## Copying files onto the cluster

The [File transfer](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters#File_transfer) section in the wiki has some useful
command lines that you might want to use to transfer files/directories onto the cluster.

For example, to transfer the Pycharm project I'm currently working on onto the cluster, I used the command 

`scp -r /Users/leonardobarberi/PycharmProjects/AML_task2 lbarberi@euler.ethz.ch:/cluster/home/lbarberi`

from my terminal. I can now find the folder `AML_task2` and all its files from the virtual machine, on the directory
`./AML_task2/`

To read an output from a job, in this example with the file name `AML_task2/output`, you can run the command `cat AML_task2/output`.
I'm not entirely sure but I think the virtual machines run on Ubuntu so all the commands from there should be the same.

## Sending Jobs
The whole batch system is explained in [section 5: Using the batch system](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters#Using_the_batch_system). To run any 
command, I recommend using the [LSF/Slurm Submission Line Advisor](https://scicomp.ethz.ch/public/lsla/index2.html). 

In this folder, you can find a screenshot of the inputs I put for the command to run the script in `test.py`.
You then copy the command in the output and put it in your terminal where you're connected to the Euler machine.

For an actual job that requires high computation, just change the necessary parameters in the submission line advisor