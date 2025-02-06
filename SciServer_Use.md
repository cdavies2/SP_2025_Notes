# Working with Files
* Use "Files" to upload reasonably sized files or directories through the web interface. You can upload through persistent and scratch space, and you can work with files of up to 250 GB.

# Starting Containers
* The "Compute" button is used for running interactive compute jobs (like Jupyter notebooks)
  * Within "Compute", click the green "Create container" button
  * Enter a "Container name" for your container (EX: "[your_username}_test_1")
  * Select a "Domain" to choose the hardware resources allocated (details for said domain are displayed in a pull down menu)
  * Select your desired "Compute Image"
  * Check the desired User and Data volumes to mount within your container
  * Click the green "Create" button
  * If the container starts successfully and shows "running for Status, click the container name. If it shows "stopped" refresh the page, and if clicking on the name opens another tab, close said tab and click the name again (might have been too fast)
  * At this point you should be in a working Jupyter or Xwindows environment and able to read/write files to your storage volumes....
    * ~/workspace/Storage/(username)/persistent
    * ~/workspace/Temporary/(username)/scratch
* Stop the container (red square) when done with a session, as it uses allocated computing resources.
 
# Data Transfer Use File app in SciServer
* SciServer's File app is used to do basic file transfer between a workstation/laptop and SciServer. You can zip/unzip or tar/untar from any compute container, but the File app cannot interact with other servers on the Internet

## Use BNL SDCC as Stage of Data Transfer
* If you have a BNL SDCC account, you can use Globus to transfer files from outside to SDCC's Globus endpoint. sftp can also be used to transfer files to SDCC's sftp server sftp.sdcc.bnl.gov. On SDCC's account users can utilize /hpcgfps01/scratch area for temporary data up to 2TB (files are deleted if untouched for 30 days).
* You need a copy of your sciserver private key to transfer data from SciServer to SDCC storage.

## Use dtn Container for ssh/sftp Based Data Transfer
* On SciServer compute, you can create and start a container in "sciserver-dtn1" compute domain using the "vcn-kde-dtn" image. Make sure to mount the volumes you wish to transfer data to/from you create the container, some user volumes aren't mounted by default.
* This domain has outgoing ssh access to the Internet. From the terminal you can start ssh/scp/sftp/rsync sessions the internet ssh/sftp servers. It also makes ssh-key based git push/pull easier.
* Remember, stop the container after data transfer.

## Use Globus Personal Connect Client
* Useful in transferring data between sciserver and data on a public Globus Endpoint

## Initial Setup for Using Globus on SciServer
1. Start a SciServer Session using the sciserver-dtnl domain and the vnc-kde-dtn image
2. Create a personal globus endpoint: create an endpoint (with a name), generate a key for setup and copy it
3. Start Globus Connect Personal and use the key to setup your personal endpoint   
```
globusconnectpersonal -dir /home/idies/workspace/Storage/[your_username]/persistent/.globusonline -setup --setup-key ${your_setup_key}
```
4. Start your Globus Connect Personal endpoint
```
globusconnectpersonal -dir /home/idies/workspace/Storage/[your_username]/persistent/.globusonline -start &
```
* You should now be able to see your personal endpoint via the Collection search bar
5.  Stop your endpoint
```
globusconnectpersonal -dir /home/idies/workspace/Storage/[your_username]/persistent/.globusonline -stop
```

## Regular Use
* After completing the initial setup as detailed above, your globus-related dot files should be stored in a persistent storage location, so you can start and stop the endpoint by using the commands below.

```
globusconnectpersonal -dir /home/idies/workspace/Storage/[your_username]/persistent/.globusonline -start &
```
```
globusconnectpersonal -dir /home/idies/workspace/Storage/[your_username]/persistent/.globusonline -stop
```
# Conda Virtual Env
* The directories used by Conda for packages and default virtual env paths can be set in ~/.condarc, but that, like other ~/.* files, are lost when the sciserver container restarts.
* It is suggested to create an init script in your persistent storage, and run it whenever you start or restart the container.
* A .condarc file looks like this
```
envs_dirs:
  - /home/idies/workspace/Storage/{YourUsername}/persistent/conda/conda_envs
pkgs_dirs:
  - /home/idies/workspace/Storage/{YourUserName}/persistent/conda/conda_pkgs
```
* A sample init.sh script you run every time you start/restart the container looks as seen below
```
#/bin/bash


# script you run every time your container start or restart
#to customize container

#username can be found in ls  ${HOME}/workspace/Storage/
MY_USERNAME=johndoe


export MY_TMPDIR=${HOME}/workspace/Temporary/${MY_USERNAME}/scratch/tmp


#if you need to preserve those directories, 
rm -rf ~/.cache
if [ ! -d ${MY_TMPDIR}/.cache ] ; then
 mkdir -p ${MY_TMPDIR}/.cache
fi
ln -s ${MY_TMPDIR}/.cache ${HOME}/.cache


# if you have  customerized .bashrc etc , copy it over or link it.
export my_username=$(ls /home/idies/workspace/Storage)
if [ -f /home/idies/workspace/Storage/${my_username}/persistent/init/.bashrc ]; then
cp  /home/idies/workspace/Storage/${my_username}/persistent/init/.bashrc ${HOME}/.bashrc
 ln -sf  /home/idies/workspace/Storage/${my_username}/persistent/init/.bashrc ${HOME}/.bashrc
fi


# if you use conda env  and have .condarc might want cp or link
if [ -f /home/idies/workspace/Storage/${my_username}/persistent/conda/.condarc ] ; then
cp  /home/idies/workspace/Storage/${my_username}/persistent/conda/.condarc ${HOME}/
ln -sf  /home/idies/workspace/Storage/${my_username}/persistent/conda/.condarc ${HOME}/.condarc
fi


# if you have created your own jupyter kernels
# and made backup from ./local/hare/jupyter/kernels/
# you can copy back to container or link it
# if you want you can relink whole .local folder too
if [ -d /home/idies/workspace/Storage/${my_username}/persistent/conda/jupyter_kernels ] ; then
 #cp  -r /home/idies/workspace/Storage/${my_username}/persistent/conda/jupyter_kernels/* ${HOME}/.local/share/jupyter/kernels/
 # or use link
 rm -rf ${HOME}/.local/share/jupyter/kernels/        # assume you have backup
 ln -s  /home/idies/workspace/Storage/${my_username}/persistent/conda/jupyter_kernels ${HOME}/.local/share/jupyter/kernels
fi
## some other .files/directories you may have made backup
# e.g. .mozilla
if [  ! -d /home/idies/workspace/Storage/${my_username}/persistent/init/.mozilla ]; then
 mkdir /home/idies/workspace/Storage/${my_username}/persistent/init/.mozilla
fi
if [ -d /home/idies/.mozilla ] ; then
 rm -rf /home/idies/.mozilla
fi
ln -s /home/idies/workspace/Storage/${my_username}/persistent/init/.mozilla /home/idies/.mozilla

```
* The conda virtualenv can be created as so
```
conda create -n my_testenv  python=3.10
#activate env
source activate my_testenv
```
* Install packages with this command
```
#install packages use conda 
conda install numpy
#install package use pip
pip install torch torchvision torchaudio
```
Source: https://sciserver.sdcc.bnl.gov/scisrvdoc/03_conda_env/condaenv.html

# SciServer Compute
* SciServer compute allows users to create and run Jupyter Notebooks containing code and instructions to analyze and process SciServer hosted data sets.
* The full capabilties of Jupyter are available to users within containers
* A container in SciServer Compute is a defined environment within which Jupyter Notebook runs, and it isolates users' work from both the rest of the SciServer system and other users.
* Data that you want to keep should NEVER be stored in the container itself. Always store data files in the _Storage_ or _Temporary_ storage pools. These are external to the container and accessible when containers are closed or deleted.
 ## Creating New Containers
 * When you press the "Create Container" button, the following options appear....
  * Container Name: Chosen by You
  * Domain: Drop down that should always be left at the default value "Interactive Docker Compute Domain"
  * Image: defines a "software environment" for the Jupyter notebooks that you want to run. The images contain libraries tailored to different needs. You choose an image that supports the language you are interested in (python, R, Matlab, etc), but there are "specialty" science domain specific images that you might have access to if the creator of those images has shared it with you
  * User Volumes: list of all User Volumes that you have access to, either which you own, or which have been shared with you. When you select some of these, on container creation these folders will be "mounted" and will be accessible as if they were local files. This simplifies file access and management.
  * Data Volumes: series of special Data Volumes that are either shared publicly with all users, or for which you have been given special access privileges to see. Selecting these Volumes will mount them and make them appear "local" in the Container. These will always be mounted "readonly"

# SciServer Compute Jobs
* Jobs allow a user to run a Jupyter Notebook or a standard script in offline batch mode.
* A job might be created because....
 * Executing the notebook may take a long time, and you want to set it running and do something else without worrying about browser sessions timing out
 * You may develop your code interactively to make sure the algorithm works, using a small amount of data to test it out, but you really want to run the code against a full dataset, which will require massive resources for memory and CPU, as well as execution time.
 * You are provided with far more resources (CPU and memory) to execute a Job than you are in an Interactive Session.
* SciServer allows two "types" of jobs....
   1. Specify a script to execute, or a command line command
   2. Specift an existing Jupyter Notebook that you previously developed (meaning you can develop your Jupyter Notebook interactively and "submit" the same notebook as a Job)

## Creating and Running a Job
1. Go to the Compute Jobs Page
2. Click "Run Existing Notebook"
3. On the "Compute Domain" Tab, choose the Compute Domain, for which most cases currently there will only be one option, and you can enter a "Job Alias" to more easily find your job later
4. On the "Compute Image" Tab, pick the "Image" you need to use (each one contains different tools and programming language support)
5. On the "Data Volumes" Tab, select all the data volumes with appropriate permissions needed for the job
6. On the "User Volumes" Tab, select all Folder systems that you would like to be made accessible to your Compute notebook. For Folders that you own, or that have been shared with you and you were given the appropriate permissions, you can select whether a given folder is read only or writable. Folders that you don't own will be readonly by default.
7. On the "Notebook" tab, navigate to the Notebook you wish to use as the basis for your Job and select it. Enter any additional parameters that the Notebook can read in to affect how the code is executed. Choose a directory where the output results will go, by default these will go to jobs within which subdirectories will be created and your results written to. You can also choose a specific directory to output results.
8. When everything has been entered you can press "Create Job", and the Job will be submitted and displayed in a Jobs Table view.
9. The table will be refreshed every few seconds and tell you the status of the Job.
10. While the job is still running there is a red X that can cancel it.
11. Pressing the down triangle on the right-hand side expands the view and shows more information. This includes status information about the Job, the path to the location of the results, and links to the results output. Those links include...
  * _Browse Working Directory_: take you to the Dashboard Files tab and show you the output files as well as the original Python Notebooks
  * _Download Standard Output_ and _Download Standard Error_ allow you to download two text files as appropriate and according to your browser settings. 
* Source: https://www.sciserver.org/support/how-to-use-sciserver/
