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
