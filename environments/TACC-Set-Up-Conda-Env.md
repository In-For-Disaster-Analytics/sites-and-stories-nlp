---
created: 2023-10-11T12:48
updated: 2023-10-19T12:10
---
# Working with Jupyter Notebook and Conda Environments at TACC
Jupyter notebooks provide an interactive web based computing platform for working on coding projects. Conda envrionemnts enable users to specify a siloed set of versioned python packages to work with. 

This notebook will walk you through the process of setting up a custom Conda env on TACC systems, and using that environmnet in a jupyter notebook.

## TACC Access and Permissions
The first step is ensuring you have the necessary access to TACC systems.  For this process you will need the following:  
* TACC username and password (Check: can you log-in to [TACC Portal](https://tacc.utexas.edu/portal/))
* TACC Token 2 factor authentication app
* Assignment to an allocation to use compute resources at TACC (Check the 'Projects & Allocations' tab on the TACC Portal and ensure you have an Active Project with an unexpired allocation)

## Creating a Conda environment at TACC and adding it to Notebook kernels
Python environments (here managed with Conda) allow you to isolate a particular set of packages (and their versions) in order to run your code using a very defined set of software.  To set up and utilize a Conda environment at TACC you will do the following step:
1. Access TACC systems via the Command Line
2. Navigate to proper TACC directory
3. Install Conda (only necessary the first time)
4. Create Conda environment either from set of packages devined in a .yml file or by manually installing them
5. Add the new environment as a kernel for use in Jupyter

## 1. Accessing the TACC Command Line

First you'll need to ssh into ls6. Be sure to have your 2-factor TACC token app working and available, as you will need to enter this after your password.

### 1a. Mac
If you're on a Mac use the following in the command line. 
```
ssh jdoe@ls6.tacc.utexas.edu
```

### 1b. Windows
If you're using windows you'll need to use the Putty Software

- Double-click the **Putty** icon
- In the **PuTTY Configuration** window
    - make sure the **Connection type** is `**SSH**`
    - enter **[ls6.tacc.utexas.edu](http://ls6.tacc.utexas.edu/)** for Host Name
        - Optional: to save this configuration for further use:
            - Enter **Lonestar6** into the **Saved Sessions** text box, then click **Save**
            - Next time select **Lonestar6** from the **Saved Sessions** list and click **Load**.
    - click **Open** button
    - answer **Yes** to the SSH security question
- In the **PuTTY** terminal
    - enter your TACC user id after the **"login as**:**"** prompt, then **Enter**
    - enter the password associated with your TACC account
    - provide your 2-factor authentication code
*The putty instructions were copied from [Getting Started at TACC Wiki](https://wikis.utexas.edu/display/CoreNGSTools/Getting+started+at+TACC) by Anna M Battenhouse.*

Note that you won't see the cursor move while you type your password or token into the putty terminal. Simply type and then hit enter.

## 2. Navigate to proper TACC directory
When accessing TACC systems, you are initially located in the $HOME directory.  TACC has three files systems: $HOME, $WORK, and $SCRATCH.  A fuller discussion of these systems are available on the [TACC Lonestar6 documentation](https://docs.tacc.utexas.edu/hpc/lonestar6/#files).  For this exercise we will work on $SCRATCH.

For those new to working in the terminal, a few commands that are helpful to know include:

* pwd: display current working directory
* ls: list files
* ls -a: list all files, including hidden
* cd: change directory
* mkdir {name}: make a new directory called {name}

If you type `pwd` in the terminal, you will see the actual path to your $HOME directory. This should look something like `/home1/01234/username`.  Since the longer addresses would be tedious to type all the time when moving between directories, TACC provides the following alias commands:
| **Alias** 	|** Command**  	| 
|---	|---	|
|`cd` or `cdh`  	| `cd $HOME`  	|  	
|`cds`  	| `cd $SCRATCH`  	|  	
|`cdw`  	| `cd $WORK`  	|  	
	

***For this exercise: navigate into $SCRATCH with `cds`***

## 3. Install conda if necessary
If you don't already have conda, you can install directly from the command line into the current directory with the following code. (Note: These are the [Linux quickstart](https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install) directions with the ~/ removed so that the installation occurs on $Scratch instead of $Home)

```
mkdir -p miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
bash miniconda3/miniconda.sh -b -u -p miniconda3
rm -rf miniconda3/miniconda.shcd
```
Note: the installation may take a few minutes.

Make sure to close out your terminal / putty after installing conda and reopen to have conda available for use. 


## 4. Create Conda environment

### 4a. Transfer .yml file into directory
If you don't already have a directory and files for your project, use `mkdir {name}` to create a new directory and `cd {name}` to move into it. The use the [TACC data transfer protocols](https://docs.tacc.utexas.edu/basics/datatransferguide/) to copy a .yml file from your local computer into this directory. Use `pwd` to get the proper filepath.

### 4b. Create Conda environment 
To create an environment **from scratch**: 
Simply follow standard [Conda environment management](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) protocols. 

To create a pre-set environment **from a .yml file**: 
Navigate into the directory where your environment.yml file is located and create a my-new-env environment by running the following commands to create and then activate your new environment:

    conda env create --name my-new-env --file environment.yml 
    onda activate my-new-env

To add the new environment as a kernel in your Jupyter notebook you install ipykernel in the activated environment and give it a display name for the Jupyter dropdown:

    python -m ipykernel install --user --name my-new-env --display-name "Python (My New Env)"

## 5. Working with Jupyter Notebook at TACC
To run a Jupyter Notebook at TACC you will request a job via the [TACC Analysis Portal](https://tap.tacc.utexas.edu). Once logged in, you should see a dashboard with a 'Submit New Job' option in the top left. If you can log in but don't get this option you likely don't have an active allocation.

Request a Jupyter Notebook job using the following default elections - update these if appropriate for your project:
* System: Lonestar6
* Application: Jupyter notebook
* Project: project sponsoring the work / which allocation you want to use
* Queue: development 

Submit job request and wait until it launches.

Once it launches you can either start a new notebook, or use file transfer protocols to bring an existing notebook into your files for use.
