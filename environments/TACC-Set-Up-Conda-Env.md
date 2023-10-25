---
created: 2023-10-11T12:48
updated: 2023-10-19T12:10
---
# Working with Jupyter Notebook and Conda Environments at TACC
Jupyter notebooks provide an interactive web based computing platform for working on coding projects. Conda envrionments enable users to specify an isolated set of versioned python packages to work with. 

This notebook will walk you through the process of setting up a custom Conda env on TACC systems, and using that environmnet in a jupyter notebook.

## TACC Access and Permissions
The first step is ensuring you have the necessary access to TACC systems.  For this process you will need ALL of the following:  
1. TACC username and password (Check: can you log-in to [TACC Portal](https://tacc.utexas.edu/portal/))
2. [TACC multifactor authentication](https://docs.tacc.utexas.edu/basics/mfa/), you will use this to enter an authenticating 'TACC Token' as needed
3. Assignment to an allocation to use compute resources at TACC (Check the 'Projects & Allocations' tab on the TACC Portal and ensure you have an Active Project with an unexpired allocation on Lonestar6)

## Creating a Conda environment at TACC and adding it to Notebook kernels
Python environments (here managed with Conda) allow you to isolate a particular set of packages (and their versions) in order to run your code using a very defined set of software.  To set up and utilize a Conda environment at TACC you will do the following steps:
1. Access TACC systems via the Command Line
2. Navigate to proper TACC directory
3. Install Conda (only necessary the first time)
4. Create Conda environment either from set of packages defined in a .yml file or by manually installing them
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
    - answer **Yes** to the SSH security question if it appears
- In the **PuTTY** terminal
    - enter your TACC user id after the **"login as**:**"** prompt, then **Enter**
    - enter the password associated with your TACC account
    - provide your 2-factor authentication code  

**Note that you won't see the cursor move while you type your password or token into the putty terminal. Simply type and then hit enter.**

*The putty instructions were copied from [Getting Started at TACC Wiki](https://wikis.utexas.edu/display/CoreNGSTools/Getting+started+at+TACC) by Anna M Battenhouse.*



## 2. Navigate to proper TACC directory
Those new to working in the terminal / TACC should review the [Getting Started at TACC](https://wikis.utexas.edu/display/CoreNGSTools/Getting+started+at+TACC) documentation.

When accessing TACC systems, you are initially located in the $HOME directory.  TACC has three files systems: $HOME, $WORK, and $SCRATCH.  A fuller discussion of these systems are available on the [TACC Lonestar6 documentation](https://docs.tacc.utexas.edu/hpc/lonestar6/#files).  For this exercise we will work on $SCRATCH.

A few unix commands that are helpful to know are listed below, while a more extensive unix cheat sheet is available [here](https://mally.stanford.edu/~sr/computing/basic-unix.html)

* pwd: display current working directory
* ls: list files
* cd: change directory
* mkdir {name}: make a new directory called {name}

If you type `pwd` in the terminal, you will see the actual path to your $HOME directory. This should look something like `/home1/01234/username`. For moving between directories, TACC provides the following alias commands:
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
Note: the installation may take a while.

Make sure to close out your terminal / putty after installing conda and reopen to have conda available for use. 


## 4. Create Conda environment

### 4a. Transfer .yml file into directory
If you don't already have a directory and files for your project, use `mkdir new_directory_name` to create a new directory and `cd new_directory_name` to move into it. Use the TACC data transfer protocols ([TACC documentation](https://docs.tacc.utexas.edu/basics/datatransferguide/)) to copy a .yml file from your local computer into your new directory.  

In the TACC terminal, type 'pwd' and copy this information - this is the destination you are trying to save your file *to*.

On your local computer, open the command prompt and either navigate into the directory holding the .yml file. 

use the scp command to transfer the file from your local computer to TACC.  This command will look something like the following, replacing 'environment.yml' with the name of your yml file, 'username' with your TACC username and '/scratch/01234/username/new_directory_name' with the output from `pwd`:

    scp environment.yml username@ls6.tacc.utexas.edu:/scratch/01234/username/new_directory_name

### 4b. Create Conda environment 
To create a pre-set environment **from a .yml file**: 
Activate the base conda environment with

    conda activate
    
Navigate into the directory where your environment.yml file is located and create a new environment by running the following commands - replacing 'my-new-env' with a name of your no choice (but no spaces!) and 'environment.yml' with the filename of your environment .yml file. 

    conda env create --name my-new-env --file environment.yml 
    

To create an environment **from scratch**: 
Simply follow standard [Conda environment management](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) protocols. 

### 4b. Add Conda environment as a kernel in Jupyter
To add the new environment as a kernel in your Jupyter notebook you first activate your enfironment, install ipykernel in the activated environment and, then create a Jupyter kernel from your env:

    conda activate my-new-env
    pip install ipykernel
    python -m ipykernel install --user --name my-new-env --display-name "Python (My New Env)"

## 5. Working with Jupyter Notebook at TACC
To run a Jupyter Notebook at TACC you will request a job via the [TACC Analysis Portal](https://tap.tacc.utexas.edu). Once logged in, you should see a dashboard with a 'Submit New Job' option in the top left. If you can log in but don't get this option you likely don't have an active allocation.

Request a Jupyter Notebook job using the following default elections - update these if appropriate for your project:
* System: Lonestar6
* Application: Jupyter notebook
* Project: project sponsoring the work / which allocation you want to use
* Queue: development 

Submit job request and wait until it launches. When the message says your session is running, simply click the green 'Connect' button or copy and paste the URL into the browser of your choice.

Once Jupyter launches you can either start a new notebook, or use file transfer protocols to bring an existing notebook into your files for use.

### Adding additional packages
If you find your use cases requires additional packages, simply search [Anaconda.org](https://anaconda.org/) for the package and isntallation code, then install them from the terminal.  Then restart the package in Jupyter to make the package code available.
