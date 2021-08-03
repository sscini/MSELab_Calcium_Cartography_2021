![](simulation_Demo_Output.gif)

This is a repository for the associated Python code for the manuscript titled: From spikes to intercellular waves: tuning intercellular Ca<sup>2+</sup> signaling dynamics modulates organ size control. The code was built by [Dr. Ramezan Paravi Torghabeh](https://www.linkedin.com/in/ramezan-paravi-torghabeh-phd-b432baa0/). The code was then later updated and adapted for creation of the figures in the manuscript by [Francisco Huizar](https://www.linkedin.com/in/francisco-huizar-82bb1a127/). Experimental work and validation was led by [Dharsan Soundarrajan](https://scholar.google.com/citations?user=AWv4OiIAAAAJ&hl=en). This work was done within the [Multicellular Systems Engineering Lab](http://sites.nd.edu/zartmanlab/) at the University of Notre Dame. Please direct any questions to the lab principal investigator, [Dr. Jeremiah Zartman](http://sites.nd.edu/zartmanlab/contacts/).

All code was done using [Jupyter](https://jupyter.org/) notebooks where all code was run and the visible outputs are saved in the iPython notebooks. The code is available to be downloaded and run after installation of the latest version of [Jupyter](https://jupyter.org/) and [Python](https://www.python.org/). All simulation outputs were coded to be saved in the [simulationResults](https://github.com/fjhuizar/MSELab_Calcium_Cartography_2021/tree/master/simulationResults) folder, and are labeled according to figure numbers. Use of [Epitools](https://epitools.ausvet.com.au/) was done to create a 2-D geometrically accurate representation of the Drosophila wing imaginal disc. The corresponding folder for this is in the epitools (Figure S1 Creation) folder [here](https://github.com/fjhuizar/MSELab_Calcium_Cartography_2021/tree/master/epitools%20(Figure%20S1%20Creation)).

# Sample demonstration of the code
Anyone can run a sample of the code used in the manuscript by using the [Google Colaboratory](https://research.google.com/colaboratory/) notebook located [here](https://colab.research.google.com/drive/1Md-OkLtQ3TdeEt3aBj__tua_hnYCPsdj?usp=sharing). Instructions to run the demonstration code are found on the linked page. A full demonstration simulation will take approximately 25 minutes to run on the Google Colaboratory virtual machine, and simulations can be run faster on a personal machine (see instructions below for use on a personal machine). In the demonstration page, you are able to choose between the following outputs that correspond to Figure 2 in the manuscript:
- Single cell calcium spikes
- Intercellular calcium transients
- Intercellular calcium waves
- Global fluttering calcium activity

# Instructions to run the code on a personal machine
1. You can download all of the associated code [here](https://github.com/fjhuizar/MSELab_Calcium_Cartography_2021/archive/master.zip)
  - Each notebook is labeled corresponding to its associated figure in the main text of the manuscript.
  - **NOTE:** The full folder is approximately 2+ GB of data as it contains the pre-run simulation output pictures and videos.
2. Extract the downloaded .zip folder where you would like to run the code using Jupyter
![](/Installation_Instructions/Step_Two.png)
3. Navigate to and open the extracted folder
![](/Installation_Instructions/Step_Three.png)
4. Within the now extracted folder extract the 'geometry.zip' folder as well
![](/Installation_Instructions/Step_Four.2.png)
5. Open and run Jupyter Notebook
6. Navigate in the Jupyter Notebook host to where the extracted files are
![](/Installation_Instructions/Step_Six.png)
7. Open a new, untitled notebook in Jupyter
![](/Installation_Instructions/Step_Seven.png)
8. Install all necessary Python packages using [pip install requirements.txt](https://pip.pypa.io/en/stable/cli/pip_install/).
  - This is done by entering the following into a cell:
  ```
  !pip install -r /path/to/requirements.txt
  ```
  - **NOTE**: 'path/to/' should be changed to the correct path where the code was extracted to
![](/Installation_Instructions/Step_Eight.png)
9. Use Jupyter to open each individual notebook to run the simulations as desired

# Code for repeated simulations
You can find the code and corresponding outputs for repeated simulations (with different random number generator seeds) for the manuscript's main figures in the following locations:
- [Figure 2 repeated simulations](https://github.com/fjhuizar/MSELab_Calcium_Cartography_2021/tree/master/simulationResults/Figure_2_Repeated_Simulations)
- [Figure 3 repeated simulations](https://github.com/fjhuizar/MSELab_Calcium_Cartography_2021/tree/master/simulationResults/Figure_3_Repeated_Simulations)
- [Figure 4 repeated simulations](https://github.com/fjhuizar/MSELab_Calcium_Cartography_2021/tree/master/simulationResults/Figure_4_Repeated_Simulations)

# Code for model sensitivity to parameters
You can find the code and corresponding outputs for the sensitivity analysis of parameters and their effects on a baseline intercellular calcium wave [here](https://github.com/fjhuizar/MSELab_Calcium_Cartography_2021/tree/master/simulationResults/Sensitivity_Analysis_Simulations). This code corresponds to the supplementary information figures.

# Acknowledgements
We would like to thank [Trent Robinett](https://www.linkedin.com/in/trent-robinett-5a8979161/) for his contributions towards completion of the manuscript. In particular, we would like to highlight his efforts in carrying out experiments alongside [Dharsan Soundarrajan](https://scholar.google.com/citations?user=AWv4OiIAAAAJ&hl=en).

The work in this manuscript was supported by NIH Grant R35GM124935 and NSF Award CBET-1553826. The authors gratefully acknowledge the Notre Dame Center for Research Computing (CRC) for providing computational facilities in addition to [members of the Zartman Lab](http://sites.nd.edu/zartmanlab/members/) for helpful discussions and feedback.

# Repository last updated: August 03, 2021 6:00PM EST
