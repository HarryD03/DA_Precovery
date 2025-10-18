To follow the methodology of the "Orbit Determination Under Uncertainty: Differential Algebra Applications for Apophis Precovery" Masters Thesis:

1) Open the main_scripts directory
2) Run main_final.py to complete Differential Algebra based Orbit Determination and Propagation. This produces .pkl files, stored within the Simulation_Data directory, which the query.py script reads
3) Run query.py to obtain the CADC, ESO and SSOIS. This copies and appends the number of catalogue images and size of the search area to the .pkl files. The files are stored under Query_Data directory and is read, and therefore is required, for the plots.py script
4) Run Plots.py to obtain the Figures used for analysis. This generates .png files stored within the Plots directory

Note: As the scripts have been executed in order to complete the analysis, all directories are populated with the required .pkl and .png files. If you execute the sequence above, you will overwrite these files and they will not be the .pkl/.png files used in the Thesis.
