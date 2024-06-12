import ROOT
from JES_BalanceFitter import JES_BalanceFitter
import numpy as np
import pandas as pd
import pickle

import scipy as scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import mplhep as hep

import os

def resolutionFunc(x,a,b,c):
    return np.sqrt((a/np.sqrt(x))**2+(b/x)**2+c**2)
    
def resolutionFunc2(x,a,b,c):
    return np.sqrt((a/x)+(b/x)**2+c)

def sliceAndDice(listOfRootFilePaths,slices,slicingAxis):
    projectionAxis = "y"
    projectionRebinWidth = 2
    responseAxis = "x"
    nSigmaForFit = 1.3
    fitOptString = "RESQ"

    # Loop over the file paths
    for rootFilePath in listOfRootFilePaths:
        inFile = ROOT.TFile.Open(rootFilePath) # Open root file containing 3D histograms
        
        # Loop over the slices
        for currentSlice in slices:
            listOfKeys = inFile.GetListOfKeys() # Get the 3D histogram names from the root file

            dictionaryList=[] # Initialize an empty list to house the dictionaries, one per slice
            
            # Loop over the 3D Histogram names
            for key in listOfKeys:
                TH3Name = key.GetName() # Get the actual name as a string
                
                if TH3Name[0:9] != "scaled_h_": # Skip the #D histograms which are not scaled 
                    continue
                elif inFile.Get(TH3Name).GetEntries()==0.0: # Print a warning if a 3D histogram is empty
                    print("WARNING:",TH3Name," is empty!")
                    continue
                else: # If everything is good, append an empty pandas series to the dictionary list
                    dictionaryList.append(pd.Series({
                                         "x"             :[],
                                         "y"             :[],
                                         "xError"        :[],
                                         "yError"        :[],
                                         "sigma"         :[],
                                         "sigmaError"    :[],
                                         "sigmaOverY"    :[],
                                         "sigmaOverYError"    :[],
                                         "fitAmplitude"  :[],
                                         "fitMin"        :[],
                                         "fitMax"        :[],

                                         "TH1BinEdges"   :[],
                                         "TH1BinEntries" :[],
                                         "TH1BinErrors"  :[],
                                       },
                                          name=TH3Name))
            # Define a path and name for the dataframe based on the slice range
            dfPath = rootFilePath.split(".")[0]+"_"+slicingAxis+"["+str(currentSlice[0])+","+str(currentSlice[1])+"]"
            print(dfPath)
            df = pd.DataFrame(dictionaryList) # Create the dataframe form the dictionary list

            # Within the current slice, loop over the 3D histogram names in the empty dataframe
            for TH3Name in df.index:
                inTH3 = inFile.Get(TH3Name) # Retreive the current 3D histogram form the root file
                h3D = inTH3.Clone() # Clone it ot keep it intact

                #Set fitting options for the JES_BalanceFitter tool
                JESBfitter = JES_BalanceFitter(nSigmaForFit)
                JESBfitter.SetGaus()
                JESBfitter.SetFitOpt(fitOptString)

                # Set the 3D histogram range to correspond to the desired slice range
                if slicingAxis == "y":
                    h3D.GetYaxis().SetRangeUser(currentSlice[0], currentSlice[1])     
                elif slicingAxis == "z":
                    h3D.GetZaxis().SetRangeUser(currentSlice[0], currentSlice[1]) 

                # Project the 3D histogram with the sliced axis range into a 2D histogram
                h2D=h3D.Project3D(responseAxis + projectionAxis)

                # Rebin the 2D histogram x-axis according to the desired rebinningFactor
                h2D.RebinX(projectionRebinWidth)

                # Loop over the bins in the 2D histogram x-axis
                for currentRebinnedBin in range(1, h2D.GetNbinsX()+1): # Histograms start at bin 1, plus one to include last bin
                    # Name of the 1D projection
                    projName = "slice"+str(currentSlice[0])+"to"+str(currentSlice[1])+"_projectionBin"+str(h2D.GetXaxis().GetBinLowEdge(currentRebinnedBin))+"to"+str(h2D.GetXaxis().GetBinUpEdge(currentRebinnedBin))
                    
                    # Project the current 2D histogram bin into a 1D histogram
                    h1D=h2D.ProjectionY(projName, currentRebinnedBin, currentRebinnedBin)

                    # If the current 2D bin is empty skip it
                    if h1D.GetEntries() == 0:
                        #print("empty 1D hist, skipping!")
                        continue

                    # Set the fit limits based on the 1D histogram properties
                    fitMax = h1D.GetMean() + nSigmaForFit * h1D.GetRMS()
                    fitMin = h1D.GetMean() - nSigmaForFit * h1D.GetRMS()

                    # Obtain fit using JES_BalanceFitter           
                    JESBfitter.Fit(h1D, fitMin, fitMax)
                    fit = JESBfitter.GetFit()
                    histFit = JESBfitter.GetHisto()
                    Chi2Ndof = JESBfitter.GetChi2Ndof()

                    # Initialize empty lists to hold the values of each 1D histogram.
                    # There will be as many 1D histograms as Y axis bins of the 3D histogram
                    binEdges=[]
                    binEntries=[]
                    binErrors=[]
                    
                    # Loop over the response bins in the 1D histogram
                    for i in range(1, h1D.GetNbinsX()+1):# Plus one to include last bin
                      binEdges.append(h1D.GetXaxis().GetBinLowEdge(i)) # Get the current bin edge for plotting later
                      binEntries.append(h1D.GetBinContent(i)) # Get the number of entries in the bin for plotting later
                      binErrors.append(h1D.GetBinError(i)) # Get the bin error to get the width of the bin... For plotting later
                    binEdges.append(h1D.GetXaxis().GetBinUpEdge(h1D.GetNbinsX()))# Append the right most edge of the last bin

                    # Append all the lists for this 1D histogram to the list of lists in the dataframe.
                    # One dataframe per slice, per root file path
                    
                    # Append all the values of the 1D histogram gaussian fit to the lists of the current data frame series
                    # There are ~4 series per data frame, 1 dataframe per slice, per root file path
                    df["x"].loc[TH3Name].append(float(h2D.GetXaxis().GetBinCenter(currentRebinnedBin)))
                    df["y"].loc[TH3Name].append(float(fit.GetParameter(1)))
                    df["xError"].loc[TH3Name].append(float((h2D.GetXaxis().GetBinWidth(currentRebinnedBin)/2.0)))#half bin width
                    df["yError"].loc[TH3Name].append(float(fit.GetParError(1)))
                    df["sigma"].loc[TH3Name].append(float(fit.GetParameter(2)))
                    df["sigmaError"].loc[TH3Name].append(float(fit.GetParError(2)))
                    try:
                        df["sigmaOverY"].loc[TH3Name].append(float(fit.GetParameter(2) / float(fit.GetParameter(1))))
                        df["sigmaOverYError"].loc[TH3Name].append(np.sqrt((fit.GetParError(2)/fit.GetParameter(2))**2+(fit.GetParError(1)/fit.GetParameter(1))**2))
                    except:
                        df["sigmaOverY"].loc[TH3Name].append(0)
                        df["sigmaOverYError"].loc[TH3Name].append(0)
                    
                    # Append the lists of 1D histogram info to the list of lists in the dataframe data series
                    df["TH1BinEdges"].loc[TH3Name].append(binEdges)
                    df["TH1BinEntries"].loc[TH3Name].append(binEntries)
                    df["TH1BinErrors"].loc[TH3Name].append(binErrors)
            
            df.to_pickle(dfPath+".pickle") # Save the dataframe to disk as pickle. One dataframe per slice per root file path

    inFile.Close() # Close the root file
    
# Return wether the root file comes from Monte Carlo simulation (MC) or data
def isItDataOrMC(rootFilePath):
    if("mc" in rootFilePath.split("/")[-1]):
        return "MC"
    elif("data" in rootFilePath.split("/")[-1]):
        return "Data"
    else:
        "Cannot determine if it is Data or MC"
    
# Return wether the 3D histogram is from Online or Offline objects
def isItOnlineOrOffline(dfIndex):
    if("Online" in dfIndex):
        return "Online"
    elif("Offline" in dfIndex):
        return "Offline"
    else:
        return "Cannot determine if it is Online or Offline"
        
def plotResponse(listOfRootFilePaths,slices,slicingAxis):
    xAxisUnit = "mjj" # Usually mjj, pT, or energy. Essentaily the variable name of the response
    xAxisLabel = "Truth $"+xAxisUnit[0]+"_{"+xAxisUnit[1:].upper()+"}$ [GeV]"
    yAxisLabel = "$"+xAxisUnit[0]+"_{"+xAxisUnit[1:].upper()+"}$ Response"
    skipEnergyScales = ["SmearedMomentum"] # A list of energy scale names to skip
    # The legend titles are lists of strings, where parameters from the code are input
    # between and after the different title entries
    responseLegendTitle = ["Energy Scale Response","$\eta = $",]
    resolutionLegendTitle = ["Online vs. Offline Resolution","$\eta = $",]
    resolutionRatioLegendTitle = ["Resolution Ratio:",]
    # Set the x and y-axis limits
    xLimits = (100,5000)
    yLimits = (0.5,1.2)

    # Loop over the file paths
    for rootFilePath in listOfRootFilePaths:
        # Loop over the slices
        for currentSlice in slices:
            # Initialize figure 185 mm wide, wiht a 800:600 widht:height ratio
            f, ax = plt.subplots(figsize=(18.3*(1/2.54), 13.875*(1/2.54)))
            
            # Define a path and name for the dataframe based on the slice range
            dfPath = rootFilePath.split(".")[0]+"_"+slicingAxis+"["+str(currentSlice[0])+","+str(currentSlice[1])+"]"
            df = pd.read_pickle(dfPath+".pickle") # Read the appropriate dataframe. One per slice, per file path
            
            # Get data information based on root file path and TH3 name
            DataOrMC = isItDataOrMC(rootFilePath)
            OnlineOrOffline = isItOnlineOrOffline(df.index[0])
            campagin = rootFilePath.split("/")[-1].split(".")[0].split("_")[1]
            
            # Define markers and colors for plot data series
            markers = ["o","^",">","v","<"]
            colors = ["black","crimson","darkorange","dodgerblue","forestgreen"]
            i=0 # Set an iterator for the markes and colors
            
            # Iterate over TH3s in data frame, in the reverse order
            for TH3Name in reversed(df.index):
                skipping = False # Set upp skipping flag
                if xAxisUnit not in TH3Name: # Skip pT plots if we are interested in mjj. For example
                    skipping = True
                # A quick loop over the list of skipped energy scales
                for skippedEnergyScales in skipEnergyScales: # Skip if the 3D histogram name is in the list to skip
                    if skippedEnergyScales in TH3Name:
                        skipping = True
                if skipping: continue
                    
                # Get data information based on TH3 name
                numeratorEnergyScale = TH3Name.split("_-_")[1].split("_")[0].split("-")[0]
                denominatorEnergyScale = TH3Name.split("_-_")[1].split("_")[2].split("-")[1]

                # Assign data from dataframe for the current 3D histogram name
                x=df["x"].loc[TH3Name]
                y=df["y"].loc[TH3Name]
                x_error=df["xError"].loc[TH3Name]
                y_error=df["yError"].loc[TH3Name]

                # Plot data
                ax.errorbar(x, y, yerr=y_error, xerr=x_error,
                            linestyle='None',
                            marker=markers[i],
                            color=colors[i],
                            markersize=2,
                            linewidth=0.5,
                            label=numeratorEnergyScale+"$_{"+xAxisUnit+"}$"+" / "+denominatorEnergyScale.capitalize()+"$_{"+xAxisUnit+"}$",
                           )
                i+=1 # Increment color and marker iterator

            # Add legend
            leg = ax.legend(borderpad=0.5, loc=1, ncol=2, frameon=True,facecolor="white",framealpha=1)
            leg._legend_box.align = "left"
            leg.set_title(campagin+" "+OnlineOrOffline+" "+responseLegendTitle[0]+"\n"+responseLegendTitle[1]+"["+str(currentSlice[0])+","+str(currentSlice[1])+"]")

            # Set limits and labels
            ax.set_xlim(xLimits)
            ax.set_ylim(yLimits)

            # Set log scale
            ax.set_xscale("log")

            # Set axis labels
            ax.set_xlabel(xAxisLabel, fontsize=14, ha='right',x=1.0)
            ax.set_ylabel(yAxisLabel, fontsize=14, ha='right', y=1.0)


            # Add grid and custom tick markers
            ax.grid(True)
            tickList = [1,2,3,4,5,6,7,8,9,
            10,20,30,40,50,60,70,80,90,
            100,200,300,400,500,600,700,800,900,
            1000,2000,3000,4000,5000,6000,7000,8000,9000,
            10000]
            ax.set_xticks(tickList[tickList.index(xLimits[0]):tickList.index(xLimits[1])])
            ax.set_xticklabels(tickList[tickList.index(xLimits[0]):tickList.index(xLimits[1])])
            plt.xticks(rotation=45)

            # Add ATLAS label
            hep.atlas.text("Internal",ax=ax)

            # Use tight layout
            plt.tight_layout()
            
            # Save plot as .pdf
            f.savefig("output/"+campagin+"_"+OnlineOrOffline+"_"+xAxisUnit+"_"+"response"+"_"+"eta"+"["+str(currentSlice[0])+","+str(currentSlice[1])+"]"".pdf")

def plotResolution(listOfRootFilePaths,slicingAxis,slices):
    listOfComparisonPaths = [
                    listOfRootFilePaths[0],
                    listOfRootFilePaths[1],
                    ]
    energyScale = "GSC"
    xAxisUnit = "mjj"
    
    resolutionLegendTitle = ["Online vs. Offline Resolution","$\eta = $",]
    
    rootFilePath = listOfRootFilePaths[0]
    TH3Name = "scaled_h_-_GSC-Online_over_-truth_-_mjj_-_eta"
    currentSlice=[-2.8,2.8]
    dfPath = rootFilePath.split(".")[0]+"_"+slicingAxis+"["+str(currentSlice[0])+","+str(currentSlice[1])+"]"
    df= pd.read_pickle(dfPath+".pickle")
    series = df.loc[TH3Name]
    # Assign data from dataframe for the current 3D histogram name
    x = series['x']
    y = series['y']

    linspace=np.linspace(x[0],x[-1],num=10000) # Define a linspace to be used for plotting the fit
    color_list=["red","black"] # Define the colors used by the two data series

    #guesses = [[0.937760293466364, 11.803208703991855, 0.13229820524075314],[0.9208120809096343, -9.716149218976808, 0.13347830553680345]]
    values = [[]]
    # Loop over the slices
    for currentSlice in slices:
        # Initialize figure 185 mm wide, wiht a 800:600 widht:height ratio
        f, axisList = plt.subplots(2,1,figsize=(18.3*(1/2.54), 13.875*(1/2.54)),sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Initialize empty lists of lists to hold the two data series.
        # This is later used for a ratio plot
        x_list = []
        y_list = []
        
        # Loop over the two file paths
        for i,rootFilePath in enumerate(listOfComparisonPaths):
            # Define a path and name for the dataframe based on the slice range
            dfPath = rootFilePath.split(".")[0]+"_"+slicingAxis+"["+str(currentSlice[0])+","+str(currentSlice[1])+"]"
            df= pd.read_pickle(dfPath+".pickle") # Read the appropriate dataframe. One per slice, per file path
            
            # Get data information based on root file path and TH3 name
            DataOrMC = isItDataOrMC(rootFilePath)
            OnlineOrOffline = isItOnlineOrOffline(df.index[0])

            # Only select the data series / 3D histogram following our earlier specifications
            TH3Name = df[[((energyScale in s) and (xAxisUnit in s)) for s in df.index]].index[0]
            
            # Assign data from data series
            x=df["x"].loc[TH3Name]
            x_list.append(x) # Append list for the ratio plot
            y=df["sigmaOverY"].loc[TH3Name] # The resolution is defined as the error of the response fit mean over the response mean
            y_list.append(y) # Append list for the ratio plot
            x_error=df["xError"].loc[TH3Name]
            y_error=df["sigmaOverYError"].loc[TH3Name]

            # Plot the resolution
            axisList[0].errorbar(x, y, yerr=y_error, xerr=x_error,
                        linestyle='None',
                        marker="o",
                        color=color_list[i],
                        markersize=2,
                        linewidth=0.5,
                        label=rootFilePath.split("/")[-1].split("_")[1]+" "+OnlineOrOffline)

            # Fit the resolution with the resolution function
            popt_resolutionFunc, pcov_resolutionFunc = curve_fit(resolutionFunc, x, y, sigma=y_error)

            print(currentSlice,rootFilePath.split("/")[-1].split("_")[1]+" "+OnlineOrOffline)
            #print(popt_resolutionFunc)
            print(f"[{popt_resolutionFunc[0]**2},{popt_resolutionFunc[1]},{popt_resolutionFunc[2]**2}]")
            
            # Plot the fit
            axisList[0].plot(linspace, resolutionFunc(linspace,*popt_resolutionFunc), lw=1, label=rootFilePath.split("/")[-1].split("_")[1]+OnlineOrOffline+" Fit",color=color_list[i])

        # Add the fit from 2016 Online resolution
        axisList[0].plot(linspace[240:], resolutionFunc(linspace[240:],0.27,10.6,0.039), lw=1, label=r'2016 Online Fit',color="blue")
        axisList[0].plot(linspace[:240], resolutionFunc(linspace[:240],0.27,10.6,0.039), lw=1, label=r'2016 Extrapolated Fit',color="blue",linestyle='dashed')

        #axisList[0].plot(linspace, resolutionFunc2(linspace,0.7733344544738501,11.803208703991855,0.00030634853677219924), lw=1, label=r'test Online',color="grey")
        #axisList[0].plot(linspace, resolutionFunc2(linspace,0.718925741688585,9.716149218976808,0.0003174261774109307), lw=1, label=r'test Offline',color="grey")

        # Legend
        leg = axisList[0].legend(borderpad=0.5, frameon=True, loc=1,ncol=2,facecolor="white",framealpha=1.0)
        leg.set_title(resolutionLegendTitle[0]+"\n"+resolutionLegendTitle[1]+"["+str(currentSlice[0])+","+str(currentSlice[1])+"]")
        leg._legend_box.align = "left"

        axisList[0].set_ylabel(r'$m_{jj}$ Resolution', fontsize=14, ha='right', y=1.0)
        
        hep.atlas.text("Simulation Internal",ax=axisList[0])

        # Set limits and labels of the top plot
        axisList[0].set_xlim(100,5000)
        axisList[0].set_xscale("log")
        axisList[0].set_ylim(0.00,0.2)
        axisList[0].grid()

        # Plot the ratio of Online resolution over Offline resolution
        axisList[1].errorbar(np.array(x_list[0]),np.array(y_list[0])/np.array(y_list[1]),linestyle='None',marker="o",color="black",markersize=2,linewidth=0.5,label="Ratio")

        # Set limits and grid to the ratio plot
        axisList[1].set_ylim(0.98,1.2)
        axisList[1].grid()
        
        # Add 10% graphic lines
        axisList[1].plot( [100,5000], [1.0,1.0],color="red",lw=1)
        axisList[1].plot( [100,5000], [1.1,1.1],color="red",lw=1)

        # Add custom tick markers
        axisList[1].set_axisbelow(True)
        axisList[1].set_xticks([100,400,1000,5000])
        axisList[1].set_xticklabels([100,400,1000,5000])

        # Add axis labels
        axisList[1].set_xlabel(r"Truth $m_{jj}$ [GeV]", fontsize=14, ha='right', x=1.0)
        axisList[1].set_ylabel(r"Online/Offline", fontsize=14)

        # Add ATLAS label
        hep.atlas.text("Internal",ax=axisList[0])
        
        # Use tight layout
        plt.tight_layout()
        
        # Save plot as .pdf
        f.savefig("output/"+rootFilePath.split("/")[-1].split("_")[1]+"_"+"onlineVsOffline"+"_"+slicingAxis+"["+str(currentSlice[0])+","+str(currentSlice[1])+"]"+".pdf")
        
def getBinning(listOfRootFilePaths,slicingAxis,slices):
    rootFilePath = listOfRootFilePaths[0]
    TH3Name = "scaled_h_-_GSC-Online_over_-truth_-_mjj_-_eta"
    currentSlice=[-2.8,2.8]
    dfPath = rootFilePath.split(".")[0]+"_"+slicingAxis+"["+str(currentSlice[0])+","+str(currentSlice[1])+"]"
    df= pd.read_pickle(dfPath+".pickle")

    x=df["x"].loc[TH3Name]
    y=df["sigmaOverY"].loc[TH3Name]
    x_error=df["xError"].loc[TH3Name]
    y_error=df["sigmaOverYError"].loc[TH3Name]
    popt_resolutionFunc, pcov_resolutionFunc = curve_fit(resolutionFunc, x, y, sigma=y_error)

    bins1 = [] # Initialize an empty list to hold the bottom half of the list of bin edges
    currentEdge = 531 # Define the starting bind edge

    # Loop until the bin edge is less than 100
    while (currentEdge > 100):
        # Calculate a new bin width by evaluating the resolution fit at the current edge
        currentBinwidth = int(round( resolutionFunc(currentEdge, *popt_resolutionFunc)*currentEdge ))
        currentEdge -= currentBinwidth # Define a new current edge by subtraction
        bins1.append(currentEdge) # Append the new current edge

    bins2 = [] # Initialize an empty list to hold the top half of the list of bin edges
    currentEdge = 531# Define the starting bind edge

    # Loop until the bin edge is less than 100
    while (currentEdge < 5000):
        # Calculate a new bin width by evaluating the resolution fit at the current edge
        currentBinwidth = int(round( resolutionFunc(currentEdge, *popt_resolutionFunc)*currentEdge ))
        currentEdge += currentBinwidth # Define a new current edge by addition
        bins2.append(currentEdge) # Append the new current edge
        
    bins=bins1+bins2
    bins.append(531)
    bins.sort()

    print("Bin edges: \n",bins)

    import json
    # Create a dictionary of different types of binning, one containing our binning
    mjjBins = {
        'TLAdefault':   bins,
        'TLAlowMu'  :   []
    }

    outfile = open('output/mjjBins.json','w') # Open a JSON file
    # Write the dictionary to the JSON file
    outfile.write(json.dumps(mjjBins, sort_keys=True, indent=4, separators=(',', ': ')))
    outfile.close()
    
def main():
    pathList=os.getcwd()
    path=pathList
    listOfRootFilePaths = [path+"/data/v14/merged_mc16d_mjj_v14.root", #mc16d Online
                           path+"/data/v15/merged_mc16d_mjj_v15.root", #mc16d Offline
                           path+"/data/v18/merged_mc16a_mjj_v18.root", #mc16a Offline
                           path+"/data/v19/merged_mc16a_mjj_v19.root", #mc16a Online
                          ]
    slices = [[-2.8,2.8],[-0.6,0.6]]
    slicingAxis = "z"
    
    #sliceAndDice(listOfRootFilePaths,slices,slicingAxis)
    #plotResponse(listOfRootFilePaths,slices,slicingAxis)
    plotResolution(listOfRootFilePaths,slicingAxis,slices)
    getBinning(listOfRootFilePaths,slicingAxis,slices)

if __name__ == "__main__":
    main()
