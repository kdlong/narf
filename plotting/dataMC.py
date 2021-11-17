import matplotlib
import matplotlib.pyplot as plt
import uproot
import mplhep as hep
import numpy as np
import argparse
import os
import logging
import narf
import shutil
from datasets import samplegroups2016
from datasets import datasets2016
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", type=str, help="Input root file(s)", required=True)
parser.add_argument("-s", "--samples", nargs="*", type=str, help="Samples to plot", required=True)
parser.add_argument("-b", "--hist", type=str, help="Distribution to plot", required=True)
parser.add_argument("-p", "--outputPath", type=str, default=os.path.expanduser("~/www/PlottingResults"))
parser.add_argument("-o", "--outputFolder", type=str, default="Test")
parser.add_argument("-a", "--append", type=str, default="", help="string to append to output file name")
parser.add_argument("-r", "--ratioRange", type=float, nargs=2, default=[0.9, 1.1])
parser.add_argument("-x", "--xRange", type=float, nargs=2, default=[])
parser.add_argument("--xlabel", type=str)
parser.add_argument("--scaleleg", type=float, default=1.)
parser.add_argument("--noHtml", action='store_true', help="Don't make html in output folder")
args = parser.parse_args()

plt.style.use([hep.style.ROOT])
cmap = matplotlib.cm.get_cmap('tab20b')    
# For a small number of clusters, make them pretty
all_colors_ = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
all_colors_.insert(0, 'black')

def plotHists(bins, datasets, central, ratioRange=[0.9, 1.1], width=1, xlim=[],
        xlabel="", scaleleg=1.):
    fig = plt.figure(figsize=(8*width,8))
    ax1 = fig.add_subplot(4, 1, (1, 3)) 
    ax2 = fig.add_subplot(4, 1, 4) 
    centralHist = datasets[central]["hist"]
    for name, dataset in datasets.items():
        hist = dataset["hist"]
        hep.histplot(hist, bins, **dataset['style'], ax=ax1)
        ratio = np.divide(hist, centralHist, out=np.zeros_like(hist), where=centralHist!=0)
        if dataset['style']['histtype'] == 'fill':
            dataset['style']['histtype'] = 'step'
        hep.histplot(ratio, bins, **dataset['style'], ax=ax2)
    ax2.set_ylim(ratioRange)
    ax1.set_xticklabels([])
    if xlim:
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
    ax1.legend(prop={'size' : 20*scaleleg})
    ax2.set_xlabel(xlabel)
    ax1.set_ylabel("Events/bin")
    return fig

def compareDistributions():
    samples = args.samples

    outputPath = "/".join([args.outputPath, args.outputFolder, "plots"])
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    mchists = {}
    datahists = {}

    groups = samplegroups2016.allSampleGroups()
    datasets = {x.name : x for x in datasets2016.allDatasets()}

    lumi = 0
    rtfile = uproot.open(args.input_file)    
    hists = {}
    
    for samplename in samples:
        sample = groups[samplename]

        hist = None
        data = False
        for hn in sample.members:
            histname = "/".join([hn, args.hist])
            h,bins = rtfile[histname].to_numpy()
            hist = h if not hist else np.sum(hist, h)
            if datasets[hn].is_data:
                data = True
                l,_ = rtfile["/".join([hn, "lumi"])].to_numpy()
                lumi += np.sum(l)
            elif datasets[hn].xsec is not None:
                s,_ = rtfile["/".join([hn, "sumweights"])].to_numpy()
                sumw = np.sum(s)
                hist = hist*datasets[hn].xsec/sumw

        hists[samplename] = {"hist" : hist, "style" : sample.drawproperties, "data" : data}

    for h in hists.values():
        if not h["data"]:
            h["hist"] = h["hist"]*lumi*1000

    fig = plotHists(bins, hists, samples[0], ratioRange=args.ratioRange, 
            xlim=args.xRange, xlabel=args.xlabel, scaleleg=args.scaleleg,
            width=(1 if "unrolled" not in args.hist else 2))
    append = []
    if args.append:
        append = append + [args.append]
    outfile = "%s/%s.pdf" % (outputPath, args.hist+"_".join(append))

    fig.savefig(outfile)
    fig.savefig(outfile.replace(".pdf", ".png"))
    logging.info(f"Wrote output file {outfile}")

    html_temp = "/".join(["plotting/Templates", "index.php"])
    html = "/".join([outputPath, "index.php"])
    if not os.path.isfile(html):
        shutil.copy(html_temp, html)

def main():
    compareDistributions()

if __name__ == "__main__":
    main()
