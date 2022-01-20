import narf
import logging
import subprocess
import glob
import pathlib

lumicsv = f"{pathlib.Path(__file__).parent.parent}/data/bylsoutput.csv"
lumijson = f"{pathlib.Path(__file__).parent.parent}/data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"

def loadFiles(files, maxfiles):
    return files[:len(files) if maxfiles < 0 else maxfiles]

def allDatasets(maxfiles=-1):
    data_files = loadFiles(data_files_, maxfiles)
    zmc_files = loadFiles(zmc_files_, maxfiles)
    wplus_files = loadFiles(wplus_files_, maxfiles)
    wminus_files = loadFiles(wminus_files_, maxfiles)
    data = narf.Dataset(name = "dataPostVFP",
                        filepaths = data_files,
                        is_data = True,
                        lumi_csv = lumicsv,
                        lumi_json = lumijson)

    zmc = narf.Dataset(name = "ZmumuPostVFP",
                    filepaths = zmc_files,
                    is_data = False,
                    xsec = 1990.5,
                    )

    wplus = narf.Dataset(name = "WplusmunuPostVFP",
                    filepaths = wplus_files,
                    is_data = False,
                    xsec = 11572.19,
                    )

    wminus = narf.Dataset(name = "WminusmunuPostVFP",
                    filepaths = wminus_files,
                    is_data = False,
                    xsec = 8562.66,
                    )

    return [data, zmc, wplus, wminus]
    #return [zmc, wplus, wminus]
    #return [zmc]
    #return [data]
    #return [wplus, wminus]

def buildXrdFileList(path, xrd):
    xrdpath = path[path.find('/store'):]
    logging.debug(f"Looking for path {xrdpath}")
    f = subprocess.check_output(['xrdfs', f'root://{xrd}', 'ls', xrdpath]).decode(sys.stdout.encoding)
    return filter(lambda x: "root" in x[-4:], f.split())

#data_files_ = buildXrdFileList("/store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/SingleMuon/NanoV8Data/Run2016G_210302_203023/0000", "root://cmseos.cern.ch")[:10]
#zmc_files_ = buildXrdFileList("/store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/MCPostVFPWeightFix/211114_184608/0000/", "root://cmseos.cern.ch")[:10]
zmc_files_ = glob.glob("/scratch/shared/originalNANO/DYJetsToMuMu_postVFP/*/*.root")
wplus_files_ = glob.glob("/scratch/shared/originalNANO_newWithAltPDF/WplusJetsToMuNu_postVFP/*/*.root")
wminus_files_ = glob.glob("/scratch/shared/originalNANO_newWithAltPDF/WminusJetsToMuNu_postVFP/*/*.root")
dataPostF_files_ = glob.glob("/scratch/shared/originalNANO/NanoV8Data_Mar2021/Run2016F_postVFP/*")
dataG_files_ = glob.glob("/scratch/shared/originalNANO/NanoV8Data_Mar2021/Run2016G/*")
dataH_files_ = glob.glob("/scratch/shared/originalNANO/NanoV8Data_Mar2021/Run2016H/*")
data_files_ = dataG_files_+dataH_files_+dataPostF_files_
#data_files_ = 100*data_files_
