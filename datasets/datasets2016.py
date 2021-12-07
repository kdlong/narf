import narf 
import logging
import subprocess
import glob

lumicsv = "data/bylsoutput.csv"
lumijson = "data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"

def allDatasets():
    data = narf.Dataset(name = "dataPostVFP",
                        filepaths = data_files_,
                        is_data = True,
                        lumi_csv = lumicsv,
                        lumi_json = lumijson)

    zmc = narf.Dataset(name = "ZmumuPostVFP",
                    filepaths = zmc_files_,
                    is_data = False,
                    xsec = 1990.5,
                    )
                    

    return [data, zmc]


def buildXrdFileList(path, xrd):
    xrdpath = path[path.find('/store'):]
    logging.debug(f"Looking for path {xrdpath}")
    f = subprocess.check_output(['xrdfs', f'root://{xrd}', 'ls', xrdpath]).decode(sys.stdout.encoding)
    return filter(lambda x: "root" in x[-4:], f.split())

#data_files_ = buildXrdFileList("/store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/SingleMuon/NanoV8Data/Run2016G_210302_203023/0000", "root://cmseos.cern.ch")[:10]
#zmc_files_ = buildXrdFileList("/store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/MCPostVFPWeightFix/211114_184608/0000/", "root://cmseos.cern.ch")[:10]
zmc_files_ = glob.glob("/scratch/shared/originalNANO/DYJetsToMuMu_postVFP/*/*.root")[:200]
data_files_ = glob.glob("/scratch/shared/originalNANO/NanoV8Data_Mar2021/Run2016H/*")[:200]
