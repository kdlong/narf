import narf 

def allSampleGroups():
    data = narf.DataGroup(name = "dataPostVFP",
            label = "data (post-VFP)",
            members = ["dataPostVFP"],
            is_data = True,
            draw = {"color" : "#000000",
                "histtype" : "errorbar",
                "marker" : "o",
                "markersize" : 5,
            },
        )

    zmc = narf.DataGroup(name = "ZmumuPostVFP",
            label = r"Z\to\mu\mu",
            members = ["ZmumuPostVFP"],
            draw = {"color" : "#ADD8E6",
                "histtype" : "fill",
            },
        )

    return {x.name : x for x in [zmc, data]}
