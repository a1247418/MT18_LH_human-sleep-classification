# Use the following to get rid of files with the old filter
# find -type f -regextype posix-extended ! -iregex '.*_newfilt\.mat$' -delete
#
# This script is used to unify the naming convention of the sleep recording files
import os

data_dir = "/cluster/scratch/llorenz/data/caro_new/"

files = [f for f in os.listdir(data_dir) if
             os.path.isfile(os.path.join(data_dir, f))]

cnt = 0
for file in files:
    if not file.startswith("WESA"):
        if "_corr" in file:
            # Use corrected files
            os.rename(data_dir + file, data_dir + file.replace("_corr", ""))
            file = file.replace("_corr","")
        if "V_ML" in file:
            # unify in verum naming scheme
            os.rename(data_dir + file, data_dir + file.replace("V_ML", "_ML"))
            file = file.replace("V_ML","_ML")
        new_name = "_".join(prt for prt in file.split("_") if not "msco" in prt)
        new_name = "WESA_" + new_name
        new_name = new_name.replace("_ML", "_N1S_ML" if "msco1" in file else "_N2S_ML")
        new_name = new_name.replace("_newfilt", "")

        os.rename(data_dir+file, data_dir+new_name)
        print(file,"--->",new_name)
        cnt += 1
    else:
        new_name = file.replace("_newfilt", "")
        os.rename(data_dir+file, data_dir+new_name)
        print(file,"--->",new_name)
        cnt += 1


print("Done.", f"Renamed {cnt} files.")
