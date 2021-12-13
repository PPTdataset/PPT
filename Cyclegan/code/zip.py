import zipfile
import os

def zipdir(path, file):
    z = zipfile.ZipFile(file,'w',zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(path):
        fpath = dirpath.replace(path,'') 
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
    z.close()
    
zipdir('../temp_data/result/', '../result/data.zip')