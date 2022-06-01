import zipfile
import os, os.path
import argparse


def zipdir3(dirname, zipfilename):
    filelist = []
    if os.path.isfile(dirname):
        filelist.append(dirname)
    else :
        for root, dirs, files in os.walk(dirname):
            for name in files:
                filelist.append(os.path.join(root, name))
    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:
        arcname = tar[len(dirname):]
        #print arcname
        zf.write(tar,arcname)
    zf.close()

def zipdir2(dirpath, outFullName):
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        fpath = path.replace(dirpath, '')
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()

def zipdir(path, file):
    z = zipfile.ZipFile(file,'w',zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(path):
        fpath = dirpath.replace(path,'') 
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
    z.close()

def Unzip(zipfilename, unziptodir):
    if not os.path.exists(unziptodir): os.makedirs(unziptodir)
    zfobj = zipfile.ZipFile(zipfilename)
    for name in zfobj.namelist():
        name = name.replace('\\','/')
        if name.endswith('/'):
            os.mkdir(os.path.join(unziptodir, name))
        else:
            ext_filename = os.path.join(unziptodir, name)
            ext_dir= os.path.dirname(ext_filename)
            if not os.path.exists(ext_dir) : os.makedirs(ext_dir)
            outfile = open(ext_filename, 'wb')
            outfile.write(zfobj.read(name))
            outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, default="../temp_data/result/", help='path to files')
    parser.add_argument('--output_dir', type=str, default="../result/data.zip", help='path to saves')
    opt = parser.parse_args()

    zipdir3(opt.input_dir, opt.output_dir)
    # Unzip(opt.output_dir, "../result/")
    # zipdir('../temp_data/result/', '../result/data.zip')



