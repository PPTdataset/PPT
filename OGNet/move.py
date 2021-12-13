import os
import shutil

'''alllist=os.listdir("PPT")
freelist=os.listdir("../TC_img_gt")

for fn in alllist:
    if fn in freelist:
        print("inline"+fn)
        shutil.move("PPT/"+fn, "data/test/0/"+fn)
    else:
        print("outline"+fn)
        shutil.move("PPT/"+fn, "data/test/1"+fn)'''
#print(len(os.listdir("data/test/")))
num=0
for fn in os.listdir("../PPT_trainset_180k"):
    if num>=5000:
        break
    shutil.copy("../PPT_trainset_180k/"+fn,"data/train/0/sub/"+fn)
    print(str(num)+"  "+fn)
    num+=1

    