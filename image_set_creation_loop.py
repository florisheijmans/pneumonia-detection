import os
from random import randrange
from shutil import copyfile

src_rootdir_train = '/Users/euan/documenten/studie/blok2.2/pattern_recognition/project/CellData/chest_xray/train'
src_rootdir_test = '/Users/euan/documenten/studie/blok2.2/pattern_recognition/project/CellData/chest_xray/test'

dstdir_val = '/Users/euan/documenten/studie/blok2.2/pattern_recognition/project/CellData/chest_xray/new_files/val'
dstdir_train = '/Users/euan/documenten/studie/blok2.2/pattern_recognition/project/CellData/chest_xray/new_files/train'
dstdir_test = '/Users/euan/documenten/studie/blok2.2/pattern_recognition/project/CellData/chest_xray/new_files/test'

# Create test set
os.chdir(src_rootdir_test)
for dirs, subdir, files in os.walk(src_rootdir_test):
    os.chdir(dirs)
    print(f"dirs = {dirs}")

    if ('NORMAL' in dirs):
        for file in files:
            if ("NORMAL" in file):
                cur_dstdir = os.path.join(dstdir_test, 'NORMAL', file)
                copyfile(file, cur_dstdir)
            else:
                print(f"{file} doesn't belong to any of the predefined cases of: 'NORMAL'.")

    if ('PNEUMONIA' in dirs):
        for file in files:
            if ("BACTERIA" in file):
                cur_dstdir = os.path.join(dstdir_test, 'BACTERIA', file)
                copyfile(file, cur_dstdir)
            elif ("VIRUS" in file):
                cur_dstdir = os.path.join(dstdir_test, 'VIRUS', file)
                copyfile(file, cur_dstdir)
            else:
                print(f"{file} doesn't belong to any of the predefined cases of: 'BACTERIA', or 'VIRUS'.")

# Create train and validation set
os.chdir(src_rootdir_train)
for dirs, subdir, files in os.walk(src_rootdir_train):
    os.chdir(dirs)
    print(f"dirs = {dirs}")

    if ('NORMAL' in dirs):    
        print(f"case is NORMAL, with {int(0.10*len(files))} images, of total images: {len(files)}")
        coppied_files = []
        for i in range(int(0.10*len(files))):
            cur_file = randrange(len(files))
            while cur_file in coppied_files:
                cur_file = randrange(len(files))
            coppied_files.append(cur_file)

        for cur_file in range(len(files)):
            file = files[cur_file]
            #print(f"cur_file = {file}")

            if cur_file in coppied_files:
                if ("NORMAL" in file):
                    cur_dstdir = os.path.join(dstdir_val, 'NORMAL', file)
                    copyfile(file, cur_dstdir)
                else:
                    print(f"{file} doesn't belong to any of the predefined cases of: 'NORMAL'.")
            else:
                if ("NORMAL" in file):
                    cur_dstdir = os.path.join(dstdir_train, 'NORMAL', file)
                    copyfile(file, cur_dstdir)
                else:
                    print(f"{file} doesn't belong to any of the predefined cases of: 'NORMAL'.")

    if ('PNEUMONIA' in dirs):
        bact = 0
        virus = 0
        for file in files:
            if 'BACTERIA' in file:
                bact += 1
            if 'VIRUS' in file:
                virus += 1

        count_pneum = [virus, bact]
        pneum = ['BACTERIA', 'VIRUS'] # inverse on purpose for bool in while of random array
        
        for j in range(2):
            print(f"case is not: {pneum[j]}, with {int(0.10*count_pneum[j])} images, of total images: {count_pneum[j]}")
            coppied_files = []
            for i in range(int(0.10*count_pneum[j])):
                cur_file = randrange(len(files))
                while (cur_file in coppied_files) or (pneum[j] in files[cur_file]):
                    cur_file = randrange(len(files))
                coppied_files.append(cur_file)

            
            for cur_file in range(len(files)):
                file = files[cur_file]
                #print(f"cur_file = {file}")

                if cur_file in coppied_files:
                    if ("BACTERIA" in file):
                        cur_dstdir = os.path.join(dstdir_val, 'BACTERIA', file)
                        copyfile(file, cur_dstdir)
                    elif ("VIRUS" in file):
                        cur_dstdir = os.path.join(dstdir_val, 'VIRUS', file)
                        copyfile(file, cur_dstdir)
                    else:
                        print(f"{file} doesn't belong to any of the predefined cases of: 'BACTERIA', or 'VIRUS'.")
                else:
                    if ("BACTERIA" in file):
                        cur_dstdir = os.path.join(dstdir_train, 'BACTERIA', file)
                        copyfile(file, cur_dstdir)
                    elif ("VIRUS" in file):
                        cur_dstdir = os.path.join(dstdir_train, 'VIRUS', file)
                        copyfile(file, cur_dstdir)
                    else:
                        print(f"{file} doesn't belong to any of the predefined cases of: 'BACTERIA', or 'VIRUS'.")
                    

rootdir = '/Users/euan/documenten/studie/blok2.2/pattern_recognition/project/CellData/chest_xray/train'
dstdir_train = '/Users/euan/documenten/studie/blok2.2/pattern_recognition/project/CellData/chest_xray/new_files/train'
os.chdir(rootdir)

