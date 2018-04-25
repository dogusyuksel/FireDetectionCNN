import glob, os

def rename(dir, pattern, titlePattern, startIdx):
    counter = startIdx
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename, 
                  os.path.join(dir, titlePattern+'.'+str(counter)+'.jpg'))
        counter = counter + 1
        
rename(r'\dataset\training_set\nonfire', r'*.jpg', r'fire', 1)