import math
import pickle
from PIL import Image
import glob
import time

def correlogramDifference(cg1, cg2):
    sum = 0
    for c in range(0, 64):
        for d in range(0,5):
            sum = sum + (abs(cg1[c][d]- cg2[c][d])/((1 + cg1[c][d] + cg2[c][d])))
    return sum

def  returnSimilarImages(query, autocorrelograms):
    queryac = autocorrelograms[query]
    ranker = {}
    differences = [] 
    for cgs in autocorrelograms.keys():
        diff = correlogramDifference(queryac, autocorrelograms[cgs])
        differences.append(diff)
        ranker[diff] =  cgs

    differences.sort(reverse = False)
    similarImages = []
    for i in range(0,100):
        image = ranker[differences[i]]
        similarImages.append(image)
    return similarImages

class Query:
    
    def __init__(self, queryName):
        self.queryName = queryName
        self.queryLocation = "HW1/train/query/"+queryName + "_query.txt"
        self.gtgoodLocation = "HW1/train/ground_truth/" + queryName + "_good.txt"
        self.gtjunkLocation = "HW1/train/ground_truth/" + queryName + "_junk.txt"
        self.gtokLocation = "HW1/train/ground_truth/" + queryName + "_ok.txt"
        self.readQueryFile()
        self.readGTs()
    
    def readQueryFile(self):
        # print(queries[0])
        file = open(self.queryLocation, 'r')
        self.query = file.readline()
        li = self.query.split(" ")
        self.queryImage = li[0]
        self.queryImage = (self.queryImage[5:])

    def readGTs(self):
        File = open(self.gtgoodLocation, 'r')
        self.good = []
        for line in File:
            self.good.append(line[:-1])
        
        File = open(self.gtokLocation , 'r')
        self.ok = []
        for line in File:
            self.ok.append(line[:-1])
        
        File = open(self.gtjunkLocation, 'r')
        self.junk = []
        for line in File:
            self.junk.append(line[:-1])
    
    def returnQueryImage(self):
        return self.queryImage

    def runQuerySuccess(self, li):
        goodCount = 0
        okCount = 0
        junkCount = 0
        uhhhCount = 0
        for l in li:
            if l in self.good:
                goodCount+=1
            elif l in self.ok:
                okCount+=1
            elif l in self.junk:
                junkCount+=1
            else:
                uhhhCount+=1
        
        print(self.queryImage)
        print("Good Count  : " + str(goodCount))
        print("OK Count    : " + str(okCount))
        print("Junk Count  : " + str(junkCount))
        print("Uhhh Count  : " + str(uhhhCount))

        precision = (goodCount+okCount+junkCount)/(len(li))
        recall = (goodCount+okCount+junkCount)/(len(self.good) +len(self.ok) +  len(self.junk))
        F1S = (2*precision*recall)/(precision+recall)
        return([self.queryName, precision*100, recall*100, F1S ,goodCount, okCount, junkCount])

queryFolder ="HW1/train/query/"
imageFolder ="HW1/images\\"

queries = (glob.glob(queryFolder +  "*.txt"))
pickleFile = open("ac.p", "rb")
autocorrelograms = pickle.load(pickleFile)

workbook = Workbook()
sheet = workbook.active

sheet['A1'] = 'Query' 
sheet['B1'] = 'Precision' 
sheet['C1'] = 'Recall'
sheet['D1'] = 'F1-Score'
sheet['E1'] = 'Good' 
sheet['F1'] = 'Ok' 
sheet['G1'] = 'Junk'
sheet['H1'] = 'Time' 

results = []
for i in range(len(queries)):    
    start = time.time()
    queries[i] = queries[i][len(queryFolder):-10]
    q = Query(queries[i])
    im =q.returnQueryImage()
    similarQueryImages = returnSimilarImages(im, autocorrelograms)
    res = q.runQuerySuccess(similarQueryImages)
    end = time.time()
    timeElapsed = (end - start)
    res.append(timeElapsed)
    results.append(res)
    print(res)

for i in range(len(results)):
    sheet['A' + str(i+2)] = results[i][0]
    sheet['B' + str(i+2)] = results[i][1]
    sheet['C' + str(i+2)] = results[i][2]
    sheet['D' + str(i+2)] = results[i][3]
    sheet['E' + str(i+2)] = results[i][4] 
    sheet['F' + str(i+2)] = results[i][5] 
    sheet['G' + str(i+2)] = results[i][6]
    sheet['H' + str(i+2)] = results[i][7]

workbook.save(filename = "autocsss.xlsx")
