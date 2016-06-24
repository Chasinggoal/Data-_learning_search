import csv
import random

##****** creates a csv file of test data given the number of queries, objects per query, and features per object


def header(featureNum):
    head = ['Object', 'qid']
    for i in range(1, featureNum + 1):
        head.append('feature ' + str(i))
    return head

def featureType(featureNum):
    #returns list where each entry is a random feature type, 0 = binary, 1 = continuous
    featType = []
    for i in range(featureNum):
        rand = random.randint(0,100)
        if rand > 25:
            featType.append(1)
        else:
            featType.append(0)
    return featType
    

def create(queryNum, objectsNum, featureNum):

    #create file to write in
    with open('data.csv', 'wb') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        
        i = 1 #doc-id
        types = featureType(featureNum)

        #write in header
        wr.writerow(header(featureNum))

        #write in feature data
        for q in range(1, queryNum + 1):
            for o in range(objectsNum):
                info = [i, q]
                prob = random.randint(0,100)/float(100)
                for f in range(featureNum):
                    if types[f] == 0:
                        info.append(random.randint(0,1))
                    else:
                        info.append(random.randint(0,1000)/float(1000))
                    f += 1
                i += 1
                #print info
                wr.writerow(info)
                o += 1
            q += 1

#ask user to input # queries, #objects/query, # feature/object
queryNum = input("Enter number of queries: ")
objectsNum = input("Enter number of objects per query: ")
featureNum = input("Enter number of features per object: ")
create(queryNum, objectsNum, featureNum)

