import csv

csvf = input('Enter a csv file to parse: ')
with open(csvf, 'rb') as csvfile:
    f = open("Output.txt", "w")
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    rownum = 0 
    for row in spamreader:
        if rownum != 0:
            for col in row:
                s = col.split(',')
                rank = s[len(s)-1] 
                qid = s[1]
                docid = s[0]
                f.write(rank + ' qid:' + qid + ' ')
                for i in range(2,len(s)-2):
                    f.write(str(i-1) + ':' + s[i] + ' ')
                f.write('#docid = ' + docid + '\n')
        rownum += 1
    f.close()
