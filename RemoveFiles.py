import os

NN_name = 'ResNet v1'
files = os.listdir('NNs/'+NN_name)
ages = [int(f[len(NN_name)+2:-3]) for f in files]
maxAge = max(ages)
for i in ages:
    if i%50 != 0 and i < maxAge:
        os.remove('NNs/' + NN_name + '/' + NN_name + ', ' + str(i) + '.h5')
print('done')