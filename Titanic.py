import tflearn
from tflearn.datasets import titanic
from tflearn.data_utils import load_csv


titanic.download_dataset('titanic_dataset.csv')
data, labels = load_csv('titanic_dataset.csv', target_column = 0, categorical_labels = True,
                        n_classes = 2, columns_to_ignore = [2, 7])

for p in data:
    if p[1] == 'female':
        p[1] = 1
    else:
        p[1] = 0


def avcost():
    sum = 0
    count = 0
    for g in data:
        sum += float(g[5])
        count += 1
    return str(sum/count)[:5]


net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(data, labels, n_epoch=50, batch_size=16, show_metric=True)

Jack = [1, 'Aagrim Hoysal', 'male', 12, 1, 2, 'N/A', 10.000]

def proccess(lis):
    ans = []
    x = ''
    for i in lis:
        if  isinstance(i, str) == True:
            if i == 'male':
                x = 0
                ans.append(x)
            elif i == 'female':
                x = 1
                ans.append(x)
            else:
                pass
        else:
            ans.append(i)
    return [ans]


r = model.predict(proccess(Jack))
r = r[0][1]
print 'Your chance of surviving: ' + str(r*100)[:5]
print 'Average chance of surviving: 32.36'
print 'You have a higher chance of surviving than most others!' if float(str(r*100)[:5]) >= 32.36 else 'You have less of a chance of surviving the Titanic than others.'
print
print 'Average Ticket Price: $' + avcost() + ', or $' + str(float(str(r*100)[:5])*23.45) + ' ajusting for inflation.'
print