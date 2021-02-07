import csv, xlwt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

FILENAME = 'mt_k_y0.csv'
t = []
Mt = []

with open(FILENAME) as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        t.append(float(line[0]))
        # 残留要 1 -
        Mt.append(1 - float(line[1]) / 100)

t = np.array(t).reshape(len(t), 1)
Mt = np.array(Mt).reshape(len(Mt), 1)

class model(tf.keras.Model):

    def __init__(self):
        super(model, self).__init__()

        self.a = tf.Variable(tf.random.uniform([1], minval=-1, maxval=0))
        self.y0 = tf.Variable(tf.random.uniform([1], minval=0, maxval=1))

    def call(self, t):
        f = self.y0 + self.a * tf.math.log(t)
        return tf.math.sigmoid(f)

    def paras(self):
        print('a = {}'.format(self.a))
        print('y0 = {}'.format(self.y0))
        return self.a, self.y0

m = model()

m.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
)

m.fit(t, Mt, epochs=400)
a, y0 = m.paras()

NUM = 1000
x = np.linspace(min(t), max(t), NUM)
y = a * np.log(x) + y0
y = tf.math.sigmoid(y)
y = y.numpy()

f = xlwt.Workbook()
sheet1 = f.add_sheet('log', cell_overwrite_ok=True)
sheet1.write(0, 0, 'x')
sheet1.write(0, 1, 'y')

for i in range(NUM):
    sheet1.write(i + 1, 0, x[i, 0])
    sheet1.write(i + 1, 1, str(y[i, 0]))

f.save('log.xls')

plt.scatter(t, Mt)
plt.plot(x, y)
plt.show()